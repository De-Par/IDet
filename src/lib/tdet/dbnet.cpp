#include "dbnet.h"

#include "nms.h"
#include "opencv_headers.h"
#include "ort_headers.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

// 0 => optimized & accurate (contours on low-res prob_map, then scale to orig)
// 1 => legacy behavior (upsample prob_map to orig, then contours)
#define DBNET_POSTPROCESS_UPSAMPLE 0

namespace {

struct BindingCtx {
    Ort::IoBinding io;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<float> in_buf;
    std::vector<float> out_buf;
    std::vector<int64_t> in_shape;
    std::vector<int64_t> out_shape;

    int curW = 0;
    int curH = 0;
    int curOW = 0;
    int curOH = 0;
    bool bound = false;

    Ort::Value in_tensor{nullptr};
    Ort::Value out_tensor{nullptr};

    explicit BindingCtx(Ort::Session& s) : io(s) {}
};

struct OrtEnvHolder {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dbnet"};
};

} // namespace

struct DBNet::Impl {
    OrtEnvHolder env_holder;
    Ort::Session session{nullptr};
    Ort::SessionOptions so;
    Ort::AllocatorWithDefaultOptions alloc;
    std::string in_name;
    std::string out_name;
    std::vector<std::unique_ptr<BindingCtx>> pool;

    Impl(const std::string& model_path, const tdet::TextDetectorConfig& cfg, bool verbose);
};

static inline float sigmoidf_stable(float x) noexcept {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        const float z = std::exp(x);
        return z / (1.0f + z);
    }
}

static inline int align_down32_safe(int v) noexcept {
    v = std::max(32, v);
    return v & ~31;
}

static inline cv::Mat ensure_bgr8(const cv::Mat& img) {
    if (img.empty()) return img;
    if (img.type() == CV_8UC3) return img;

    cv::Mat bgr;
    if (img.channels() == 4) {
        cv::cvtColor(img, bgr, cv::COLOR_BGRA2BGR);
    } else if (img.channels() == 1) {
        cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    } else {
        throw std::runtime_error("[ERROR] Unsupported image format: expected 1/3/4 channels");
    }
    return bgr;
}

static inline cv::Point2f clamp_pt(cv::Point2f p, const int W, const int H) {
    if (p.x < 0.f)
        p.x = 0.f;
    else if (p.x > (float)(W - 1))
        p.x = (float)(W - 1);

    if (p.y < 0.f)
        p.y = 0.f;
    else if (p.y > (float)(H - 1))
        p.y = (float)(H - 1);

    return p;
}

// Limit expansion so corners remain inside image (safe unclip)
static float safe_scale_for_rect(const cv::RotatedRect& rr, const int W, const int H, const float desired) {
    cv::Point2f c = rr.center, q[4];
    rr.points(q);
    float smax = desired;

    auto upd = [&](const float ccoord, const float v, const float lo, const float hi) {
        if (v > 1e-6f)
            smax = std::min(smax, (hi - ccoord) / v);
        else if (v < -1e-6f)
            smax = std::min(smax, (lo - ccoord) / v);
    };

    for (int i = 0; i < 4; ++i) {
        cv::Point2f v = q[i] - c;
        upd(c.x, v.x, 0.f, (float)(W - 1));
        upd(c.y, v.y, 0.f, (float)(H - 1));
    }

    if (smax < 1.f) smax = 1.f;
    return std::min(desired, smax * 0.999f);
}

DBNet::Impl::Impl(const std::string& model_path, const tdet::TextDetectorConfig& cfg, bool verbose) {
    if (verbose) {
        std::cout << "[INFO] " << Ort::GetBuildInfoString() << "\n";
        std::vector<std::string> providers = Ort::GetAvailableProviders();
        std::cout << "[INFO] Available Execution Providers:\n";
        for (const auto& p : providers)
            std::cout << " - " << p << "\n";
        std::cout << "\n";
    }

    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (cfg.threads.ort_intra_threads < 0)
        so.SetIntraOpNumThreads(1);
    else if (cfg.threads.ort_intra_threads > 0)
        so.SetIntraOpNumThreads(cfg.threads.ort_intra_threads);

    if (cfg.threads.ort_inter_threads < 0)
        so.SetInterOpNumThreads(1);
    else if (cfg.threads.ort_inter_threads > 0)
        so.SetInterOpNumThreads(cfg.threads.ort_inter_threads);

    so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

#if USE_ACL
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ACL(so, /*fast_math=*/true));
#endif

    session = Ort::Session(env_holder.env, model_path.c_str(), so);

    Ort::AllocatedStringPtr in0 = session.GetInputNameAllocated(0, alloc);
    Ort::AllocatedStringPtr out0 = session.GetOutputNameAllocated(0, alloc);
    in_name = in0.get() ? in0.get() : std::string("input");
    out_name = out0.get() ? out0.get() : std::string("output");
}

DBNet::DBNet(const std::string& model_path, tdet::TextDetectorConfig cfg, bool verbose)
    : impl_(std::make_unique<Impl>(model_path, cfg, verbose)), cfg_(std::move(cfg)) {}

DBNet::~DBNet() = default;

void DBNet::ensure_pool_size(int n) {
    if (n <= 0) n = 1;
    if ((int)impl_->pool.size() >= n) return;

    const int old = (int)impl_->pool.size();
    impl_->pool.resize(n);
    for (int i = old; i < n; ++i)
        impl_->pool[i] = std::make_unique<BindingCtx>(impl_->session);
}

void DBNet::preprocess_dynamic(const cv::Mat& img_bgr, cv::Mat& resized, cv::Mat& blob) const {
    cv::Mat bgr = ensure_bgr8(img_bgr);
    if (bgr.empty()) throw std::runtime_error("[ERROR] preprocess_dynamic: empty image");

    const int h = bgr.rows, w = bgr.cols;
    float scale = 1.0f;

    if (cfg_.infer.limit_side_len > 0) {
        const int max_side = std::max(h, w);
        if (max_side > cfg_.infer.limit_side_len) scale = (float)cfg_.infer.limit_side_len / (float)max_side;
    }

    int nh = std::max(1, (int)std::lround(h * scale));
    int nw = std::max(1, (int)std::lround(w * scale));

    // safe align-down to 32 (never becomes 0)
    nh = align_down32_safe(nh);
    nw = align_down32_safe(nw);

    // For text detection, INTER_LINEAR is a safe default (often better recall than AREA)
    cv::resize(bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    blob.create(1, 3 * nh * nw, CV_32F);
    float* dst = (float*)blob.data;
    const int stride = nh * nw;
    constexpr float s = 1.0f / 255.0f;

    for (int y = 0; y < nh; ++y) {
        const uchar* p = resized.ptr<uchar>(y);
        for (int x = 0; x < nw; ++x) {
            const float B = p[0] * s, G = p[1] * s, R = p[2] * s;
            const int idx = y * nw + x;
            dst[idx] = R;
            dst[stride + idx] = G;
            dst[2 * stride + idx] = B;
            p += 3;
        }
    }
}

void DBNet::preprocess_fixed_into(float* dst_chw, const cv::Mat& img_bgr, const int W, const int H) const {
    cv::Mat bgr = ensure_bgr8(img_bgr);
    if (bgr.empty()) throw std::runtime_error("[ERROR] preprocess_fixed_into: empty image");
    if (W <= 0 || H <= 0) throw std::runtime_error("[ERROR] preprocess_fixed_into: non-positive target size");

    thread_local cv::Mat resized;
    resized.create(H, W, CV_8UC3);

    // Keep LINEAR for better text recall
    cv::resize(bgr, resized, resized.size(), 0, 0, cv::INTER_LINEAR);

    const int stride = W * H;
    constexpr float s = 1.0f / 255.0f;

    for (int y = 0; y < H; ++y) {
        const uchar* p = resized.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            const float B = p[0] * s, G = p[1] * s, R = p[2] * s;
            const int idx = y * W + x;
            dst_chw[idx] = R;
            dst_chw[stride + idx] = G;
            dst_chw[2 * stride + idx] = B;
            p += 3;
        }
    }
}

void DBNet::prepare_binding(int ctx_idx, const int W, const int H) {
    if (ctx_idx < 0 || ctx_idx >= (int)impl_->pool.size()) {
        throw std::runtime_error("[ERROR] prepare_binding: ctx_idx out of range");
    }

    auto& ctx = *impl_->pool[ctx_idx];
    if (ctx.bound && ctx.curW == W && ctx.curH == H) return;
    if (W <= 0 || H <= 0) throw std::runtime_error("[ERROR] prepare_binding: non-positive W/H");

    ctx.in_shape = {1, 3, H, W};

    // Probe run once to discover output shape (works for models with dynamic output)
    const size_t inN = (size_t)3 * (size_t)H * (size_t)W;
    ctx.in_buf.assign(inN, 0.0f);

    Ort::Value in_tmp = Ort::Value::CreateTensor<float>(ctx.mem, ctx.in_buf.data(), ctx.in_buf.size(),
                                                        ctx.in_shape.data(), ctx.in_shape.size());

    const char* in_names[] = {impl_->in_name.c_str()};
    const char* out_names[] = {impl_->out_name.c_str()};

    auto outs = impl_->session.Run(Ort::RunOptions{nullptr}, in_names, &in_tmp, 1, out_names, 1);
    if (outs.size() != 1 || !outs[0].IsTensor()) {
        throw std::runtime_error("[ERROR] Unexpected output in probe(binding)");
    }

    auto ti = outs[0].GetTensorTypeAndShapeInfo();
    ctx.out_shape = ti.GetShape();
    const size_t outN = ti.GetElementCount();

    if (ctx.out_shape.size() < 2) throw std::runtime_error("[ERROR] Output rank < 2 in probe(binding)");

    ctx.curOH = (int)ctx.out_shape[ctx.out_shape.size() - 2];
    ctx.curOW = (int)ctx.out_shape[ctx.out_shape.size() - 1];
    if (ctx.curOH <= 0 || ctx.curOW <= 0) throw std::runtime_error("[ERROR] Non-positive OH/OW in probe(binding)");

    ctx.out_buf.resize(outN);

    ctx.io.ClearBoundInputs();
    ctx.io.ClearBoundOutputs();

    ctx.in_tensor = Ort::Value::CreateTensor<float>(ctx.mem, ctx.in_buf.data(), ctx.in_buf.size(), ctx.in_shape.data(),
                                                    ctx.in_shape.size());

    ctx.out_tensor = Ort::Value::CreateTensor<float>(ctx.mem, ctx.out_buf.data(), ctx.out_buf.size(),
                                                     ctx.out_shape.data(), ctx.out_shape.size());

    ctx.io.BindInput(impl_->in_name.c_str(), ctx.in_tensor);
    ctx.io.BindOutput(impl_->out_name.c_str(), ctx.out_tensor);

    ctx.curW = W;
    ctx.curH = H;
    ctx.bound = true;
}

std::vector<Detection> DBNet::postprocess(const cv::Mat& prob_map, const ImageSize& orig) const {
    if (prob_map.empty()) return {};
    if (prob_map.type() != CV_32F) throw std::runtime_error("[ERROR] postprocess: prob_map must be CV_32F");
    if (orig.width <= 0 || orig.height <= 0) return {};

#if DBNET_POSTPROCESS_UPSAMPLE
    // Legacy (slower): upsample prob_map to orig then contours
    cv::Mat prob_up;
    cv::resize(prob_map, prob_up, orig, 0, 0, cv::INTER_LINEAR);

    if (cfg_.infer.apply_sigmoid) {
        for (int y = 0; y < prob_up.rows; ++y) {
            float* row = prob_up.ptr<float>(y);
            for (int x = 0; x < prob_up.cols; ++x)
                row[x] = sigmoidf_stable(row[x]);
        }
    }

    cv::Mat bin;
    cv::compare(prob_up, cfg_.infer.bin_thresh, bin, cv::CMP_GT);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    const int W = orig.width, H = orig.height;
    std::vector<Detection> dets;
    dets.reserve(contours.size());

    for (auto& c : contours) {
        if ((int)c.size() < 3) continue;

        const float sc = contour_score(prob_up, c);
        if (sc < cfg_.infer.box_thresh) continue;

        cv::RotatedRect rr = cv::minAreaRect(c);
        const float s = safe_scale_for_rect(rr, W, H, cfg_.infer.unclip);
        rr.size.width *= s;
        rr.size.height *= s;

        cv::Point2f q[4];
        rr.points(q);
        order_quad(q);
        for (int k = 0; k < 4; ++k)
            q[k] = clamp_pt(q[k], W, H);

        const float dw = std::hypot(q[1].x - q[0].x, q[1].y - q[0].y);
        const float dh = std::hypot(q[3].x - q[0].x, q[3].y - q[0].y);
        if (std::min(dw, dh) < cfg_.min_text_size) continue;

        Detection d;
        d.score = sc;
        for (int k = 0; k < 4; ++k)
            d.pts[k] = q[k];
        dets.push_back(d);
    }
    return dets;
#else
    // Optimized & accurate: contours on low-res prob_map, then scale contour points to orig
    const int ow = prob_map.cols;
    const int oh = prob_map.rows;
    const float sx = (float)orig.width / (float)std::max(1, ow);
    const float sy = (float)orig.height / (float)std::max(1, oh);

    // Use a working prob map if sigmoid is needed (do NOT mutate prob_map view into ORT buffers)
    const cv::Mat* pprob = &prob_map;
    thread_local cv::Mat prob_work;
    if (cfg_.infer.apply_sigmoid) {
        prob_work.create(prob_map.size(), CV_32F);
        prob_map.copyTo(prob_work);
        for (int y = 0; y < prob_work.rows; ++y) {
            float* row = prob_work.ptr<float>(y);
            for (int x = 0; x < prob_work.cols; ++x)
                row[x] = sigmoidf_stable(row[x]);
        }
        pprob = &prob_work;
    }

    thread_local cv::Mat bin;
    cv::compare(*pprob, cfg_.infer.bin_thresh, bin, cv::CMP_GT); // 0/255, CV_8U

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    const int W = orig.width, H = orig.height;
    std::vector<Detection> dets;
    dets.reserve(contours.size());

    thread_local std::vector<cv::Point2f> scaled;
    for (auto& c : contours) {
        if ((int)c.size() < 3) continue;

        const float sc = contour_score(*pprob, c);
        if (sc < cfg_.infer.box_thresh) continue;

        // scale contour points to original image coordinates
        scaled.clear();
        scaled.reserve(c.size());
        for (const auto& p : c) {
            scaled.emplace_back(p.x * sx, p.y * sy);
        }

        cv::RotatedRect rr = cv::minAreaRect(scaled);

        const float s = safe_scale_for_rect(rr, W, H, cfg_.infer.unclip);
        rr.size.width *= s;
        rr.size.height *= s;

        cv::Point2f q[4];
        rr.points(q);
        order_quad(q);
        for (int k = 0; k < 4; ++k)
            q[k] = clamp_pt(q[k], W, H);

        const float dw = std::hypot(q[1].x - q[0].x, q[1].y - q[0].y);
        const float dh = std::hypot(q[3].x - q[0].x, q[3].y - q[0].y);
        if (std::min(dw, dh) < cfg_.min_text_size) continue;

        Detection d;
        d.score = sc;
        for (int k = 0; k < 4; ++k)
            d.pts[k] = q[k];
        dets.push_back(d);
    }

    return dets;
#endif
}

std::vector<Detection> DBNet::infer_unbound(const cv::Mat& img_bgr, double* ms_out) {
    if (img_bgr.empty()) throw std::runtime_error("[ERROR] infer_unbound: empty image");

    static Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Timer t;
    t.tic();

    // If fixed_W/H are set => fixed path, else dynamic path (best recall for full-frame)
    if (cfg_.infer.fixed_W > 0 && cfg_.infer.fixed_H > 0) {
        const int W = cfg_.infer.fixed_W;
        const int H = cfg_.infer.fixed_H;

        thread_local std::vector<float> blob;
        const size_t inN = (size_t)3 * (size_t)W * (size_t)H;
        blob.resize(inN);

        preprocess_fixed_into(blob.data(), img_bgr, W, H);

        const std::vector<int64_t> ishape = {1, 3, H, W};
        Ort::Value in =
            Ort::Value::CreateTensor<float>(cpu_mem, blob.data(), blob.size(), ishape.data(), ishape.size());

        const char* in_names[] = {impl_->in_name.c_str()};
        const char* out_names[] = {impl_->out_name.c_str()};

        auto out = impl_->session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
        if (ms_out) *ms_out = t.toc_ms();

        if (out.size() != 1 || !out[0].IsTensor()) throw std::runtime_error("[ERROR] Bad output");

        auto ti = out[0].GetTensorTypeAndShapeInfo();
        auto dims = ti.GetShape();

        int oc = 1, oh = 0, ow = 0;
        if (dims.size() == 4) {
            oc = (int)dims[1];
            oh = (int)dims[2];
            ow = (int)dims[3];
        } else if (dims.size() == 3) {
            oc = 1;
            oh = (int)dims[1];
            ow = (int)dims[2];
        } else if (dims.size() == 2) {
            oc = 1;
            oh = (int)dims[0];
            ow = (int)dims[1];
        } else
            throw std::runtime_error("[ERROR] Unsupported output rank");

        if (oc != 1) throw std::runtime_error("[ERROR] Expected single-channel output, got oc=" + std::to_string(oc));
        if (oh <= 0 || ow <= 0) throw std::runtime_error("[ERROR] Invalid output dims");

        const float* prob = out[0].GetTensorData<float>();
        cv::Mat prob_map(oh, ow, CV_32F, const_cast<float*>(prob));
        return postprocess(prob_map, ImageSize{img_bgr.cols, img_bgr.rows});
    }

    // Dynamic path (accurate for non-tiled full image)
    cv::Mat resized, blob;
    preprocess_dynamic(img_bgr, resized, blob);
    const int H = resized.rows;
    const int W = resized.cols;

    const size_t inN = (size_t)3 * (size_t)H * (size_t)W;
    const std::vector<int64_t> ishape = {1, 3, H, W};

    Ort::Value in = Ort::Value::CreateTensor<float>(cpu_mem, (float*)blob.data, inN, ishape.data(), ishape.size());

    const char* in_names[] = {impl_->in_name.c_str()};
    const char* out_names[] = {impl_->out_name.c_str()};

    auto out = impl_->session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    if (ms_out) *ms_out = t.toc_ms();

    if (out.size() != 1 || !out[0].IsTensor()) throw std::runtime_error("[ERROR] Bad output");

    auto ti = out[0].GetTensorTypeAndShapeInfo();
    auto dims = ti.GetShape();

    int oc = 1, oh = 0, ow = 0;
    if (dims.size() == 4) {
        oc = (int)dims[1];
        oh = (int)dims[2];
        ow = (int)dims[3];
    } else if (dims.size() == 3) {
        oc = 1;
        oh = (int)dims[1];
        ow = (int)dims[2];
    } else if (dims.size() == 2) {
        oc = 1;
        oh = (int)dims[0];
        ow = (int)dims[1];
    } else
        throw std::runtime_error("[ERROR] Unsupported output rank");

    if (oc != 1) throw std::runtime_error("[ERROR] Expected single-channel output, got oc=" + std::to_string(oc));
    if (oh <= 0 || ow <= 0) throw std::runtime_error("[ERROR] Invalid output dims");

    const float* prob = out[0].GetTensorData<float>();
    cv::Mat prob_map(oh, ow, CV_32F, const_cast<float*>(prob));
    return postprocess(prob_map, ImageSize{img_bgr.cols, img_bgr.rows});
}

std::vector<Detection> DBNet::infer_bound(const cv::Mat& img_bgr, int ctx_idx, double* ms_out) {
    if (img_bgr.empty()) throw std::runtime_error("[ERROR] infer_bound: empty image");
    if (ctx_idx < 0 || ctx_idx >= (int)impl_->pool.size()) {
        throw std::runtime_error("[ERROR] infer_bound: ctx_idx out of range (pool not prepared?)");
    }

    auto& ctx = *impl_->pool[ctx_idx];

    int W = cfg_.infer.fixed_W > 0 ? cfg_.infer.fixed_W : ((cfg_.infer.limit_side_len > 0) ? cfg_.infer.limit_side_len : 640);
    int H = cfg_.infer.fixed_H > 0 ? cfg_.infer.fixed_H : ((cfg_.infer.limit_side_len > 0) ? cfg_.infer.limit_side_len : 640);
    W = align_down32_safe(std::max(1, W));
    H = align_down32_safe(std::max(1, H));

    prepare_binding(ctx_idx, W, H);
    preprocess_fixed_into(ctx.in_buf.data(), img_bgr, W, H);

    Timer t;
    t.tic();
    impl_->session.Run(Ort::RunOptions{nullptr}, ctx.io);
    if (ms_out) *ms_out = t.toc_ms();

    const size_t rank = ctx.out_shape.size();
    if (rank < 2) throw std::runtime_error("[ERROR] Output rank < 2 (bound)");

    int oc = 1;
    const int oh = ctx.curOH;
    const int ow = ctx.curOW;

    if (rank == 4)
        oc = (int)ctx.out_shape[1];
    else if (rank == 3)
        oc = 1;
    else if (rank == 2)
        oc = 1;
    else
        throw std::runtime_error("[ERROR] Unsupported output rank(bound)");

    if (oc != 1)
        throw std::runtime_error("[ERROR] Expected single-channel output(bound), got oc=" + std::to_string(oc));
    if (oh <= 0 || ow <= 0) throw std::runtime_error("[ERROR] Invalid output dims(bound)");

    const float* plane = ctx.out_buf.data(); // channel 0
    cv::Mat prob_map(oh, ow, CV_32F, const_cast<float*>(plane));
    return postprocess(prob_map, ImageSize{img_bgr.cols, img_bgr.rows});
}
