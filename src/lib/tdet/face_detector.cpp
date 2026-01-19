#include "face_detector.h"

#include "geometry.h"
#include "internal/model_blob_factory.h"
#include "opencv_headers.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace {
inline Detection rect_to_detection(const cv::Rect2f& r, float score) {
    Detection d;
    d.score = score;
    d.pts[0] = cv::Point2f(r.x, r.y);
    d.pts[1] = cv::Point2f(r.x + r.width, r.y);
    d.pts[2] = cv::Point2f(r.x + r.width, r.y + r.height);
    d.pts[3] = cv::Point2f(r.x, r.y + r.height);
    return d;
}
} // namespace

FaceDetector::FaceDetector(tdet::FaceDetectorConfig cfg) : cfg_(std::move(cfg)) {
    so_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (cfg_.threads.ort_intra_threads > 0) so_.SetIntraOpNumThreads(cfg_.threads.ort_intra_threads);
    if (cfg_.threads.ort_inter_threads > 0) so_.SetInterOpNumThreads(cfg_.threads.ort_inter_threads);
    so_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    if (!cfg_.paths.model_path.empty()) {
        session_ = Ort::Session(env_, cfg_.paths.model_path.c_str(), so_);
    } else {
        ModelBlob blob = get_face_blob();
        if (!blob.data || blob.size == 0) {
            throw std::runtime_error("[ERROR] FaceDetector: empty model path and no embedded model");
        }
        session_ = Ort::Session(env_, blob.data, blob.size, so_);
    }

    Ort::AllocatedStringPtr in0 = session_.GetInputNameAllocated(0, alloc_);
    in_name_ = in0 ? in0.get() : std::string("input");

    // Expect two outputs: boxes and scores
    Ort::AllocatedStringPtr out0 = session_.GetOutputNameAllocated(0, alloc_);
    Ort::AllocatedStringPtr out1 = session_.GetOutputNameAllocated(1, alloc_);
    boxes_name_ = out0 ? out0.get() : std::string("boxes");
    scores_name_ = out1 ? out1.get() : std::string("scores");
}

void FaceDetector::preprocess(const cv::Mat& img_bgr, cv::Mat& resized, cv::Mat& blob, float& sx, float& sy,
                              int force_w, int force_h) const {
    cv::Mat bgr = img_bgr;
    if (bgr.empty()) throw std::runtime_error("[ERROR] FaceDetector: empty image");

    const int h = bgr.rows, w = bgr.cols;
    int target_w = force_w > 0 ? force_w : cfg_.infer.fixed_W;
    int target_h = force_h > 0 ? force_h : cfg_.infer.fixed_H;

    if (target_w <= 0 || target_h <= 0) {
        float scale = 1.0f;
        if (cfg_.infer.limit_side_len > 0) {
            const int max_side = std::max(h, w);
            if (max_side > cfg_.infer.limit_side_len) scale = (float)cfg_.infer.limit_side_len / (float)max_side;
        }
        target_w = std::max(1, (int)std::lround(w * scale));
        target_h = std::max(1, (int)std::lround(h * scale));
    }

    if (force_w <= 0 || force_h <= 0) {
        // Align to stride to avoid shape mismatch on some models (SCRFD is stride 32 max)
        const int align = 32;
        target_w = (target_w + align - 1) / align * align;
        target_h = (target_h + align - 1) / align * align;
    }

    cv::resize(bgr, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
    sx = (float)target_w / (float)w;
    sy = (float)target_h / (float)h;

    const int nh = resized.rows;
    const int nw = resized.cols;

    blob.create(1, 3 * nh * nw, CV_32F);
    float* dst = (float*)blob.data;
    const int stride = nh * nw;
    constexpr float mean = 127.5f;
    constexpr float inv_std = 1.0f / 128.0f;
    for (int y = 0; y < nh; ++y) {
        const uchar* p = resized.ptr<uchar>(y);
        for (int x = 0; x < nw; ++x) {
            const float B = (p[0] - mean) * inv_std;
            const float G = (p[1] - mean) * inv_std;
            const float R = (p[2] - mean) * inv_std;
            const int idx = y * nw + x;
            // SCRFD training uses BGR order, keep it here.
            dst[idx] = B;
            dst[stride + idx] = G;
            dst[2 * stride + idx] = R;
            p += 3;
        }
    }
}

std::vector<Detection> FaceDetector::postprocess(const float* boxes, const float* scores, size_t count,
                                                 const cv::Size& orig, float sx, float sy) const {
    std::vector<Detection> dets;
    dets.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        const float score = scores[i];
        if (score < cfg_.infer.box_thresh) continue;
        const float x1 = boxes[i * 4 + 0] / sx;
        const float y1 = boxes[i * 4 + 1] / sy;
        const float x2 = boxes[i * 4 + 2] / sx;
        const float y2 = boxes[i * 4 + 3] / sy;
        const cv::Rect2f r(x1, y1, std::max(0.f, x2 - x1), std::max(0.f, y2 - y1));
        if (r.width <= 0 || r.height <= 0) continue;

        // clamp
        cv::Rect2f rc = r & cv::Rect2f(0.f, 0.f, (float)orig.width, (float)orig.height);
        if (rc.width <= 1.f || rc.height <= 1.f) continue;

        dets.push_back(rect_to_detection(rc, score));
    }
    // simple score-sort
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    return dets;
}

static std::vector<Detection> decode_scrfd_heads(const std::unordered_map<std::string, Ort::Value>& outs,
                                                 const cv::Size& orig, float sx, float sy, float score_thr,
                                                 bool apply_sigmoid, int min_w, int min_h, bool verbose, int in_h,
                                                 int in_w) {
    struct Head {
        const float* score = nullptr;
        const float* bbox = nullptr;
        int h = 0, w = 0, stride = 0;
        bool score_chw = false;
        bool bbox_chw = false;
        int score_ch = 1;
        int anchors = 1;
        bool flat = false; // score/bbox flattened to [1, HW*A, C]
    };

    if (0) {
        for (const auto& kv : outs) {
            auto shp = kv.second.GetTensorTypeAndShapeInfo().GetShape();
            std::cerr << "[DEBUG] output " << kv.first << " shape=[";
            for (size_t i = 0; i < shp.size(); ++i) {
                std::cerr << shp[i];
                if (i + 1 < shp.size()) std::cerr << ",";
            }
            std::cerr << "]\n";
        }
    }

    auto head_from = [&](const std::string& sname, const std::string& bname, int stride) -> std::optional<Head> {
        auto s_it = outs.find(sname);
        auto b_it = outs.find(bname);
        if (s_it == outs.end() || b_it == outs.end()) return std::nullopt;

        auto sshape = s_it->second.GetTensorTypeAndShapeInfo().GetShape();
        auto bshape = b_it->second.GetTensorTypeAndShapeInfo().GetShape();

        Head h;
        h.score = s_it->second.GetTensorData<float>();
        h.bbox = b_it->second.GetTensorData<float>();
        h.stride = stride;
        h.h = std::max(1, in_h / stride);
        h.w = std::max(1, in_w / stride);

        if (sshape.size() == 4) { // [1,C,H,W]
            h.h = (int)sshape[2];
            h.w = (int)sshape[3];
            h.score_chw = true;
            h.score_ch = (int)std::max<int64_t>(1, sshape[1]);
        } else if (sshape.size() == 3 && sshape[2] == 1) { // [1, HW*A, 1]
            h.flat = true;
            h.score_ch = 1;
            int64_t total = sshape[1];
            const int hw = h.h * h.w;
            if (hw > 0 && total % hw == 0) h.anchors = (int)(total / hw);
        } else if (sshape.size() >= 3) { // [1,H,W,1] or [N,H,W]
            h.h = (int)sshape[1];
            h.w = (int)sshape[2];
            h.score_ch = (sshape.size() >= 4) ? (int)std::max<int64_t>(1, sshape[3]) : 1;
        } else {
            return std::nullopt;
        }

        if (bshape.size() == 4 && bshape[1] == 4) { // [1,4,H,W]
            h.bbox_chw = true;
        } else if (bshape.size() == 3 && bshape[0] == 1 && bshape[2] == 4) {
            h.bbox_chw = false; // [1,N,4]
            h.flat = true;
            const int hw = h.h * h.w;
            if (hw > 0 && bshape[1] % hw == 0) h.anchors = (int)(bshape[1] / hw);
        } else if (bshape.size() == 4 && bshape[3] == 4) {
            h.bbox_chw = false; // [1,H,W,4]
        } else {
            return std::nullopt;
        }
        return h;
    };

    std::vector<Head> heads;
    if (auto h = head_from("score_8", "bbox_8", 8)) heads.push_back(*h);
    if (auto h = head_from("score_16", "bbox_16", 16)) heads.push_back(*h);
    if (auto h = head_from("score_32", "bbox_32", 32)) heads.push_back(*h);

    if (0) {
        for (const auto& h : heads) {
            std::cerr << "[DEBUG] head stride=" << h.stride << " score_hw=" << h.h << "x" << h.w << " ch=" << h.score_ch
                      << " anchors=" << h.anchors << " flat=" << h.flat << " bbox_chw=" << h.bbox_chw << "\n";
        }
    }

    std::vector<Detection> dets;
    for (const auto& head : heads) {
        if (!head.score || !head.bbox || head.h <= 0 || head.w <= 0) continue;

        size_t passed = 0;
        float min_raw = 1e9f, max_raw = -1e9f;
        for (int y = 0; y < head.h; ++y) {
            for (int x = 0; x < head.w; ++x) {
                for (int a = 0; a < head.anchors; ++a) {
                    const int loc = (y * head.w + x) * head.anchors + a;

                    float score = 0.0f;
                    if (head.score_chw) {
                        const int stride_hw = head.h * head.w;
                        const int ch = (head.score_ch > 1) ? 1 : 0; // pick foreground if 2 channels
                        score = head.score[ch * stride_hw + (y * head.w + x)];
                    } else if (head.flat) {
                        const int ch_stride = head.score_ch;
                        const int ch = (head.score_ch > 1) ? 1 : 0;
                        score = head.score[loc * ch_stride + ch];
                    } else {
                        const int ch_stride = head.score_ch;
                        const int ch = (head.score_ch > 1) ? 1 : 0;
                        score = head.score[(y * head.w + x) * ch_stride + ch];
                    }
                    min_raw = std::min(min_raw, score);
                    max_raw = std::max(max_raw, score);
                    if (apply_sigmoid) score = 1.0f / (1.0f + std::exp(-score));
                    if (score < score_thr) continue;
                    ++passed;

                    float cx = (x + 0.5f) * head.stride;
                    float cy = (y + 0.5f) * head.stride;

                    float dl, dt, dr, db;
                    if (head.bbox_chw) {
                        const int stride_hw = head.h * head.w;
                        dl = head.bbox[0 * stride_hw + (y * head.w + x)] * head.stride;
                        dt = head.bbox[1 * stride_hw + (y * head.w + x)] * head.stride;
                        dr = head.bbox[2 * stride_hw + (y * head.w + x)] * head.stride;
                        db = head.bbox[3 * stride_hw + (y * head.w + x)] * head.stride;
                    } else if (head.flat) {
                        dl = head.bbox[loc * 4 + 0] * head.stride;
                        dt = head.bbox[loc * 4 + 1] * head.stride;
                        dr = head.bbox[loc * 4 + 2] * head.stride;
                        db = head.bbox[loc * 4 + 3] * head.stride;
                    } else {
                        const int idx = y * head.w + x;
                        dl = head.bbox[idx * 4 + 0] * head.stride;
                        dt = head.bbox[idx * 4 + 1] * head.stride;
                        dr = head.bbox[idx * 4 + 2] * head.stride;
                        db = head.bbox[idx * 4 + 3] * head.stride;
                    }

                    float x1 = (cx - dl) / sx;
                    float y1 = (cy - dt) / sy;
                    float x2 = (cx + dr) / sx;
                    float y2 = (cy + db) / sy;
                    if (x2 <= x1 || y2 <= y1) continue;

                    cv::Rect2f r(x1, y1, x2 - x1, y2 - y1);
                    r &= cv::Rect2f(0.f, 0.f, (float)orig.width, (float)orig.height);
                    if (r.width <= 1.f || r.height <= 1.f) continue;
                    if (r.width < min_w || r.height < min_h) continue;

                    dets.push_back(rect_to_detection(r, score));
                }
            }
        }
        if (0) {
            std::cerr << "[DEBUG] head stride=" << head.stride << " passed " << passed << " / "
                      << (head.h * head.w * head.anchors) << " (thr=" << score_thr << ") raw_min=" << min_raw
                      << " raw_max=" << max_raw << "\n";
        }
    }

    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    if (0) {
        for (size_t i = 0; i < dets.size() && i < 5; ++i) {
            const auto& d = dets[i];
            std::cerr << "[DEBUG] det" << i << " score=" << d.score << " tl=(" << d.pts[0].x << "," << d.pts[0].y
                      << ") br=(" << d.pts[2].x << "," << d.pts[2].y << ")\n";
        }
    }
    return dets;
}

std::vector<Detection> FaceDetector::detect(const cv::Mat& img_bgr, double* ms_out) {
    cv::Mat resized, blob;
    float sx = 1.0f, sy = 1.0f;
    preprocess(img_bgr, resized, blob, sx, sy);

    const int H = resized.rows;
    const int W = resized.cols;
    const size_t inN = (size_t)3 * (size_t)H * (size_t)W;
    const std::vector<int64_t> ishape = {1, 3, H, W};

    static Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(cpu_mem, (float*)blob.data, inN, ishape.data(), ishape.size());

    const char* in_names[] = {in_name_.c_str()};
    std::vector<const char*> out_names = {"score_8", "score_16", "score_32", "bbox_8", "bbox_16", "bbox_32"};

    Timer t;
    t.tic();
    auto outs = session_.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names.data(),
                             static_cast<size_t>(out_names.size()));
    if (ms_out) *ms_out = t.toc_ms();

    std::unordered_map<std::string, Ort::Value> out_map;
    for (size_t i = 0; i < outs.size(); ++i) {
        auto name = session_.GetOutputNameAllocated((int)i, alloc_);
        std::string key = name ? name.get() : out_names[i];
        out_map.emplace(std::move(key), std::move(outs[i]));
    }

    return decode_scrfd_heads(out_map, img_bgr.size(), sx, sy, cfg_.infer.box_thresh, cfg_.infer.apply_sigmoid,
                              cfg_.min_size_w, cfg_.min_size_h, cfg_.output.verbose, H, W);
}

bool FaceDetector::prepare_binding(int target_w, int target_h, int /*contexts*/) {
    bound_w_ = target_w;
    bound_h_ = target_h;
    input_buf_.assign((size_t)3 * target_w * target_h, 0.f);
    bound_tensors_.clear();
    out_bufs_.clear();
    out_shapes_.clear();
    binding_.reset();

    static Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Input
    const std::vector<int64_t> ishape = {1, 3, target_h, target_w};
    Ort::Value in =
        Ort::Value::CreateTensor<float>(cpu_mem, input_buf_.data(), input_buf_.size(), ishape.data(), ishape.size());

    // Outputs (flattened layout)
    auto head_hw = [&](int stride) { return std::pair<int, int>{target_h / stride, target_w / stride}; };
    std::vector<std::vector<int64_t>> shapes;
    shapes.reserve(out_names_.size());
    for (const auto& name : out_names_) {
        int stride = 8;
        if (name.find("16") != std::string::npos) stride = 16;
        if (name.find("32") != std::string::npos) stride = 32;
        auto [h, w] = head_hw(stride);
        const int anchors = 2;
        if (name.find("score") != std::string::npos) {
            shapes.push_back({1, (int64_t)(h * w * anchors), 1});
        } else {
            shapes.push_back({1, (int64_t)(h * w * anchors), 4});
        }
    }

    out_bufs_.resize(out_names_.size());
    out_shapes_ = shapes;
    bound_tensors_.reserve(out_names_.size() + 1);
    bound_tensors_.push_back(std::move(in));
    for (size_t i = 0; i < out_names_.size(); ++i) {
        size_t numel = 1;
        for (auto v : shapes[i])
            numel *= (size_t)v;
        out_bufs_[i].assign(numel, 0.f);
        bound_tensors_.push_back(Ort::Value::CreateTensor<float>(cpu_mem, out_bufs_[i].data(), out_bufs_[i].size(),
                                                                 shapes[i].data(), shapes[i].size()));
    }

    binding_ = std::make_unique<Ort::IoBinding>(session_);
    binding_->BindInput(in_name_.c_str(), bound_tensors_[0]);
    for (size_t i = 0; i < out_names_.size(); ++i) {
        binding_->BindOutput(out_names_[i].c_str(), bound_tensors_[i + 1]);
    }
    return true;
}

std::vector<Detection> FaceDetector::detect_bound(const cv::Mat& img_bgr, int /*ctx_idx*/, double* ms_out) {
    if (!binding_ || bound_w_ <= 0 || bound_h_ <= 0) {
        return detect(img_bgr, ms_out);
    }

    cv::Mat resized, blob;
    float sx = 1.0f, sy = 1.0f;
    preprocess(img_bgr, resized, blob, sx, sy, bound_w_, bound_h_);

    // copy blob into pre-bound input
    std::memcpy(input_buf_.data(), blob.data, sizeof(float) * input_buf_.size());

    Timer t;
    t.tic();
    session_.Run(Ort::RunOptions{nullptr}, *binding_);
    if (ms_out) *ms_out = t.toc_ms();

    std::unordered_map<std::string, Ort::Value> out_map;
    static Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    for (size_t i = 0; i < out_names_.size(); ++i) {
        out_map.emplace(out_names_[i],
                        Ort::Value::CreateTensor<float>(cpu_mem, out_bufs_[i].data(), out_bufs_[i].size(),
                                                        out_shapes_[i].data(), out_shapes_[i].size()));
    }

    return decode_scrfd_heads(out_map, img_bgr.size(), sx, sy, cfg_.infer.box_thresh, cfg_.infer.apply_sigmoid,
                              cfg_.min_size_w, cfg_.min_size_h, cfg_.output.verbose, bound_h_, bound_w_);
}
