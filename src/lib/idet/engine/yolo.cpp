/**
 * @file yolo.cpp
 * @ingroup idet_engine
 * @brief YOLO-family detector engine implementation (ORT backend).
 *
 * @details
 * See @ref yolo.h for the supported output conventions (in-graph NMS vs raw).
 * The implementation auto-detects the convention at construction time by probing
 * a dummy inference and inspecting the resulting tensor shapes.
 */

#include "engine/yolo.h"

#include "internal/chw_preprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace idet::engine {

namespace {

/** @brief Aligns @p v up to the next multiple of @p a. */
static inline int align_up_(int v, int a) noexcept {
    if (a <= 1) return v;
    return (v + a - 1) / a * a;
}

/** @brief Stable sigmoid for class score logits. */
static inline float sigmoid_(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

/** @brief Clamp helper. */
static inline float clampf_(float v, float lo, float hi) noexcept {
    return std::max(lo, std::min(hi, v));
}

/** @brief Find an output index by case-insensitive substring match. */
static int find_output_idx_(const std::vector<std::string>& names, const char* needle) noexcept {
    if (!needle) return -1;
    const std::string n = needle;
    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto& s = names[i];
        // Case-sensitive substring match is sufficient for typical YOLO export name conventions
        // (e.g. "num_dets", "num_detections", "detections", "boxes").
        if (s.find(n) != std::string::npos) return static_cast<int>(i);
    }
    return -1;
}

/**
 * @brief Read a numeric value from an Ort tensor as int64, regardless of element type.
 *
 * @details
 * Many YOLO end-to-end exports expose @c num_detections as @c int32 / @c int64 / @c float, so
 * we read whichever is reported by ORT and convert to int64.
 *
 * @return The value at index 0 of the tensor, or @p fallback if the tensor is empty/unknown.
 */
static std::int64_t read_scalar_count_(const Ort::Value& v, std::int64_t fallback) noexcept {
    try {
        auto info = v.GetTensorTypeAndShapeInfo();
        if (info.GetElementCount() == 0) return fallback;

        switch (info.GetElementType()) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return static_cast<std::int64_t>(v.GetTensorData<std::int32_t>()[0]);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return v.GetTensorData<std::int64_t>()[0];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return static_cast<std::int64_t>(v.GetTensorData<float>()[0]);
        default:
            return fallback;
        }
    } catch (...) {
        return fallback;
    }
}

} // namespace

YOLO::YOLO(const DetectorConfig& cfg) : IEngine(cfg, "idet-yolo") {
    const Status vs = cfg_.validate();
    if (!vs.ok()) throw std::runtime_error(vs.message);

    if (cfg_.task != Task::Cloth) throw std::runtime_error("YOLO: cfg.task must be Cloth");
    if (cfg_.engine != EngineKind::Yolo) throw std::runtime_error("YOLO: cfg.engine must be Yolo");

    auto s = create_session_(cfg_.model_path, cfg_.engine);
    if (!s.ok()) throw std::runtime_error(s.message);

    init_io_names_();
    cache_hot_();
}

void YOLO::cache_hot_() noexcept {
    apply_sigmoid_ = cfg_.infer.apply_sigmoid;
    // box_thresh is reused as YOLO confidence threshold (consistent with SCRFD).
    score_thr_ = cfg_.infer.box_thresh;
    max_img_ = cfg_.infer.max_img_size;
    min_w_ = cfg_.infer.min_roi_size_w;
    min_h_ = cfg_.infer.min_roi_size_h;
}

Status YOLO::update_hot(const DetectorConfig& next) noexcept {
    const Status chk = check_hot_update_(next);
    if (!chk.ok()) return chk;

    apply_hot_common_(next);
    cache_hot_();
    return Status::Ok();
}

void YOLO::init_io_names_() {
    Ort::AllocatedStringPtr in0 = session_.GetInputNameAllocated(0, alloc_);
    in_name_ = in0 ? in0.get() : std::string("images");

    const std::size_t nout = session_.GetOutputCount();
    out_names_.clear();
    out_names_.reserve(nout);
    for (std::size_t i = 0; i < nout; ++i) {
        Ort::AllocatedStringPtr on = session_.GetOutputNameAllocated(i, alloc_);
        out_names_.push_back(on ? on.get() : ("out_" + std::to_string(i)));
    }
}

void YOLO::compute_input_size_(int orig_w, int orig_h, int& in_w, int& in_h) const noexcept {
    if (bound_w_ > 0 && bound_h_ > 0) {
        in_w = align_up_(bound_w_, 32);
        in_h = align_up_(bound_h_, 32);
        return;
    }

    int tw = std::max(1, orig_w);
    int th = std::max(1, orig_h);

    if (max_img_ > 0) {
        const int max_side = std::max(tw, th);
        if (max_side > max_img_) {
            const float scale = static_cast<float>(max_img_) / static_cast<float>(max_side);
            tw = std::max(1, static_cast<int>(std::lround(tw * scale)));
            th = std::max(1, static_cast<int>(std::lround(th * scale)));
        }
    }

    in_w = align_up_(tw, 32);
    in_h = align_up_(th, 32);
}

idet::internal::LetterboxInfo YOLO::fill_input_chw_(float* dst, int in_w, int in_h,
                                                    const internal::BgrImageView& bgr) const {
    // YOLO normalization: x / 255.0 with no mean. Letterbox pad value is 114 by convention
    // (gray pad used during training across the YOLO family).
    const float mean[3] = {0.0f, 0.0f, 0.0f};
    const float inv_std[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};

    return internal::bgr_u8_to_chw_f32_letterbox(bgr, in_w, in_h, /*pad_value=*/114, dst, mean, inv_std);
}

Status YOLO::probe_layout_(int in_h, int in_w) noexcept {
    try {
        if (in_w <= 0 || in_h <= 0) return Status::Invalid("YOLO::probe_layout: non-positive shape");

        std::vector<float> chw(
            static_cast<std::size_t>(3) * static_cast<std::size_t>(in_h) * static_cast<std::size_t>(in_w), 0.0f);

        Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::vector<int64_t> ishape = {1, 3, in_h, in_w};
        Ort::Value in_tensor =
            Ort::Value::CreateTensor<float>(cpu_mem, chw.data(), chw.size(), ishape.data(), ishape.size());

        std::vector<const char*> out_names_c;
        out_names_c.reserve(out_names_.size());
        for (auto& s : out_names_)
            out_names_c.push_back(s.c_str());

        const char* in_names[] = {in_name_.c_str()};
        auto outs =
            session_.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names_c.data(), out_names_c.size());

        if (outs.empty()) return Status::Internal("YOLO::probe_layout: no outputs");

        // Heuristics:
        //   Looking for an in-graph NMS export. Patterns we accept:
        //     - single output  [B, K, 6]                           -> InGraphNms (no num_dets)
        //     - two outputs    [B, K, 6] + [B] (num_dets)          -> InGraphNms
        //     - four outputs   num_dets, boxes, scores, classes    -> InGraphNms (split form)
        //   Otherwise look for raw single output:
        //     - [B, 4+nc, N] with nc + 4 in [5, 1024] and N >> nc  -> Raw, ChannelsFirst
        //     - [B, N, 4+nc]                                       -> Raw, ChannelsLast
        // We err on the side of choosing the "obvious" format. Models that don't fit any of the
        // above are reported as Unsupported, not silently mis-decoded.

        // --- check for in-graph NMS ---
        // Look for explicit num_detections / num_dets output.
        int num_idx = find_output_idx_(out_names_, "num_dets");
        if (num_idx < 0) num_idx = find_output_idx_(out_names_, "num_detections");

        // Look for a "detections" / "boxes" output of rank 3.
        int det_idx = -1;
        for (std::size_t i = 0; i < outs.size(); ++i) {
            auto sh = outs[i].GetTensorTypeAndShapeInfo().GetShape();
            if (sh.size() == 3 && sh.back() == 6) {
                det_idx = static_cast<int>(i);
                break;
            }
        }

        if (det_idx >= 0) {
            mode_ = Mode::InGraphNms;
            layout_ = Layout::Unknown;
            detections_idx_ = det_idx;
            num_detections_idx_ = num_idx; // may be -1 (no explicit count)
            probed_in_w_ = in_w;
            probed_in_h_ = in_h;
            return Status::Ok();
        }

        // --- check for raw outputs (single rank-3 tensor) ---
        if (outs.size() >= 1) {
            auto sh = outs[0].GetTensorTypeAndShapeInfo().GetShape();
            if (sh.size() == 3) {
                const std::int64_t a = sh[1];
                const std::int64_t b = sh[2];

                // ChannelsFirst: [B, 4+nc, N], N is much larger than 4+nc.
                if (a >= 5 && a <= 1024 && b > a) {
                    mode_ = Mode::Raw;
                    layout_ = Layout::ChannelsFirst;
                    const std::int64_t feat = a;
                    // Distinguish YOLOv5 (objectness present) from YOLOv8+ (objectness folded
                    // into class scores). If feat == 5 + something_like_classes is hard to
                    // detect at probe-time without per-class names; use a conservative default:
                    // disable objectness multiplication (matches v8/v11), and let users opt in
                    // via apply_sigmoid for YOLOv5-style logit exports.
                    has_objectness_ = false;
                    num_classes_ = static_cast<int>(feat - 4);
                    detections_idx_ = 0;
                    probed_in_w_ = in_w;
                    probed_in_h_ = in_h;
                    return Status::Ok();
                }

                // ChannelsLast: [B, N, 4+nc].
                if (b >= 5 && b <= 1024 && a > b) {
                    mode_ = Mode::Raw;
                    layout_ = Layout::ChannelsLast;
                    const std::int64_t feat = b;
                    has_objectness_ = false;
                    num_classes_ = static_cast<int>(feat - 4);
                    detections_idx_ = 0;
                    probed_in_w_ = in_w;
                    probed_in_h_ = in_h;
                    return Status::Ok();
                }
            }
        }

        return Status::Unsupported(
            "YOLO: cannot infer output convention (expected [B,K,6] or [B,4+nc,N] / [B,N,4+nc])");
    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory("YOLO::probe_layout: bad_alloc");
    } catch (const std::exception& e) {
        return Status::Internal(std::string("YOLO::probe_layout: ") + e.what());
    } catch (...) {
        return Status::Internal("YOLO::probe_layout: unknown");
    }
}

std::vector<algo::Detection> YOLO::decode_in_graph_nms_(const std::vector<Ort::Value>& outs,
                                                        const idet::internal::LetterboxInfo& lb, int orig_w,
                                                        int orig_h) const {
    std::vector<algo::Detection> dets;
    if (detections_idx_ < 0 || static_cast<std::size_t>(detections_idx_) >= outs.size()) return dets;

    const auto& det_v = outs[static_cast<std::size_t>(detections_idx_)];
    auto info = det_v.GetTensorTypeAndShapeInfo();
    auto sh = info.GetShape();
    if (sh.size() != 3 || sh.back() != 6) return dets;

    const std::int64_t batch = std::max<std::int64_t>(1, sh[0]);
    const std::int64_t kmax = std::max<std::int64_t>(0, sh[1]);
    if (batch == 0 || kmax == 0) return dets;

    std::int64_t k_valid = kmax;
    if (num_detections_idx_ >= 0 && static_cast<std::size_t>(num_detections_idx_) < outs.size()) {
        k_valid = std::min(kmax, std::max<std::int64_t>(
                                     0, read_scalar_count_(outs[static_cast<std::size_t>(num_detections_idx_)], kmax)));
    }

    const float* data = det_v.GetTensorData<float>();
    const float inv_scale = (lb.scale > 0.0f) ? (1.0f / lb.scale) : 1.0f;
    const float pad_x = static_cast<float>(lb.pad_x);
    const float pad_y = static_cast<float>(lb.pad_y);

    dets.reserve(static_cast<std::size_t>(k_valid));

    for (std::int64_t i = 0; i < k_valid; ++i) {
        const float* row = data + i * 6; // (x1, y1, x2, y2, score, class)
        const float score = row[4];
        if (score < score_thr_) continue;

        const float x1 = clampf_((row[0] - pad_x) * inv_scale, 0.0f, static_cast<float>(orig_w));
        const float y1 = clampf_((row[1] - pad_y) * inv_scale, 0.0f, static_cast<float>(orig_h));
        const float x2 = clampf_((row[2] - pad_x) * inv_scale, 0.0f, static_cast<float>(orig_w));
        const float y2 = clampf_((row[3] - pad_y) * inv_scale, 0.0f, static_cast<float>(orig_h));

        if (x2 <= x1 || y2 <= y1) continue;
        if (min_w_ > 0 && (x2 - x1) < static_cast<float>(min_w_)) continue;
        if (min_h_ > 0 && (y2 - y1) < static_cast<float>(min_h_)) continue;

        algo::Detection d;
        d.score = score;
        d.pts[0] = {x1, y1};
        d.pts[1] = {x2, y1};
        d.pts[2] = {x2, y2};
        d.pts[3] = {x1, y2};
        dets.push_back(d);
    }

    std::sort(dets.begin(), dets.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    return dets;
}

std::vector<algo::Detection> YOLO::decode_raw_buffer(const float* data, std::int64_t N, int num_classes,
                                                     bool has_objectness, Layout layout, bool apply_sigmoid,
                                                     float score_thr, const idet::internal::LetterboxInfo& lb,
                                                     int orig_w, int orig_h, int min_w, int min_h) {
    std::vector<algo::Detection> dets;
    if (!data || N <= 0 || num_classes <= 0) return dets;
    if (layout != Layout::ChannelsFirst && layout != Layout::ChannelsLast) return dets;

    const float inv_scale = (lb.scale > 0.0f) ? (1.0f / lb.scale) : 1.0f;
    const float pad_x = static_cast<float>(lb.pad_x);
    const float pad_y = static_cast<float>(lb.pad_y);

    const int feat = num_classes + 4 + (has_objectness ? 1 : 0);
    const int cls_off = 4 + (has_objectness ? 1 : 0);

    auto get = [&](std::int64_t i, int c) -> float {
        if (layout == Layout::ChannelsFirst) {
            // [1, feat, N]: data[c * N + i]
            return data[static_cast<std::size_t>(c) * static_cast<std::size_t>(N) + static_cast<std::size_t>(i)];
        }
        // ChannelsLast: [1, N, feat]: data[i * feat + c]
        return data[static_cast<std::size_t>(i) * static_cast<std::size_t>(feat) + static_cast<std::size_t>(c)];
    };

    // Logit-space precomputed threshold so that, in apply_sigmoid mode, we can short-circuit
    // anchors whose logit score is already below the threshold without paying for exp(). We
    // do this only when objectness is NOT present: if objectness is folded into the score via
    // multiplication, the per-anchor threshold cannot be transformed into a single logit cut.
    // sigmoid is monotonic, so for the apply_sigmoid && !has_objectness branch the result is
    // bit-identical to the previous code path for anchors that pass; we just skip exp() calls
    // for anchors that would have been rejected anyway.
    const bool fast_logit = apply_sigmoid && !has_objectness && score_thr > 0.0f && score_thr < 1.0f;
    const float logit_thr =
        fast_logit ? std::log(score_thr / (1.0f - score_thr)) : -std::numeric_limits<float>::infinity();

    // Heuristic reserve: we typically keep < ~1% of anchors. Use a small fraction of N with
    // a sane floor/ceiling so we avoid both grow-on-every-push and over-allocation for tiny N.
    const std::size_t reserve_n =
        std::min<std::size_t>(std::max<std::size_t>(static_cast<std::size_t>(N / 64), 32), 1024);
    dets.reserve(reserve_n);

    for (std::int64_t i = 0; i < N; ++i) {
        // Class score = max(class scores), optionally sigmoid for logit-style exports.
        float best_cls = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < num_classes; ++c) {
            const float v_c = get(i, cls_off + c);
            if (v_c > best_cls) best_cls = v_c;
        }
        // Fast-path: skip exp() when the logit is already below threshold (sigmoid is monotonic).
        if (fast_logit && best_cls < logit_thr) continue;

        float score = apply_sigmoid ? sigmoid_(best_cls) : best_cls;
        if (has_objectness) {
            const float obj = get(i, 4);
            const float obj_s = apply_sigmoid ? sigmoid_(obj) : obj;
            score *= obj_s;
        }
        if (score < score_thr) continue;

        const float cx = get(i, 0);
        const float cy = get(i, 1);
        const float bw = get(i, 2);
        const float bh = get(i, 3);

        const float nx1 = cx - 0.5f * bw;
        const float ny1 = cy - 0.5f * bh;
        const float nx2 = cx + 0.5f * bw;
        const float ny2 = cy + 0.5f * bh;

        const float x1 = clampf_((nx1 - pad_x) * inv_scale, 0.0f, static_cast<float>(orig_w));
        const float y1 = clampf_((ny1 - pad_y) * inv_scale, 0.0f, static_cast<float>(orig_h));
        const float x2 = clampf_((nx2 - pad_x) * inv_scale, 0.0f, static_cast<float>(orig_w));
        const float y2 = clampf_((ny2 - pad_y) * inv_scale, 0.0f, static_cast<float>(orig_h));

        if (x2 <= x1 || y2 <= y1) continue;
        if (min_w > 0 && (x2 - x1) < static_cast<float>(min_w)) continue;
        if (min_h > 0 && (y2 - y1) < static_cast<float>(min_h)) continue;

        algo::Detection d;
        d.score = score;
        d.pts[0] = {x1, y1};
        d.pts[1] = {x2, y1};
        d.pts[2] = {x2, y2};
        d.pts[3] = {x1, y2};
        dets.push_back(d);
    }

    std::sort(dets.begin(), dets.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    return dets;
}

std::vector<algo::Detection> YOLO::decode_raw_(const std::vector<Ort::Value>& outs,
                                               const idet::internal::LetterboxInfo& lb, int orig_w, int orig_h) const {
    std::vector<algo::Detection> dets;
    if (detections_idx_ < 0 || static_cast<std::size_t>(detections_idx_) >= outs.size()) return dets;
    if (num_classes_ <= 0) return dets;

    const auto& v = outs[static_cast<std::size_t>(detections_idx_)];
    auto sh = v.GetTensorTypeAndShapeInfo().GetShape();
    if (sh.size() != 3) return dets;

    const float* data = v.GetTensorData<float>();
    const std::int64_t N = (layout_ == Layout::ChannelsFirst) ? sh[2] : sh[1];

    return decode_raw_buffer(data, N, num_classes_, has_objectness_, layout_, apply_sigmoid_, score_thr_, lb, orig_w,
                             orig_h, min_w_, min_h_);
}

std::vector<algo::Detection> YOLO::decode_(const std::vector<Ort::Value>& outs, const idet::internal::LetterboxInfo& lb,
                                           int orig_w, int orig_h) const {
    if (mode_ == Mode::InGraphNms) return decode_in_graph_nms_(outs, lb, orig_w, orig_h);
    if (mode_ == Mode::Raw) return decode_raw_(outs, lb, orig_w, orig_h);
    return {};
}

Result<std::vector<algo::Detection>> YOLO::infer_unbound(const internal::BgrImageView& bgr_view) noexcept {
    try {
        if (!bgr_view.is_valid())
            return Result<std::vector<algo::Detection>>::Err(
                Status::Invalid("YOLO::infer_unbound: expected valid BGR view"));

        int in_w = 0, in_h = 0;
        compute_input_size_(bgr_view.width, bgr_view.height, in_w, in_h);

        // Lazy probe.
        if (mode_ == Mode::Unknown) {
            const Status ps = const_cast<YOLO*>(this)->probe_layout_(in_h, in_w);
            if (!ps.ok()) return Result<std::vector<algo::Detection>>::Err(ps);
        }

        std::vector<float> chw(
            static_cast<std::size_t>(3) * static_cast<std::size_t>(in_h) * static_cast<std::size_t>(in_w), 0.0f);
        const auto lb = fill_input_chw_(chw.data(), in_w, in_h, bgr_view);

        Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::vector<int64_t> ishape = {1, 3, in_h, in_w};
        Ort::Value in_tensor =
            Ort::Value::CreateTensor<float>(cpu_mem, chw.data(), chw.size(), ishape.data(), ishape.size());

        std::vector<const char*> out_names_c;
        out_names_c.reserve(out_names_.size());
        for (auto& s : out_names_)
            out_names_c.push_back(s.c_str());

        const char* in_names[] = {in_name_.c_str()};
        auto outs =
            session_.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names_c.data(), out_names_c.size());

        auto dets = decode_(outs, lb, bgr_view.width, bgr_view.height);
        return Result<std::vector<algo::Detection>>::Ok(std::move(dets));
    } catch (const std::bad_alloc&) {
        return Result<std::vector<algo::Detection>>::Err(Status::OutOfMemory("YOLO::infer_unbound: bad_alloc"));
    } catch (const std::exception& e) {
        return Result<std::vector<algo::Detection>>::Err(
            Status::Internal(std::string("YOLO::infer_unbound: ") + e.what()));
    } catch (...) {
        return Result<std::vector<algo::Detection>>::Err(Status::Internal("YOLO::infer_unbound: unknown"));
    }
}

Status YOLO::setup_binding(int w, int h, int contexts) noexcept {
    try {
        unset_binding();

        if (w <= 0 || h <= 0) return Status::Invalid("YOLO::setup_binding: non-positive w/h");
        if (contexts <= 0) contexts = 1;

        bound_w_ = w;
        bound_h_ = h;
        contexts_ = contexts;

        const int in_w = align_up_(w, 32);
        const int in_h = align_up_(h, 32);

        // Probe before binding so we know the per-output shapes/types.
        const Status ps = probe_layout_(in_h, in_w);
        if (!ps.ok()) return ps;

        Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        bound_in_shape_ = {1, 3, in_h, in_w};

        // Re-run a single inference with a real Ort::Value to read each output's shape and
        // element type. We need both to allocate output buffers correctly. (For dynamic-shape
        // outputs, the shape comes from the actual run — we record what the model produced.)
        std::vector<float> probe_in(
            static_cast<std::size_t>(3) * static_cast<std::size_t>(in_h) * static_cast<std::size_t>(in_w), 0.0f);
        Ort::Value in_probe = Ort::Value::CreateTensor<float>(cpu_mem, probe_in.data(), probe_in.size(),
                                                              bound_in_shape_.data(), bound_in_shape_.size());

        std::vector<const char*> out_names_c;
        out_names_c.reserve(out_names_.size());
        for (auto& s : out_names_)
            out_names_c.push_back(s.c_str());

        const char* in_names[] = {in_name_.c_str()};
        auto probe_outs =
            session_.Run(Ort::RunOptions{nullptr}, in_names, &in_probe, 1, out_names_c.data(), out_names_c.size());

        ctxs_.clear();
        ctxs_.resize(static_cast<std::size_t>(contexts_));
        for (int ci = 0; ci < contexts_; ++ci) {
            auto& c = ctxs_[static_cast<std::size_t>(ci)];

            c.binding = std::make_unique<Ort::IoBinding>(session_);
            c.in.assign(static_cast<std::size_t>(3) * static_cast<std::size_t>(in_h) * static_cast<std::size_t>(in_w),
                        0.0f);
            c.in_tensor = Ort::Value::CreateTensor<float>(cpu_mem, c.in.data(), c.in.size(), bound_in_shape_.data(),
                                                          bound_in_shape_.size());
            c.binding->BindInput(in_name_.c_str(), c.in_tensor);

            c.outs.clear();
            c.out_tensors.clear();
            c.outs.resize(probe_outs.size());

            // For each output we bind to a named buffer where allowed. To keep the engine
            // type-flexible (some YOLO end-to-end exports return int32/int64 tensors), we
            // bind by name and rely on ORT to allocate the correct buffer at run time for
            // non-float outputs; only float outputs are pre-allocated.
            for (std::size_t oi = 0; oi < probe_outs.size(); ++oi) {
                auto info = probe_outs[oi].GetTensorTypeAndShapeInfo();
                const auto sh = info.GetShape();

                if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    std::size_t numel = 1;
                    for (auto v : sh)
                        numel *= static_cast<std::size_t>(std::max<std::int64_t>(1, v));

                    c.outs[oi].assign(numel, 0.0f);
                    c.out_tensors.emplace_back(Ort::Value::CreateTensor<float>(
                        cpu_mem, c.outs[oi].data(), c.outs[oi].size(), sh.data(), sh.size()));
                    c.binding->BindOutput(out_names_[oi].c_str(), c.out_tensors.back());
                } else {
                    // Let ORT allocate the non-float output (e.g. int64 num_dets).
                    c.binding->BindOutput(out_names_[oi].c_str(), cpu_mem);
                }
            }
        }

        binding_ready_ = true;
        return Status::Ok();
    } catch (const std::bad_alloc&) {
        unset_binding();
        return Status::OutOfMemory("YOLO::setup_binding: bad_alloc");
    } catch (const std::exception& e) {
        unset_binding();
        return Status::Internal(std::string("YOLO::setup_binding: ") + e.what());
    } catch (...) {
        unset_binding();
        return Status::Internal("YOLO::setup_binding: unknown");
    }
}

void YOLO::unset_binding() noexcept {
    binding_ready_ = false;
    bound_w_ = bound_h_ = 0;
    contexts_ = 0;
    ctxs_.clear();
    bound_in_shape_.clear();
}

Result<std::vector<algo::Detection>> YOLO::infer_bound(const internal::BgrImageView& bgr_view, int ctx_idx) noexcept {
    try {
        if (!binding_ready_)
            return Result<std::vector<algo::Detection>>::Err(Status::Invalid("YOLO::infer_bound: binding not ready"));
        if (ctx_idx < 0 || ctx_idx >= contexts_)
            return Result<std::vector<algo::Detection>>::Err(
                Status::Invalid("YOLO::infer_bound: ctx_idx out of range"));
        if (!bgr_view.is_valid())
            return Result<std::vector<algo::Detection>>::Err(
                Status::Invalid("YOLO::infer_bound: expected valid BGR view"));

        auto& c = ctxs_[static_cast<std::size_t>(ctx_idx)];

        const int in_w = align_up_(bound_w_, 32);
        const int in_h = align_up_(bound_h_, 32);

        const auto lb = fill_input_chw_(c.in.data(), in_w, in_h, bgr_view);

        session_.Run(Ort::RunOptions{nullptr}, *c.binding);

        // Pull outputs out of the binding so decode can read them.
        auto outs = c.binding->GetOutputValues();
        auto dets = decode_(outs, lb, bgr_view.width, bgr_view.height);

        return Result<std::vector<algo::Detection>>::Ok(std::move(dets));
    } catch (const std::bad_alloc&) {
        return Result<std::vector<algo::Detection>>::Err(Status::OutOfMemory("YOLO::infer_bound: bad_alloc"));
    } catch (const std::exception& e) {
        return Result<std::vector<algo::Detection>>::Err(
            Status::Internal(std::string("YOLO::infer_bound: ") + e.what()));
    } catch (...) {
        return Result<std::vector<algo::Detection>>::Err(Status::Internal("YOLO::infer_bound: unknown"));
    }
}

} // namespace idet::engine
