/**
 * @file dbnet.cpp
 * @ingroup idet_engine
 * @brief DBNet-like text detector engine implementation (ORT backend).
 *
 * @details
 * This translation unit implements @ref idet::engine::DBNet:
 * - preprocessing: BGR U8 -> normalized CHW float32 (with optional resize),
 * - inference: ONNX Runtime session execution (unbound or bound via IoBinding),
 * - output handling: layout-aware extraction of an HxW probability plane,
 * - postprocessing: binarization + contour extraction + rotated-rect quad + unclipping.
 *
 * Output layout handling:
 * - The model export may produce probmap as NCHW / NHWC / N1HW / HW. The implementation uses
 *   @ref idet::internal::make_desc_probmap and @ref idet::internal::extract_hw_channel to obtain
 *   a contiguous HxW plane (channel 0 by default).
 *
 * Binding strategy:
 * - Bound mode probes the real output shape once (by a single unbound run with a zero input) to avoid
 *   forcing an assumed shape like {1,1,H,W}.
 * - Each bound context owns its own input/output buffers and @ref Ort::IoBinding instance.
 *
 * Thread-safety:
 * - Unbound inference is safe for concurrent calls.
 * - Bound inference is safe only when each concurrent caller uses a distinct context index.
 */

#include "engine/dbnet.h"

#include "algo/geometry.h"
#include "internal/chw_preprocess.h"
#include "internal/letterbox.h"
#include "internal/opencv_adapter.h"
#include "internal/opencv_geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <new>
#include <utility>

#if defined(_OPENMP)
    #include <omp.h>
#endif

namespace idet::engine {

namespace {

/**
 * @brief Align an integer value up to the next multiple of @p a.
 *
 * @param v Input value.
 * @param a Alignment (<= 1 means "no alignment").
 * @return Smallest value >= v that is a multiple of a.
 */
static inline int align_up_(int v, int a) noexcept {
    if (a <= 1) return v;
    return (v + a - 1) / a * a;
}

/**
 * @brief Numerically stable sigmoid for float (used optionally for probmap outputs).
 *
 * @note Used only when cfg_.infer.apply_sigmoid is enabled.
 */
static inline float sigmoid_(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Clamp float value to [lo, hi].
 */
static inline float clampf_(float v, float lo, float hi) noexcept {
    return std::max(lo, std::min(hi, v));
}

} // namespace

/**
 * @brief Construct DBNet engine and initialize ONNX Runtime session.
 *
 * @details
 * - Validates configuration invariants (task=Text, engine=DBNet).
 * - Creates ORT session (from file or embedded blob, depending on cfg).
 * - Caches input/output tensor names (best-effort; falls back to "input"/"output").
 * - Caches hot inference parameters into internal fields.
 *
 * @throw std::runtime_error on configuration/session initialization failure.
 *        (Factory layer is expected to catch and convert to Status.)
 */
DBNet::DBNet(const DetectorConfig& cfg) : IEngine(cfg, "idet-dbnet") {
    const Status vs = cfg_.validate();
    if (!vs.ok()) throw std::runtime_error(vs.message);
    if (cfg_.task != Task::Text) throw std::runtime_error("DBNet: cfg.task must be Text");
    if (cfg_.engine != EngineKind::DBNet) throw std::runtime_error("DBNet: cfg.engine must be DBNet");

    auto s = create_session_(cfg_.model_path, cfg_.engine);
    if (!s.ok()) throw std::runtime_error(s.message);

    try {
        Ort::AllocatedStringPtr in0 = session_.GetInputNameAllocated(0, alloc_);
        in_name_ = in0 ? in0.get() : std::string("input");

        Ort::AllocatedStringPtr out0 = session_.GetOutputNameAllocated(0, alloc_);
        out_name_ = out0 ? out0.get() : std::string("output");
    } catch (...) {
        if (in_name_.empty()) in_name_ = "input";
        if (out_name_.empty()) out_name_ = "output";
    }

    cache_hot_();
}

/**
 * @brief Cache hot-update parameters from @ref cfg_ into local fields.
 *
 * @details
 * This avoids repeatedly reading nested config fields inside the hot path.
 */
void DBNet::cache_hot_() noexcept {
    apply_sigmoid_ = cfg_.infer.apply_sigmoid;
    bin_thresh_ = cfg_.infer.bin_thresh;
    box_thresh_ = cfg_.infer.box_thresh;
    unclip_ = cfg_.infer.unclip;
    max_img_ = cfg_.infer.max_img_size;
    min_w_ = cfg_.infer.min_roi_size_w;
    min_h_ = cfg_.infer.min_roi_size_h;
}

/**
 * @brief Apply hot-update configuration.
 *
 * @details
 * Validates invariants via base helper and updates cached parameters.
 */
Status DBNet::update_hot(const DetectorConfig& next) noexcept {
    const Status chk = check_hot_update_(next);
    if (!chk.ok()) return chk;

    apply_hot_common_(next);
    cache_hot_();
    return Status::Ok();
}

/**
 * @brief Compute network input geometry (target size) for preprocessing.
 *
 * @details
 * Computes only the network input dimensions; actual letterbox parameters (uniform scale
 * and padding) are derived later inside @ref fill_input_chw_ based on the source size.
 * - If @p force_w/@p force_h are provided, uses them (aligned up to multiple of 32).
 * - Otherwise applies a "max side <= max_img" downscale that preserves aspect, then aligns
 *   both dims to multiples of 32 to satisfy the backbone's constraints.
 *
 * The returned geometry is intentionally aspect-distorted relative to (orig_w, orig_h):
 * the input buffer is rectangular and aligned, while the source content is letterboxed
 * inside it without distortion. This is what allows the model to see the same aspect
 * ratio it was trained on.
 *
 * @note Alignment (32) matches common DBNet-family backbones with downsample/upsample constraints.
 */
DBNet::NetGeom DBNet::make_geom_(int orig_w, int orig_h, int force_w, int force_h) const {
    NetGeom g{};
    const int align = 32;

    if (force_w > 0 && force_h > 0) {
        g.in_w = align_up_(force_w, align);
        g.in_h = align_up_(force_h, align);
        return g;
    }

    int tw = std::max(1, orig_w);
    int th = std::max(1, orig_h);

    if (max_img_ > 0) {
        const int max_side = std::max(tw, th);
        if (max_side > max_img_) {
            const float scale = (float)max_img_ / (float)max_side;
            tw = std::max(1, (int)std::lround(tw * scale));
            th = std::max(1, (int)std::lround(th * scale));
        }
    }

    g.in_w = align_up_(tw, align);
    g.in_h = align_up_(th, align);
    return g;
}

/**
 * @brief Letterbox + normalize a BGR U8 image into a CHW float32 tensor.
 *
 * @details
 * Performs an aspect-preserving resize-and-pad (letterbox) so the (in_w, in_h) buffer
 * receives the source image scaled by a single uniform factor with the unused area zero-
 * padded at the bottom-right. The returned @ref idet::internal::LetterboxInfo lets the
 * decode side undo the transform with a single subtract-and-divide.
 *
 * The pixel values are then normalized with ImageNet mean/std (in BGR order, in 0..255 scale).
 */
idet::internal::LetterboxInfo DBNet::fill_input_chw_(float* dst, int in_w, int in_h, const cv::Mat& bgr) const {
    // mean/std in BGR order (ImageNet)
    const float mean[3] = {0.406f * 255.0f, 0.456f * 255.0f, 0.485f * 255.0f};
    const float inv_std[3] = {1.0f / (0.225f * 255.0f), 1.0f / (0.224f * 255.0f), 1.0f / (0.229f * 255.0f)};

    cv::Mat lb_img;
    auto info = internal::letterbox_bgr(bgr, lb_img, in_w, in_h, /*pad_value=*/0);
    internal::bgr_u8_to_chw_f32_same_size(lb_img, dst, mean, inv_std);
    return info;
}

/**
 * @brief Execute ORT inference in unbound mode and return output tensor.
 *
 * @details
 * - Creates a CPU tensor view over @p in (no copy).
 * - Runs the session with single input and single output.
 * - Returns the first output tensor.
 */
Result<Ort::Value> DBNet::run_ort_unbound_(const float* in, std::size_t in_count, int in_h, int in_w) noexcept {
    try {
        Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::vector<int64_t> ishape = {1, 3, in_h, in_w};

        Ort::Value in_tensor =
            Ort::Value::CreateTensor<float>(cpu_mem, const_cast<float*>(in), in_count, ishape.data(), ishape.size());

        const char* in_names[] = {in_name_.c_str()};
        const char* out_names[] = {out_name_.c_str()};

        auto outs = session_.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);

        if (outs.empty()) return Result<Ort::Value>::Err(Status::Internal("DBNet: session.Run returned no outputs"));
        return Result<Ort::Value>::Ok(std::move(outs[0]));
    } catch (const std::bad_alloc&) {
        return Result<Ort::Value>::Err(Status::OutOfMemory("DBNet: run_ort_unbound bad_alloc"));
    } catch (const std::exception& e) {
        return Result<Ort::Value>::Err(Status::Internal(std::string("DBNet: run_ort_unbound: ") + e.what()));
    } catch (...) {
        return Result<Ort::Value>::Err(Status::Internal("DBNet: run_ort_unbound: unknown"));
    }
}

/**
 * @brief Probe output layout/shape descriptor for a given input shape.
 *
 * @details
 * Performs one inference on a zero input to read the real output tensor shape,
 * then converts it into a layout-aware descriptor via @ref idet::internal::make_desc_probmap.
 *
 * @note Used by @ref setup_binding to allocate bound output buffers with the correct size.
 */
Result<idet::internal::TensorDesc> DBNet::probe_output_desc_(int in_h, int in_w) noexcept {
    try {
        std::vector<float> zero((std::size_t)3 * (std::size_t)in_h * (std::size_t)in_w, 0.f);
        auto r = run_ort_unbound_(zero.data(), zero.size(), in_h, in_w);
        if (!r.ok()) return Result<idet::internal::TensorDesc>::Err(r.status());

        Ort::Value out = std::move(r.value());
        auto sh = out.GetTensorTypeAndShapeInfo().GetShape();

        auto desc = idet::internal::make_desc_probmap(sh);
        if (desc.layout == idet::internal::TensorLayout::Unknown || desc.H <= 0 || desc.W <= 0) {
            return Result<idet::internal::TensorDesc>::Err(
                Status::Unsupported("DBNet: cannot infer output probmap layout"));
        }
        return Result<idet::internal::TensorDesc>::Ok(std::move(desc));
    } catch (const std::exception& e) {
        return Result<idet::internal::TensorDesc>::Err(
            Status::Internal(std::string("DBNet: probe_output_desc: ") + e.what()));
    } catch (...) {
        return Result<idet::internal::TensorDesc>::Err(Status::Internal("DBNet: probe_output_desc: unknown"));
    }
}

/**
 * @brief Distance-offset polygon expansion (Vatti-style) for a rotated rectangle.
 *
 * @details
 * Approximates @c pyclipper polygon offset used by reference DBNet implementations.
 * For a rotated rectangle of size @c (w, h):
 *   D = w * h * unclip / (2 * (w + h))
 * Each edge is moved outward by @c D, so the new size is @c (w + 2D, h + 2D) with the same
 * center and angle. Long thin text quads expand much more in the short dimension (height)
 * than in the long one (width), which matches the paper's intent and makes the text
 * actually fit inside the resulting box.
 *
 * @par Why not just scale around the centroid?
 * The legacy implementation multiplied each corner offset from the centroid by @c unclip.
 * This is equivalent to @c (w * unclip, h * unclip) which over-extends the long side and
 * leaves the short side too small. Visually: long horizontal text grew sideways into
 * empty area while still cropping ascenders/descenders.
 */
idet::Quad DBNet::unclip_rect_like_(const idet::Quad& box, float unclip) noexcept {
    if (unclip <= 1.0f) return box;

    // Recover the rotated rect (center, size, angle) from the four points so we can extend
    // its size along the rect's local axes rather than along the global x/y axes.
    const auto cv_box = internal::opencv_adapter::to_cv_quad(box);
    std::vector<cv::Point2f> pts(cv_box.begin(), cv_box.end());
    cv::RotatedRect rr = cv::minAreaRect(pts);

    const float w = rr.size.width;
    const float h = rr.size.height;
    if (w <= 1.0f || h <= 1.0f) return box;

    // Vatti polygon offset distance for a rectangle: D = area * unclip / perimeter.
    const float perimeter = 2.0f * (w + h);
    const float D = (w * h * unclip) / perimeter;

    rr.size.width = w + 2.0f * D;
    rr.size.height = h + 2.0f * D;

    cv::Point2f cv_pts[4];
    rr.points(cv_pts);

    std::array<cv::Point2f, 4> out_cv{};
    for (int i = 0; i < 4; ++i)
        out_cv[i] = cv_pts[i];
    idet::Quad out = internal::opencv_adapter::from_cv_quad(out_cv);
    algo::order_quad(out.data());
    return out;
}

/**
 * @brief Postprocess a contiguous HxW probability plane into detections.
 *
 * @details
 * Pipeline:
 * 1) Optional sigmoid (if output is logits).
 * 2) Binarize with @ref bin_thresh_ to a bitmap.
 * 3) Extract contours.
 * 4) Score each contour using probability map, filter by @ref box_thresh_.
 * 5) Fit min-area rotated rectangle, optionally unclip, map back to original image space.
 *
 * @note
 * The returned detections are sorted by descending score.
 */
std::vector<algo::Detection> DBNet::postprocess_hw_(const float* prob_hw, int out_w, int out_h, const NetGeom& geom,
                                                    int orig_w, int orig_h) const {
    std::vector<algo::Detection> dets;
    if (!prob_hw || out_w <= 0 || out_h <= 0 || orig_w <= 0 || orig_h <= 0) return dets;
    if (geom.in_w <= 0 || geom.in_h <= 0) return dets;

    cv::Mat prob(out_h, out_w, CV_32F, const_cast<float*>(prob_hw));

    const float thr = clampf_(bin_thresh_, 0.0f, 1.0f);

    // Fuse optional sigmoid + binarization in a single pass to avoid an extra clone + loop.
    // prob2 holds the (possibly sigmoid-transformed) probability map used downstream for
    // contour scoring; bitmap is the binary mask consumed by cv::findContours.
    //
    // Rows are independent, so we parallelize across @c y for large enough maps.
    cv::Mat prob2;
    cv::Mat bitmap(out_h, out_w, CV_8U);
    constexpr int kParallelMinPixels = 64 * 64;
    bool parallel = (out_h * out_w) >= kParallelMinPixels;
#if defined(_OPENMP)
    // DBNet postprocess can run inside tile-level OpenMP. In that case the outer tile loop
    // already controls concurrency; avoid nested regions and their scheduling overhead.
    parallel = parallel && !omp_in_parallel();
#endif
    (void)parallel; // referenced inside the OpenMP `if (parallel)` clause

    if (apply_sigmoid_) {
        prob2.create(out_h, out_w, CV_32F);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (parallel)
#endif
        for (int y = 0; y < out_h; ++y) {
            const float* src = prob.ptr<float>(y);
            float* dst = prob2.ptr<float>(y);
            std::uint8_t* br = bitmap.ptr<std::uint8_t>(y);
            for (int x = 0; x < out_w; ++x) {
                const float s = sigmoid_(src[x]);
                dst[x] = s;
                br[x] = (s > thr) ? 255 : 0;
            }
        }
    } else {
        prob2 = prob;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (parallel)
#endif
        for (int y = 0; y < out_h; ++y) {
            const float* pr = prob2.ptr<float>(y);
            std::uint8_t* br = bitmap.ptr<std::uint8_t>(y);
            for (int x = 0; x < out_w; ++x)
                br[x] = (pr[x] > thr) ? 255 : 0;
        }
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // Coordinate chain:
    //   probmap (out_w x out_h)  --pm_to_in-->  network input (in_w x in_h)
    //   network input            --letterbox_inverse-->  original image (orig_w x orig_h)
    //
    // The probmap is typically the same size as the network input but may be downsampled
    // (e.g., 1/4) for some DBNet exports; we scale by (in_w/out_w, in_h/out_h) to recover
    // network-input pixel coordinates, then apply the inverse letterbox (uniform scale + pad).
    const float pm_to_in_x = (float)geom.in_w / (float)out_w;
    const float pm_to_in_y = (float)geom.in_h / (float)out_h;
    const float inv_scale = (geom.lb.scale > 0.0f) ? (1.0f / geom.lb.scale) : 1.0f;
    const float pad_x = (float)geom.lb.pad_x;
    const float pad_y = (float)geom.lb.pad_y;

    // Reuse a single scratch instance across all contours in this postprocess call. The internal
    // mask buffer is allocated lazily on the first contour and then reused; the per-contour
    // vector inside scratch is also reused.
    internal::opencv_geometry::ContourScoreScratch contour_scratch;

    for (auto& c : contours) {
        if (c.size() < 4) continue;

        const float score = internal::opencv_geometry::contour_score(prob2, c, contour_scratch);
        if (score < box_thresh_) continue;

        cv::RotatedRect rr = cv::minAreaRect(c);
        const float w = rr.size.width;
        const float h = rr.size.height;
        if (w <= 1.f || h <= 1.f) continue;

        // Convert size from probmap pixels to original-image pixels. The (1/scale) factor
        // undoes the letterbox uniform scaling; pm_to_in_* compensates if probmap is at a
        // different resolution than the network input.
        const float ow = w * pm_to_in_x * inv_scale;
        const float oh = h * pm_to_in_y * inv_scale;
        if (min_w_ > 0 && ow < (float)min_w_) continue;
        if (min_h_ > 0 && oh < (float)min_h_) continue;

        std::array<cv::Point2f, 4> cv_box{};
        rr.points(cv_box.data());

        idet::Quad box = internal::opencv_adapter::from_cv_quad(cv_box);

        // Apply unclip in probmap space (it's a uniform expansion of the rotated rect, so
        // the result is identical to applying it in network-input space, modulo a constant
        // scale that is handled by the mapping below).
        if (unclip_ > 1.0f) box = unclip_rect_like_(box, unclip_);

        for (auto& p : box) {
            const float xn = p.x * pm_to_in_x; // probmap -> network input
            const float yn = p.y * pm_to_in_y;
            p.x = clampf_((xn - pad_x) * inv_scale, 0.0f, (float)orig_w);
            p.y = clampf_((yn - pad_y) * inv_scale, 0.0f, (float)orig_h);
        }

        algo::order_quad(box.data());

        algo::Detection d;
        d.score = score;
        d.pts = box;
        dets.push_back(d);
    }

    std::sort(dets.begin(), dets.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    return dets;
}

Status DBNet::setup_binding(int w, int h, int contexts) noexcept {
    try {
        unset_binding();

        if (w <= 0 || h <= 0) return Status::Invalid("DBNet::setup_binding: non-positive w/h");
        if (contexts <= 0) contexts = 1;

        bound_w_ = w;
        bound_h_ = h;
        contexts_ = contexts;

        const NetGeom g = make_geom_(w, h, w, h);

        // Probe real output shape/layout once
        auto pr = probe_output_desc_(g.in_h, g.in_w);
        if (!pr.ok()) return pr.status();

        bound_out_desc_ = pr.value();
        bound_out_shape_ = bound_out_desc_.shape;
        bound_out_h_ = (int)bound_out_desc_.H;
        bound_out_w_ = (int)bound_out_desc_.W;

        Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        const std::vector<int64_t> ishape = {1, 3, g.in_h, g.in_w};
        const std::size_t out_numel = bound_out_desc_.numel;

        ctxs_.resize((std::size_t)contexts_);
        for (int i = 0; i < contexts_; ++i) {
            auto& c = ctxs_[(std::size_t)i];

            c.in.assign((std::size_t)3 * (std::size_t)g.in_h * (std::size_t)g.in_w, 0.f);
            c.out.assign(out_numel, 0.f);
            c.scratch_prob_hw.clear();

            c.binding = std::make_unique<Ort::IoBinding>(session_);

            c.in_tensor =
                Ort::Value::CreateTensor<float>(cpu_mem, c.in.data(), c.in.size(), ishape.data(), ishape.size());
            c.out_tensor = Ort::Value::CreateTensor<float>(cpu_mem, c.out.data(), c.out.size(), bound_out_shape_.data(),
                                                           bound_out_shape_.size());

            c.binding->BindInput(in_name_.c_str(), c.in_tensor);
            c.binding->BindOutput(out_name_.c_str(), c.out_tensor);
        }

        binding_ready_ = true;
        return Status::Ok();
    } catch (const std::bad_alloc&) {
        unset_binding();
        return Status::OutOfMemory("DBNet::setup_binding: bad_alloc");
    } catch (const std::exception& e) {
        unset_binding();
        return Status::Internal(std::string("DBNet::setup_binding: ") + e.what());
    } catch (...) {
        unset_binding();
        return Status::Internal("DBNet::setup_binding: unknown");
    }
}

void DBNet::unset_binding() noexcept {
    binding_ready_ = false;
    bound_w_ = bound_h_ = 0;
    contexts_ = 0;

    bound_out_desc_ = {};
    bound_out_shape_.clear();
    bound_out_w_ = bound_out_h_ = 0;

    ctxs_.clear();
}

Result<std::vector<algo::Detection>> DBNet::infer_unbound(const internal::BgrImageView& bgr_view) noexcept {
    try {
        const cv::Mat bgr = internal::opencv_adapter::wrap_bgr_view(bgr_view);
        if (bgr.empty() || bgr.type() != CV_8UC3) {
            return Result<std::vector<algo::Detection>>::Err(
                Status::Invalid("DBNet::infer_unbound: expected valid BGR view"));
        }

        const int ow = bgr.cols;
        const int oh = bgr.rows;

        NetGeom g = make_geom_(ow, oh, 0, 0);

        std::vector<float> in((std::size_t)3 * (std::size_t)g.in_h * (std::size_t)g.in_w);
        g.lb = fill_input_chw_(in.data(), g.in_w, g.in_h, bgr);

        auto rr = run_ort_unbound_(in.data(), in.size(), g.in_h, g.in_w);
        if (!rr.ok()) return Result<std::vector<algo::Detection>>::Err(rr.status());

        Ort::Value out = std::move(rr.value());
        auto sh = out.GetTensorTypeAndShapeInfo().GetShape();
        auto desc = idet::internal::make_desc_probmap(sh);

        const float* data = out.GetTensorData<float>();
        std::vector<float> scratch;
        // production default: channel 0
        const float* prob_hw = idet::internal::extract_hw_channel(data, desc, /*channel=*/0, scratch);
        if (!prob_hw) {
            return Result<std::vector<algo::Detection>>::Err(
                Status::Unsupported("DBNet: cannot extract prob HW plane"));
        }

        auto dets = postprocess_hw_(prob_hw, (int)desc.W, (int)desc.H, g, ow, oh);
        return Result<std::vector<algo::Detection>>::Ok(std::move(dets));
    } catch (const std::bad_alloc&) {
        return Result<std::vector<algo::Detection>>::Err(Status::OutOfMemory("DBNet::infer_unbound: bad_alloc"));
    } catch (const std::exception& e) {
        return Result<std::vector<algo::Detection>>::Err(
            Status::Internal(std::string("DBNet::infer_unbound: ") + e.what()));
    } catch (...) {
        return Result<std::vector<algo::Detection>>::Err(Status::Internal("DBNet::infer_unbound: unknown"));
    }
}

Result<std::vector<algo::Detection>> DBNet::infer_bound(const internal::BgrImageView& bgr_view, int ctx_idx) noexcept {
    try {
        const cv::Mat bgr = internal::opencv_adapter::wrap_bgr_view(bgr_view);
        if (!binding_ready_)
            return Result<std::vector<algo::Detection>>::Err(Status::Invalid("DBNet::infer_bound: binding not ready"));
        if (ctx_idx < 0 || ctx_idx >= contexts_)
            return Result<std::vector<algo::Detection>>::Err(
                Status::Invalid("DBNet::infer_bound: ctx_idx out of range"));
        if (bgr.empty() || bgr.type() != CV_8UC3)
            return Result<std::vector<algo::Detection>>::Err(
                Status::Invalid("DBNet::infer_bound: expected valid BGR view"));

        auto& c = ctxs_[(std::size_t)ctx_idx];

        const int ow = bgr.cols;
        const int oh = bgr.rows;
        NetGeom g = make_geom_(ow, oh, bound_w_, bound_h_);

        g.lb = fill_input_chw_(c.in.data(), g.in_w, g.in_h, bgr);

        session_.Run(Ort::RunOptions{nullptr}, *c.binding);

        const float* prob_hw =
            idet::internal::extract_hw_channel(c.out.data(), bound_out_desc_, /*channel=*/0, c.scratch_prob_hw);
        if (!prob_hw) {
            return Result<std::vector<algo::Detection>>::Err(
                Status::Unsupported("DBNet(bound): cannot extract prob HW plane"));
        }

        auto dets = postprocess_hw_(prob_hw, bound_out_w_, bound_out_h_, g, ow, oh);
        return Result<std::vector<algo::Detection>>::Ok(std::move(dets));
    } catch (const std::bad_alloc&) {
        return Result<std::vector<algo::Detection>>::Err(Status::OutOfMemory("DBNet::infer_bound: bad_alloc"));
    } catch (const std::exception& e) {
        return Result<std::vector<algo::Detection>>::Err(
            Status::Internal(std::string("DBNet::infer_bound: ") + e.what()));
    } catch (...) {
        return Result<std::vector<algo::Detection>>::Err(Status::Internal("DBNet::infer_bound: unknown"));
    }
}

} // namespace idet::engine
