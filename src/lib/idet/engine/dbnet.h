/**
 * @file dbnet.h
 * @brief DBNet-like text detector engine (ORT backend).
 *
 * @details
 * This header declares @ref idet::engine::DBNet, an ONNX Runtime-based implementation of a
 * DBNet-style text detector.
 *
 * Production guarantees:
 * - **Unbound mode** (@ref infer_unbound): allocates temporary buffers per call; safe for concurrent calls.
 * - **Bound mode** (@ref infer_bound): preallocates I/O per context; safe for concurrent calls only if each caller
 *   uses a distinct `ctx_idx`.
 * - Output tensor layout is inferred at runtime (NCHW / NHWC / N1HW / HW) and handled without undefined behavior.
 *
 * Expected model contract:
 * - Input:  float32 tensor with shape [1, 3, H, W] (CHW), normalized.
 * - Output: probability map-like tensor, commonly one of:
 *   - [1, 1, H, W] (NCHW)
 *   - [1, H, W, 1] (NHWC)
 *   - [1, H, W]    (N1HW)
 *   - [H, W]       (HW)
 *
 * @ingroup idet_engine
 */

#pragma once

#include "engine/engine.h"
#include "internal/ort_tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace idet::engine {

/**
 * @brief DBNet-like engine implementation.
 *
 * @details
 * The engine performs:
 * - preprocessing: BGR U8 -> normalized CHW float32 input,
 * - inference via ONNX Runtime,
 * - layout-aware extraction of an HxW probability plane,
 * - postprocessing to produce polygon/quad-like detections.
 *
 * Thread-safety:
 * - @ref infer_unbound is safe for concurrent calls.
 * - @ref infer_bound is safe for concurrent calls only when each thread uses a unique context index.
 */
class DBNet final : public IEngine {
  public:
    /**
     * @brief Construct DBNet engine and create its ONNX Runtime session.
     *
     * @param cfg Detector configuration snapshot.
     *
     * @warning Constructor may throw internally (ORT/OpenCV), but the factory that creates engines is expected
     *          to catch exceptions and convert them into @ref idet::Status.
     */
    explicit DBNet(const DetectorConfig& cfg);

    /** @brief Engine kind identifier. */
    EngineKind kind() const noexcept override {
        return EngineKind::DBNet;
    }

    /** @brief Task domain handled by this engine. */
    Task task() const noexcept override {
        return Task::Text;
    }

    /**
     * @brief Apply a hot configuration update.
     *
     * @details
     * Only inference-only parameters are expected to be updated (thresholds, flags, etc.).
     * Model path, task/engine kind, and runtime policy are expected to be immutable for hot updates.
     */
    Status update_hot(const DetectorConfig& cfg) noexcept override;

    /**
     * @brief Prepare bound inference for a fixed input shape and multiple contexts.
     *
     * @param w Input width in pixels (> 0).
     * @param h Input height in pixels (> 0).
     * @param contexts Number of contexts (> 0).
     */
    Status setup_binding(int w, int h, int contexts) noexcept override;

    /** @brief Tear down bound-mode state and return to unbound mode. */
    void unset_binding() noexcept override;

    /**
     * @brief Run inference in unbound mode.
     *
     * @param bgr Input BGR image (CV_8UC3).
     * @return Result with detections or error status.
     */
    Result<std::vector<algo::Detection>> infer_unbound(const cv::Mat& bgr) noexcept override;

    /**
     * @brief Run inference in bound mode using a pre-prepared binding context.
     *
     * @param bgr Input BGR image (CV_8UC3).
     * @param ctx_idx Context index in [0, bound_contexts()).
     * @return Result with detections or error status.
     */
    Result<std::vector<algo::Detection>> infer_bound(const cv::Mat& bgr, int ctx_idx) noexcept override;

  private:
    /**
     * @brief Geometry mapping between original image size and network input size.
     *
     * @details
     * `sx` and `sy` represent the scale factors used to map coordinates from the original image
     * space to the network input space (and vice versa).
     */
    struct NetGeom {
        int in_w = 0;
        int in_h = 0;
        float sx = 1.0f; // in_w / orig_w
        float sy = 1.0f; // in_h / orig_h
    };

    /**
     * @brief Per-context bound inference state.
     *
     * @details
     * Each context owns its input/output buffers and binding objects to enable parallel bound inference.
     * The struct is move-only to avoid accidental expensive copies and to respect ORT handle semantics.
     */
    struct BoundCtx {
        std::vector<float> in;              ///< CHW input buffer (size = 3 * bound_h * bound_w)
        std::vector<float> out;             ///< Raw output buffer (size = numel(bound_out_shape_))
        std::vector<float> scratch_prob_hw; ///< Scratch for NHWC -> HW extraction

        std::unique_ptr<Ort::IoBinding> binding; ///< Per-context IoBinding handle
        Ort::Value in_tensor{nullptr};           ///< Bound input tensor
        Ort::Value out_tensor{nullptr};          ///< Bound output tensor

        BoundCtx() = default;
        BoundCtx(const BoundCtx&) = delete;
        BoundCtx& operator=(const BoundCtx&) = delete;

        BoundCtx(BoundCtx&&) noexcept = default;
        BoundCtx& operator=(BoundCtx&&) noexcept = default;

        ~BoundCtx() = default;
    };

  private:
    /** @brief Refresh cached hot parameters from @ref cfg_. */
    void cache_hot_() noexcept;

    /**
     * @brief Compute input geometry (resize policy) for a given original image size.
     *
     * @param orig_w Original width.
     * @param orig_h Original height.
     * @param force_w Forced input width (0 means "auto").
     * @param force_h Forced input height (0 means "auto").
     * @return Geometry mapping structure.
     */
    NetGeom make_geom_(int orig_w, int orig_h, int force_w, int force_h) const;

    /**
     * @brief Fill CHW float32 input buffer from BGR image, including resize/normalization as needed.
     *
     * @param dst_chw Destination buffer (size = 3 * in_h * in_w).
     * @param in_w Target input width.
     * @param in_h Target input height.
     * @param bgr Source image (CV_8UC3).
     */
    void fill_input_chw_(float* dst_chw, int in_w, int in_h, const cv::Mat& bgr) const;

    /**
     * @brief Run ONNX Runtime inference in unbound mode and return the raw output tensor.
     *
     * @param in Pointer to contiguous CHW float input (size = in_count).
     * @param in_count Number of floats in the input buffer.
     * @param in_h Input height.
     * @param in_w Input width.
     * @return Result with the output tensor (Ort::Value) or error status.
     */
    Result<Ort::Value> run_ort_unbound_(const float* in, std::size_t in_count, int in_h, int in_w) noexcept;

    /**
     * @brief Probe the real output tensor descriptor for a given input shape.
     *
     * @details
     * Used during binding preparation to allocate output buffers with the correct size and
     * to remember the real output layout/shape (never assume [1,1,H,W]).
     *
     * @param in_h Input height.
     * @param in_w Input width.
     * @return Result with inferred tensor descriptor or error status.
     */
    Result<idet::internal::TensorDesc> probe_output_desc_(int in_h, int in_w) noexcept;

    /**
     * @brief Postprocess a contiguous HxW probability plane into detections.
     *
     * @param prob_hw Pointer to contiguous probability plane (size = out_h * out_w).
     * @param out_w Probability plane width.
     * @param out_h Probability plane height.
     * @param orig_w Original image width.
     * @param orig_h Original image height.
     * @return Vector of detections in original image coordinates.
     */
    std::vector<algo::Detection> postprocess_hw_(const float* prob_hw, int out_w, int out_h, int orig_w,
                                                 int orig_h) const;

    /**
     * @brief Best-effort "rect-like" polygon expansion helper used by postprocessing.
     *
     * @param box Input quad points.
     * @param unclip Expansion ratio.
     * @return Expanded quad points.
     */
    static std::array<cv::Point2f, 4> unclip_rect_like_(const std::array<cv::Point2f, 4>& box, float unclip) noexcept;

  private:
    /** @brief ONNX input tensor name. */
    std::string in_name_;

    /** @brief ONNX output tensor name. */
    std::string out_name_;

    // --------------------------- cached hot params ---------------------------

    bool apply_sigmoid_ = false;
    float bin_thresh_ = 0.3f;
    float box_thresh_ = 0.5f;
    float unclip_ = 1.0f;
    int max_img_ = 960;
    int min_w_ = 5;
    int min_h_ = 5;

    // --------------------------- binding metadata ----------------------------

    idet::internal::TensorDesc bound_out_desc_{};
    std::vector<int64_t> bound_out_shape_; // Real ORT output shape for the bound input shape.
    int bound_out_w_ = 0;
    int bound_out_h_ = 0;

    /** @brief Per-context bound inference state. */
    std::vector<BoundCtx> ctxs_;
};

} // namespace idet::engine
