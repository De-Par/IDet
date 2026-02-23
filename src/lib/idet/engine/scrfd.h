/**
 * @file scrfd.h
 * @ingroup idet_engine
 * @brief SCRFD face detector engine (ORT backend).
 *
 * @details
 * Implements a production-oriented SCRFD inference backend on top of ONNX Runtime.
 * The engine supports both:
 * - **unbound** mode (@ref infer_unbound): allocates per call, safe for concurrent calls,
 * - **bound** mode (@ref setup_binding + @ref infer_bound): preallocates/binds I/O per context
 *   for maximum throughput; concurrent use is safe only with distinct context indices.
 *
 * Output export variability:
 * SCRFD models are frequently exported with different output tensor layouts depending on
 * conversion pipeline and opset:
 * - score tensors:  rank-2/3/4 (flat or spatial maps)
 * - bbox tensors:   rank-3/4 or HW4-like
 *
 * This implementation **infers layout per head** independently for:
 * - classification/score output, and
 * - bbox regression output.
 *
 * The decoder then consumes score/bbox pointers in a layout-aware way to produce
 * @ref idet::algo::Detection results in original image coordinates.
 *
 * Thread-safety:
 * - @ref infer_unbound is thread-safe for concurrent calls.
 * - @ref infer_bound is thread-safe **only** if each concurrent caller uses a unique @p ctx_idx.
 *
 * @note
 * Input tensors are expected to be float32 in CHW layout ([1,3,H,W]) with engine-defined
 * normalization consistent with the training recipe.
 */

#pragma once

#include "engine/engine.h"
#include "internal/ort_tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace idet::engine {

/**
 * @brief SCRFD face detector engine implementation.
 *
 * @details
 * The engine runs a multi-head SCRFD model and decodes its outputs into detections.
 * A "head" typically corresponds to a feature map stride (e.g., 8/16/32) and provides:
 * - a score/classification tensor (face confidence),
 * - a bbox regression tensor (x1,y1,x2,y2 or equivalent parameterization, export-dependent).
 *
 * The implementation keeps a cached list of heads (@ref heads_) where each head stores
 * probed output indices and inferred layouts/shapes. This probing is performed during
 * binding setup (fixed shape) and may also be performed lazily for unbound inference
 * depending on the model export characteristics.
 *
 * Hot-update contract:
 * @ref update_hot accepts only changes that do not require recreating ORT session or rebinding.
 * Typical hot fields include score threshold and post-processing related parameters.
 *
 * Bound inference:
 * - @ref setup_binding creates @p contexts independent binding contexts.
 * - Each context owns its input buffer, output buffers, and an Ort::IoBinding object.
 * - The caller must ensure exclusive use of each context during @ref infer_bound.
 */
class SCRFD final : public IEngine {
  public:
    /**
     * @brief Construct SCRFD engine and create ORT session.
     *
     * @param cfg Detector configuration snapshot. Must satisfy:
     * - cfg.task == @ref idet::Task::Face
     * - cfg.engine == @ref idet::EngineKind::SCRFD
     *
     * @throws std::exception in the constructor implementation if configuration validation
     *         fails or ORT session creation fails. The factory wrapper is expected to catch it.
     */
    explicit SCRFD(const DetectorConfig& cfg);

    /** @brief Engine kind identifier. */
    EngineKind kind() const noexcept override {
        return EngineKind::SCRFD;
    }

    /** @brief Task domain handled by this engine (faces). */
    Task task() const noexcept override {
        return Task::Face;
    }

    /**
     * @brief Apply a hot update to inference-only parameters (no ORT session recreation).
     *
     * @details
     * The update is validated by @ref idet::engine::IEngine::check_hot_update_.
     * Implementations typically update cached thresholds/limits used by decoding.
     *
     * @param cfg Proposed new configuration.
     * @return Status::Ok() on success or an error status if update is not allowed.
     */
    Status update_hot(const DetectorConfig& cfg) noexcept override;

    /**
     * @brief Prepare fixed-shape bound inference and allocate per-context I/O.
     *
     * @details
     * This method is expected to:
     * - choose/align the effective input shape (engine-defined, if required),
     * - probe output indices and infer per-head layouts/shapes,
     * - allocate input and output buffers for each context,
     * - create per-context Ort::IoBinding and bind inputs/outputs.
     *
     * @param w Target input width (pixels), must be > 0.
     * @param h Target input height (pixels), must be > 0.
     * @param contexts Number of independent contexts to prepare (>= 1).
     * @return Status::Ok() on success; error status otherwise.
     */
    Status setup_binding(int w, int h, int contexts) noexcept override;

    /**
     * @brief Release bound inference resources and return to unbound mode.
     *
     * @note Safe to call multiple times.
     */
    void unset_binding() noexcept override;

    /**
     * @brief Run inference in unbound mode (per-call allocations).
     *
     * @param bgr Input image in BGR format (CV_8UC3).
     * @return Detections in original image coordinates or an error status.
     */
    Result<std::vector<algo::Detection>> infer_unbound(const cv::Mat& bgr) noexcept override;

    /**
     * @brief Run inference in bound mode using a prepared binding context.
     *
     * @param bgr Input image in BGR format (CV_8UC3).
     * @param ctx_idx Index of binding context in [0, bound_contexts()).
     * @return Detections in original image coordinates or an error status.
     *
     * @pre @ref binding_ready() is true.
     */
    Result<std::vector<algo::Detection>> infer_bound(const cv::Mat& bgr, int ctx_idx) noexcept override;

  private:
    /**
     * @brief Internal classification/bbox output layout tags for SCRFD exports.
     *
     * @details
     * This enum differentiates the common export shapes observed in practice.
     * Layout is inferred per head and per output kind (score vs bbox).
     */
    enum class Layout : std::uint8_t {
        Unknown = 0,

        /** @brief Score map in channels-first layout: [1,C,H,W]. */
        Score_CHW,

        /** @brief Flat scores: [1,N,C] or [1,N,1]. */
        Score_Flat,

        /** @brief Score plane: [H,W] or [1,H,W]. */
        Score_HW,

        /** @brief BBox map in channels-first layout: [1,4,H,W]. */
        BBox_CHW,

        /** @brief Flat bboxes: [1,N,4]. */
        BBox_Flat,

        /** @brief Per-pixel boxes: [H,W,4] or [1,H,W,4]. */
        BBox_HW4
    };

    /**
     * @brief Per-stride head metadata with inferred tensor interpretation.
     *
     * @details
     * Stores the ONNX output indices for the head's score and bbox tensors,
     * along with inferred shapes and decoder-relevant parameters (H/W/anchors/channels).
     */
    struct Head {
        int stride = 0;

        int score_idx = -1;
        int bbox_idx = -1;

        std::vector<int64_t> score_shape;
        std::vector<int64_t> bbox_shape;

        Layout score_layout = Layout::Unknown;
        Layout bbox_layout = Layout::Unknown;

        int Hs = 0;
        int Ws = 0;
        int anchors = 1;
        int score_ch = 1;
    };

    /**
     * @brief Per-context bound-mode resources.
     *
     * @details
     * Each context owns:
     * - input CHW buffer,
     * - output buffers (one per model output), stored in the same order as bound output indices,
     * - Ort::Value wrappers for outputs,
     * - Ort::IoBinding instance used for fast-path inference.
     */
    struct BoundCtx {
        std::vector<float> in;                ///< CHW input buffer
        std::vector<std::vector<float>> outs; ///< raw outputs in [score,bbox,score,bbox,...] order
        std::vector<Ort::Value> out_tensors;  ///< ORT tensor wrappers for outs

        std::unique_ptr<Ort::IoBinding> binding;
        Ort::Value in_tensor{nullptr};
    };

    /** @brief Cache cfg_.infer hot fields into POD members for fast access in the hot path. */
    void cache_hot_() noexcept;

    /** @brief Initialize model input/output node names (ORT metadata). */
    void init_io_names_();

    /**
     * @brief Probe output indices and infer per-head layouts for a fixed input shape.
     *
     * @param in_h Effective input height.
     * @param in_w Effective input width.
     * @param heads Output vector to fill with inferred head metadata.
     * @return Status::Ok() on success; error status otherwise.
     */
    Status probe_heads_layout_(int in_h, int in_w, std::vector<Head>* heads) noexcept;

    /**
     * @brief Fill CHW float input tensor from a BGR image with SCRFD normalization.
     *
     * @param dst Destination CHW buffer (size = 3 * in_h * in_w).
     * @param in_w Effective input width.
     * @param in_h Effective input height.
     * @param bgr Source image (CV_8UC3).
     */
    void fill_input_chw_(float* dst, int in_w, int in_h, const cv::Mat& bgr) const;

    /**
     * @brief Run ORT in unbound mode and return all model outputs.
     *
     * @details
     * Performs preprocessing and returns raw ORT outputs. Also reports the geometric
     * scale factors to map detections back to original image coordinates.
     *
     * @param bgr Input BGR image.
     * @param force_w If > 0, forces preprocessing to this width (engine-defined alignment may apply).
     * @param force_h If > 0, forces preprocessing to this height.
     * @param sx Output: scale X (in_w / orig_w).
     * @param sy Output: scale Y (in_h / orig_h).
     * @param in_w Output: effective preprocessed width.
     * @param in_h Output: effective preprocessed height.
     */
    Result<std::vector<Ort::Value>> run_unbound_(const cv::Mat& bgr, int force_w, int force_h, float& sx, float& sy,
                                                 int& in_w, int& in_h) noexcept;

    /** @brief Stable sigmoid helper for score decoding. */
    static inline float sigmoid_(float x) noexcept;

    /**
     * @brief Convert decoded bbox coordinates and score into a detection record.
     *
     * @param x1 Left.
     * @param y1 Top.
     * @param x2 Right.
     * @param y2 Bottom.
     * @param score Confidence score.
     */
    static algo::Detection rect_to_det_(float x1, float y1, float x2, float y2, float score);

    /**
     * @brief Decode model heads into detections.
     *
     * @details
     * Consumes per-head pointers to score tensors and bbox tensors, interpreting each
     * according to the inferred @ref Head::score_layout and @ref Head::bbox_layout.
     * Produces detections mapped to original image coordinates using scale factors.
     */
    std::vector<algo::Detection> decode_(const std::vector<Head>& heads, const std::vector<const float*>& score_ptrs,
                                         const std::vector<const float*>& bbox_ptrs, float sx, float sy, int orig_w,
                                         int orig_h) const;

  private:
    /** @brief ORT input node name (single input). */
    std::string in_name_;

    /** @brief ORT output node names (score/bbox outputs). */
    std::vector<std::string> out_names_;

    /** @brief Inferred per-stride head metadata (valid after probing/binding). */
    std::vector<Head> heads_;

    // cached hot params
    bool apply_sigmoid_ = false;
    float score_thr_ = 0.6f;
    int max_img_ = 960;
    int min_w_ = 10;
    int min_h_ = 10;

    /** @brief Subset/order of model outputs used for bound inference binding. */
    std::vector<int> bound_out_indices_;

    /** @brief Per-context bound-mode resources. */
    std::vector<BoundCtx> ctxs_;
};

} // namespace idet::engine
