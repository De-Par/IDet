/**
 * @file yolo.h
 * @ingroup idet_engine
 * @brief YOLO-family detector engine (ORT backend).
 *
 * @details
 * Implements a generic YOLO detector backend that supports the two output conventions
 * commonly seen across YOLO ONNX exports:
 *
 * 1. **In-graph NMS (end-to-end exports, e.g. YOLO26):**
 *    The model's graph already contains NonMaxSuppression and produces final
 *    detections. The output tensor has shape @c [B, K, 6] where each row is
 *    @c (x1, y1, x2, y2, score, class). An optional companion @c num_detections /
 *    @c num_dets tensor of shape @c [B] (or scalar) limits the number of valid rows.
 *    The engine reads these directly and only applies the inverse letterbox transform.
 *
 * 2. **Raw outputs (vanilla YOLOv5 / v6 / v8 / v11 / v12 / ...):**
 *    The model produces a single tensor of shape @c [B, 4+nc, N] (channels-first) or
 *    @c [B, N, 4+nc] (channels-last) where each anchor encodes
 *    @c (cx, cy, w, h, [obj], cls_0, ..., cls_{nc-1}) in network input pixel space.
 *    The engine decodes anchors, applies letterbox inverse, then runs the library's
 *    polygon NMS (@ref idet::algo::nms_poly) at the higher level via the detector
 *    pipeline (no NMS inside this engine to keep it composable with tiling).
 *
 * The engine probes the ONNX outputs at construction time and records the resolved
 * @ref idet::engine::YOLO::Mode and @ref idet::engine::YOLO::Layout. Both unbound and bound inference modes are
 * supported.
 */

#pragma once

#include "engine/engine.h"
#include "internal/letterbox_info.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace idet::engine {

/**
 * @brief YOLO-family engine implementation.
 *
 * @details
 * One engine class covers all YOLO variants. Output format is auto-detected at
 * construction time and stored as @ref idet::engine::YOLO::Mode + @ref idet::engine::YOLO::Layout.
 *
 * Thread-safety:
 * - @ref infer_unbound is safe for concurrent calls.
 * - @ref infer_bound is safe for concurrent calls only when each thread uses a
 *   unique context index.
 */
class YOLO final : public IEngine {
  public:
    /**
     * @brief Construct the YOLO engine.
     *
     * @param cfg Detector configuration. Must satisfy:
     * - cfg.task == @ref idet::Task::Cloth
     * - cfg.engine == @ref idet::EngineKind::Yolo
     *
     * @throws std::runtime_error if validation/session creation fails.
     *         The factory wrapper translates this into @ref idet::Status.
     */
    explicit YOLO(const DetectorConfig& cfg);

    /** @brief Engine kind identifier. */
    EngineKind kind() const noexcept override {
        return EngineKind::Yolo;
    }

    /** @brief Task domain handled by this engine (cloth). */
    Task task() const noexcept override {
        return Task::Cloth;
    }

    Status update_hot(const DetectorConfig& cfg) noexcept override;

    Status setup_binding(int w, int h, int contexts) noexcept override;
    void unset_binding() noexcept override;

    Result<std::vector<algo::Detection>> infer_unbound(const internal::BgrImageView& bgr_view) noexcept override;
    Result<std::vector<algo::Detection>> infer_bound(const internal::BgrImageView& bgr_view,
                                                     int ctx_idx) noexcept override;

  public:
    /**
     * @brief How the model represents detections at its output(s).
     */
    enum class Mode : std::uint8_t {
        /** @brief Unknown / not yet probed. */
        Unknown = 0,

        /**
         * @brief End-to-end model with in-graph NMS.
         *
         * @details
         * Output[0] shape: @c [B, K, 6] with rows @c (x1,y1,x2,y2,score,class) in
         * network input pixel space. An optional companion @c num_detections /
         * @c num_dets tensor of shape @c [B] (or scalar) limits valid rows.
         */
        InGraphNms = 1,

        /**
         * @brief Vanilla raw outputs that need external NMS.
         *
         * @details
         * Output[0] shape: @c [B, 4+nc, N] (channels-first) or @c [B, N, 4+nc]
         * (channels-last). Each anchor stores @c (cx,cy,w,h, [obj], cls_0..cls_{nc-1})
         * in network input pixel space. Class score = @c cls_max; YOLOv5-style exports may
         * also multiply by @c obj, while YOLOv8/v11-style exports usually fold objectness
         * into class scores; see @c has_objectness_.
         */
        Raw = 2,
    };

    /** @brief Layout for raw outputs. */
    enum class Layout : std::uint8_t {
        /** @brief Unknown / not applicable (e.g., InGraphNms). */
        Unknown = 0,

        /** @brief Channels-first: @c [B, 4+nc, N]. */
        ChannelsFirst = 1,

        /** @brief Channels-last:  @c [B, N, 4+nc]. */
        ChannelsLast = 2,
    };

  private:
    /** @brief Per-context bound-mode resources. */
    struct BoundCtx {
        std::vector<float> in;                ///< CHW input buffer
        std::vector<std::vector<float>> outs; ///< Output buffers, in same order as out_names_
        std::vector<Ort::Value> out_tensors;  ///< ORT tensor wrappers for outs

        std::unique_ptr<Ort::IoBinding> binding;
        Ort::Value in_tensor{nullptr};
    };

  private:
    /** @brief Refresh cached hot-update parameters. */
    void cache_hot_() noexcept;

    /** @brief Resolve input/output tensor names from ORT metadata. */
    void init_io_names_();

    /**
     * @brief Probe outputs to determine @ref Mode and @ref Layout.
     *
     * @details
     * Runs a single dummy inference at @p in_h x @p in_w and inspects shapes:
     * - If the primary output has rank 3 with the last dim == 6, mode is @ref Mode::InGraphNms.
     * - If the primary output has rank 3 and one of the non-batch dims is in @c [5, 1024]
     *   while the other is much larger, mode is @ref Mode::Raw with the appropriate layout.
     */
    Status probe_layout_(int in_h, int in_w) noexcept;

    /**
     * @brief Letterbox + normalize a BGR image into the CHW float input buffer.
     *
     * @details
     * YOLO models expect @c (x / 255.0) normalization with no mean subtraction and
     * letterboxing with pad value 114 by convention. The buffer @p dst must hold
     * at least @c 3 * in_h * in_w floats.
     *
     * @return Letterbox geometry used by decode side to invert the transform.
     */
    idet::internal::LetterboxInfo fill_input_chw_(float* dst, int in_w, int in_h,
                                                  const internal::BgrImageView& bgr) const;

    /**
     * @brief Compute the network input shape for a given source size.
     *
     * @details
     * If @c bound_w_ > 0 and @c bound_h_ > 0 (binding is set up), returns those
     * dimensions; otherwise picks an aspect-preserving size with the longer side
     * downscaled to <= @ref max_img_, both dims aligned up to 32.
     */
    void compute_input_size_(int orig_w, int orig_h, int& in_w, int& in_h) const noexcept;

    /**
     * @brief Decode model outputs into detections in original-image coordinates.
     *
     * @details
     * Dispatches to the appropriate routine based on the probed @ref Mode. The
     * library applies external polygon NMS at the detector pipeline level for
     * @ref Mode::Raw outputs; this method only applies score thresholding and the
     * inverse letterbox transform.
     */
    std::vector<algo::Detection> decode_(const std::vector<Ort::Value>& outs, const idet::internal::LetterboxInfo& lb,
                                         int orig_w, int orig_h) const;

    std::vector<algo::Detection> decode_in_graph_nms_(const std::vector<Ort::Value>& outs,
                                                      const idet::internal::LetterboxInfo& lb, int orig_w,
                                                      int orig_h) const;

    std::vector<algo::Detection> decode_raw_(const std::vector<Ort::Value>& outs,
                                             const idet::internal::LetterboxInfo& lb, int orig_w, int orig_h) const;

  public:
    /**
     * @brief Decode a raw YOLO output buffer into detections (testable helper).
     *
     * @details
     * Pure-math counterpart of @c decode_raw_ that operates on a contiguous float buffer
     * without involving an Ort session. Exposed for unit testing and for any caller that
     * wants to drive the YOLO post-processing from a custom inference path.
     *
     * Buffer layout:
     * - @c ChannelsFirst : @c data[c * N + i] is feature @p c of anchor @p i
     * - @c ChannelsLast  : @c data[i * feat + c]
     * Where @c feat = 4 + (has_objectness ? 1 : 0) + num_classes.
     *
     * Each anchor encodes @c (cx, cy, w, h, [obj], cls_0..cls_{nc-1}) in network-input pixel
     * space; the inverse letterbox transform brings boxes into the original image space.
     *
     * @param data           Raw float pointer to the model output.
     * @param N              Number of anchors / detections in the buffer.
     * @param num_classes    Class count (>= 1).
     * @param has_objectness If true, treat channel 4 as objectness; multiply into score.
     * @param layout         Channel layout (channels-first vs channels-last).
     * @param apply_sigmoid  If true, apply sigmoid to class/objectness logits.
     * @param score_thr      Minimum score to keep a detection.
     * @param lb             Letterbox geometry (uniform scale + padding).
     * @param orig_w         Original image width (clamp limit).
     * @param orig_h         Original image height (clamp limit).
     * @param min_w          Minimum bbox width in original-image pixels (0 disables).
     * @param min_h          Minimum bbox height in original-image pixels (0 disables).
     */
    static std::vector<algo::Detection> decode_raw_buffer(const float* data, std::int64_t N, int num_classes,
                                                          bool has_objectness, Layout layout, bool apply_sigmoid,
                                                          float score_thr, const idet::internal::LetterboxInfo& lb,
                                                          int orig_w, int orig_h, int min_w, int min_h);

  private:
    std::string in_name_;
    std::vector<std::string> out_names_;

    Mode mode_ = Mode::Unknown;
    Layout layout_ = Layout::Unknown;
    int num_classes_ = 0;
    bool has_objectness_ = false; ///< true for YOLOv5-style outputs with explicit objectness slot

    int probed_in_w_ = 0;
    int probed_in_h_ = 0;
    int num_detections_idx_ = -1; ///< output index of num_detections tensor for InGraphNms (or -1)
    int detections_idx_ = -1;     ///< output index of detections tensor for InGraphNms

    // cached hot params
    bool apply_sigmoid_ = false; ///< if true, treat raw class scores as logits and apply sigmoid
    float score_thr_ = 0.25f;
    int max_img_ = 960;
    int min_w_ = 0;
    int min_h_ = 0;

    std::vector<BoundCtx> ctxs_;
    std::vector<int64_t> bound_in_shape_;
};

} // namespace idet::engine
