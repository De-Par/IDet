/**
 * @file engine.h
 * @ingroup idet_engine
 * @brief Common ORT-based engine interface and session utilities.
 *
 * @details
 * This header defines @ref idet::engine::IEngine, a polymorphic interface implemented by
 * concrete model backends (DBNet, SCRFD, ...). It also provides small shared utilities
 * used by those engines (e.g., ORT environment/session helpers).
 *
 * Key concepts:
 * - **Unbound inference**: per-call tensor preparation (more flexible; may allocate).
 * - **Bound inference**: fixed-shape, preallocated per-context buffers + binding state
 *   (fast path for repeated inference).
 *
 * @note
 * This is an internal header (located under `src/lib/idet/engine`). It is not part of the
 * public installed API and may change between releases without notice.
 */

#pragma once

#include "algo/geometry.h"
#include "idet.h"
#include "internal/opencv_headers.h" // IWYU pragma: keep
#include "internal/ort_headers.h"    // IWYU pragma: keep
#include "status.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace idet::engine {

/**
 * @brief Abstract engine interface for model inference.
 *
 * @details
 * Engines encapsulate:
 * - preprocessing (OpenCV-based BGR input handling),
 * - ONNX Runtime session invocation,
 * - decoding raw model outputs into intermediate detections (@ref idet::algo::Detection).
 *
 * The higher-level detector facade is responsible for:
 * - converting the public API image representation into the required format,
 * - applying optional tiling and global NMS/merging policies,
 * - exposing final results in the public geometry types.
 *
 * @par Inference modes
 * - @ref infer_unbound : per-call I/O (flexible; may allocate and re-create bindings).
 * - @ref infer_bound   : preallocated and prebound I/O for fixed shape (fast path).
 *
 * @par Binding and contexts
 * @ref setup_binding prepares @p contexts independent bound contexts. Each context is expected
 * to have its own I/O buffers and binding objects so that multiple threads can run bound
 * inference concurrently by using distinct @p ctx_idx values.
 *
 * @par Thread-safety
 * - @ref infer_unbound is expected to be safe for concurrent calls (implementation-dependent).
 * - @ref infer_bound is expected to be safe only if each concurrent caller uses a distinct
 *   context index and the underlying engine does not share mutable per-call state.
 */
class IEngine {
  public:
    /**
     * @brief Virtual destructor.
     *
     * @note
     * Must be noexcept to avoid exceptions during stack unwinding across shared library boundaries.
     */
    virtual ~IEngine() noexcept = default;

    /**
     * @brief Engine kind identifier (e.g., DBNet or SCRFD).
     */
    virtual EngineKind kind() const noexcept = 0;

    /**
     * @brief Task domain handled by this engine (text or face).
     *
     * @note
     * Typically derived from @ref kind(), but exposed explicitly for clarity.
     */
    virtual Task task() const noexcept = 0;

    /**
     * @brief Current configuration snapshot used by the engine.
     *
     * @details
     * Returns a reference to the internally stored configuration. The engine may update
     * this configuration during @ref update_hot and/or during binding setup calls
     * (for example, to cache derived parameters).
     *
     * @warning
     * The returned reference remains valid only while the engine instance is alive.
     */
    const DetectorConfig& config() const noexcept {
        return cfg_;
    }

    /**
     * @brief Whether I/O binding has been prepared and is ready for bound inference.
     *
     * @details
     * When true, @ref infer_bound is expected to be available (given a valid context index).
     * When false, @ref infer_bound should return an error status.
     */
    bool binding_ready() const noexcept {
        return binding_ready_;
    }

    /**
     * @brief Bound input width in pixels for the prepared binding shape.
     *
     * @details
     * This is the width passed to @ref setup_binding, potentially after engine-specific alignment.
     */
    int bound_w() const noexcept {
        return bound_w_;
    }

    /**
     * @brief Bound input height in pixels for the prepared binding shape.
     *
     * @details
     * This is the height passed to @ref setup_binding, potentially after engine-specific alignment.
     */
    int bound_h() const noexcept {
        return bound_h_;
    }

    /**
     * @brief Number of independent binding contexts prepared for bound inference.
     *
     * @details
     * A "context" typically represents a set of pre-allocated I/O buffers and binding objects
     * that can be used independently by concurrent callers.
     *
     * @return Number of contexts, as requested by @ref setup_binding (or adjusted by implementation).
     */
    int bound_contexts() const noexcept {
        return contexts_;
    }

    /**
     * @brief Apply a hot configuration update without recreating the ONNX Runtime session.
     *
     * @details
     * This method is intended for parameters that can be changed at runtime without changing
     * model shapes or session-wide invariants, such as:
     * - detection thresholds (e.g., `box_thresh`, `bin_thresh`),
     * - NMS parameters,
     * - verbosity flags,
     * - other postprocessing knobs.
     *
     * Typical restrictions:
     * - Model path must not change.
     * - Task/engine kind must not change.
     * - Runtime policy (threading/affinity) is usually not hot-updatable because it often requires
     *   recreating ORT thread pools and/or the session.
     *
     * The base class provides helpers @ref check_hot_update_ and @ref apply_hot_common_.
     *
     * @param cfg Proposed new configuration.
     * @return @ref Status::Ok() on success, otherwise an error status explaining why the update was rejected.
     */
    virtual Status update_hot(const DetectorConfig& cfg) noexcept = 0;

    /**
     * @brief Prepare engine for bound inference at a fixed input shape and with multiple contexts.
     *
     * @details
     * Implementations are expected to:
     * - allocate and/or bind I/O buffers for the specified input dimensions,
     * - create per-context binding state to allow concurrent calls to @ref infer_bound,
     * - set @ref binding_ready_ to true on success and fill @ref bound_w_, @ref bound_h_, @ref contexts_.
     *
     * @param w Target input width in pixels (must be > 0).
     * @param h Target input height in pixels (must be > 0).
     * @param contexts Number of contexts to prepare (must be > 0).
     * @return @ref Status::Ok() on success, error status otherwise.
     *
     * @note
     * Some engines may internally align dimensions (e.g., to multiples of 32). In that case
     * @ref bound_w() / @ref bound_h() should reflect the effective bound shape.
     */
    virtual Status setup_binding(int w, int h, int contexts) noexcept = 0;

    /**
     * @brief Tear down any prepared binding state and return to unbound mode.
     *
     * @details
     * Implementations should release per-context buffers/bindings and set:
     * - @ref binding_ready_ = false
     * - bound dimensions and context count to 0 (or other safe defaults).
     *
     * @note
     * This operation should be safe to call multiple times.
     */
    virtual void unset_binding() noexcept = 0;

    /**
     * @brief Run inference in unbound mode (no pre-bound I/O).
     *
     * @details
     * Unbound mode typically:
     * - accepts a wide range of input sizes (subject to model constraints),
     * - prepares input tensors per call (and may allocate temporary buffers),
     * - is simpler but may have higher per-call overhead compared to bound mode.
     *
     * Output:
     * - returns engine-level detections (e.g., boxes/landmarks) in image coordinates as
     *   @ref idet::algo::Detection objects.
     *
     * @param bgr Input image (expected BGR, `CV_8UC3`).
     * @return Result with detections on success, or an error status on failure.
     */
    virtual Result<std::vector<algo::Detection>> infer_unbound(const cv::Mat& bgr) noexcept = 0;

    /**
     * @brief Run inference in bound mode using a pre-prepared binding context.
     *
     * @details
     * Bound mode requires a successful @ref setup_binding call. It typically:
     * - uses pre-allocated and pre-bound I/O buffers for a fixed input shape,
     * - reduces allocations and repeated shape handling,
     * - enables parallel inference by using independent contexts.
     *
     * Context index:
     * - @p ctx_idx must be in `[0, bound_contexts())`.
     * - Each context should be used exclusively by one concurrent caller at a time.
     *
     * @param bgr Input image (expected BGR, `CV_8UC3`). Size may be required to match bound shape.
     * @param ctx_idx Binding context index to use.
     * @return Result with detections on success, or an error status on failure.
     *
     * @pre @ref binding_ready() is true.
     */
    virtual Result<std::vector<algo::Detection>> infer_bound(const cv::Mat& bgr, int ctx_idx) noexcept = 0;

  protected:
    /**
     * @brief Protected constructor for derived engines.
     *
     * @details
     * Derived classes initialize the base with a configuration snapshot and a logging identifier
     * used to initialize the process-wide ONNX Runtime environment via @ref global_env_.
     *
     * Typical responsibilities of a concrete engine constructor (defined in a .cpp):
     * - copy/store @p cfg into @ref cfg_,
     * - initialize @ref env_ reference using @ref global_env_,
     * - configure @ref so_ (Ort::SessionOptions) according to runtime policy,
     * - create the ORT session via @ref create_session_.
     *
     * @param cfg Engine configuration.
     * @param log_id Logging identifier for ORT environment (may be null/empty).
     */
    explicit IEngine(const DetectorConfig& cfg, const char* log_id);

    /**
     * @brief Validate whether a config update is eligible for hot update.
     *
     * @details
     * This helper is expected to enforce invariants such as:
     * - task/engine kind must not change,
     * - model path must not change,
     * - runtime policy changes may be disallowed.
     *
     * @param next Proposed next configuration.
     * @return @ref Status::Ok() if hot update is allowed, error status otherwise.
     */
    Status check_hot_update_(const DetectorConfig& next) const noexcept;

    /**
     * @brief Apply common parts of a hot configuration update.
     *
     * @details
     * Updates shared fields in @ref cfg_ that are safe to modify without recreating the ORT session
     * and without rebinding. Concrete engines typically call this after @ref check_hot_update_
     * succeeds and then update their own cached parameters accordingly.
     *
     * @param next Proposed next configuration.
     */
    void apply_hot_common_(const DetectorConfig& next) noexcept;

    /**
     * @brief Access a process-wide ONNX Runtime environment singleton.
     *
     * @details
     * ONNX Runtime uses an environment object to manage logging and global state. This helper provides
     * a single process-wide instance:
     * - Logging level is fixed to `ORT_LOGGING_LEVEL_ERROR`.
     * - Logging identifier is taken from @p log_id if non-empty, otherwise `"idet"`.
     *
     * @warning
     * The first call constructs the static environment. Subsequent calls ignore different @p log_id
     * values because the singleton already exists. If distinct per-engine log identifiers are required,
     * the design would need to change.
     *
     * @param log_id Optional logging identifier string (may be null/empty).
     * @return Reference to the global ORT environment.
     */
    static Ort::Env& global_env_(const char* log_id) {
        static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, (log_id && log_id[0]) ? log_id : "idet");
        return env;
    }

    /**
     * @brief Create an ONNX Runtime session from a model path or (optionally) an embedded blob.
     *
     * @details
     * The implementation is expected to:
     * - resolve the model source:
     *   - from filesystem @p model_path, or
     *   - from an embedded model blob selected by @p engine_kind (if the build enables embedding),
     * - initialize @ref session_ with @ref env_ and @ref so_.
     *
     * Errors should be converted into @ref idet::Status with actionable messages.
     *
     * @param model_path Path to an ONNX model file (may be empty if using embedded model).
     * @param engine_kind Engine kind hint for selecting an embedded model (if supported).
     * @return @ref Status::Ok() on success, error status otherwise.
     */
    Status create_session_(const std::string& model_path, EngineKind engine_kind = EngineKind::None) noexcept;

  protected:
    /**
     * @brief Stored configuration snapshot for the engine instance.
     *
     * @details
     * This is the authoritative configuration state for the engine. It may be updated via
     * @ref update_hot for parameters that are safe to change at runtime.
     */
    DetectorConfig cfg_;

    /**
     * @brief Indicates whether bound-mode resources are initialized and usable.
     *
     * @see setup_binding
     * @see unset_binding
     * @see infer_bound
     */
    bool binding_ready_ = false;

    /**
     * @brief Effective bound input width for bound inference.
     *
     * @see bound_w
     */
    int bound_w_ = 0;

    /**
     * @brief Effective bound input height for bound inference.
     *
     * @see bound_h
     */
    int bound_h_ = 0;

    /**
     * @brief Number of prepared bound contexts.
     *
     * @see bound_contexts
     */
    int contexts_ = 0;

    /**
     * @brief Reference to the process-wide ONNX Runtime environment.
     *
     * @details
     * Initialized in the constructor using @ref global_env_. The referenced environment has
     * static storage duration and outlives all engine instances.
     */
    Ort::Env& env_;

    /**
     * @brief ONNX Runtime session options for this engine.
     *
     * @details
     * Contains configuration such as threading options, graph optimizations, execution providers,
     * arena settings, etc. Concrete engines typically configure this based on @ref idet::RuntimePolicy.
     */
    Ort::SessionOptions so_;

    /**
     * @brief ONNX Runtime session handle (may be null before creation).
     *
     * @details
     * The session is created by @ref create_session_.
     */
    Ort::Session session_{nullptr};

    /**
     * @brief Default ONNX Runtime allocator helper.
     *
     * @details
     * Used for allocating ORT-managed buffers and for allocator-backed queries.
     */
    Ort::AllocatorWithDefaultOptions alloc_;
};

} // namespace idet::engine
