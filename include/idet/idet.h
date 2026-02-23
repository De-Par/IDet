/**
 * @file idet.h
 * @brief Public API for IDet: detector configuration, runtime policy, and detection entry points.
 *
 * This header is the primary public interface of IDet. It defines:
 * - High-level tasks and engine kinds (@ref idet::Task, @ref idet::EngineKind).
 * - Geometry primitives for detection results (@ref idet::Point2f, @ref idet::Quad).
 * - Configuration objects for inference and runtime (@ref idet::InferenceOptions, @ref idet::RuntimePolicy,
 *   @ref idet::DetectorConfig).
 * - The main detector facade class (@ref idet::Detector) with an ABI-friendly backend
 *   (opaque implementation pointer plus an internal vtable).
 *
 * Error handling:
 * - Most APIs return @ref idet::Status or @ref idet::Result<T>.
 * - The convenience constructor `Detector(const DetectorConfig&)` throws @c std::runtime_error on failure.
 *
 * Threading:
 * - Runtime threading and global settings are controlled via @ref idet::RuntimePolicy.
 * - Some detection methods may require prepared bindings and/or explicit context indices,
 *   depending on the engine and runtime configuration.
 *
 * @ingroup idet_api
 */

/**
 * @defgroup idet_api IDet Public API
 * @brief Public API types and entry points for IDet.
 * @{
 */

#pragma once

#include "export.h"
#include "image.h"
#include "status.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace idet {

/**
 * @brief High-level detection task category.
 *
 * A task identifies what kind of objects the detector is expected to produce.
 * It is also used to validate configuration consistency (e.g., the engine kind must
 * match the task category).
 */
enum class Task : std::uint8_t {
    /** No task selected / invalid. */
    None = 0,
    /** Text detection task (e.g., DBNet). */
    Text = 1,
    /** Face detection task (e.g., SCRFD). */
    Face = 2,
};

/**
 * @brief Concrete engine implementation kind.
 *
 * The engine kind selects the underlying model plus preprocessing/postprocessing pipeline.
 * It must be compatible with the chosen @ref idet::Task.
 */
enum class EngineKind : std::uint8_t {
    /** No engine selected / invalid. */
    None = 0,
    /** DBNet text detector engine. */
    DBNet = 1,
    /** SCRFD face detector engine. */
    SCRFD = 2,
};

/**
 * @brief Maps an engine kind to its corresponding high-level task.
 *
 * @param kind Engine kind.
 * @return Task category implied by @p kind, or @ref Task::None for unknown kinds.
 */
constexpr Task engine_task(EngineKind kind) noexcept {
    switch (kind) {
    case EngineKind::DBNet:
        return Task::Text;
    case EngineKind::SCRFD:
        return Task::Face;
    default:
        return Task::None;
    }
}

/**
 * @brief 2D point in floating-point pixel coordinates.
 *
 * Coordinates are typically expressed in the input image coordinate system:
 * - origin is implementation-defined (commonly top-left of the input image),
 * - X grows to the right, Y grows downward (common in image processing).
 */
struct Point2f {
    /** @brief X coordinate (pixels). */
    float x = 0.0f;
    /** @brief Y coordinate (pixels). */
    float y = 0.0f;
};

/**
 * @brief Quadrilateral defined by 4 corner points.
 *
 * Corner ordering is engine/postprocess dependent. Unless documented otherwise,
 * do not assume a specific winding order (clockwise/counterclockwise) or a specific
 * starting corner.
 */
using Quad = std::array<Point2f, 4>;

/** @brief A dynamic list of quadrilateral detections. */
using VecQuad = std::vector<Quad>;

/**
 * @brief Discrete grid specification (rows x cols).
 *
 * Used to describe fixed input dimension grids and tiling layouts.
 */
struct GridSpec {
    /** @brief Number of rows. Typically >= 1. */
    int rows = 1;
    /** @brief Number of columns. Typically >= 1. */
    int cols = 1;
};

/**
 * @brief Inference and postprocessing options for the selected engine.
 *
 * This structure aggregates parameters affecting preprocessing, model invocation, and
 * postprocessing. Some fields are engine-specific (e.g., text-specific thresholds).
 *
 * Unless otherwise stated, values are interpreted in the input image coordinate space.
 */
struct InferenceOptions {
    /**
     * @brief Whether to apply a sigmoid activation to model outputs.
     *
     * Some models export logits; enabling this flag applies sigmoid before thresholding.
     */
    bool apply_sigmoid = false;

    /**
     * @brief Whether to pre-bind I/O (and potentially allocate buffers) ahead of time.
     *
     * Binding may improve performance by avoiding repeated allocations and shape negotiation.
     * Binding is typically configured via @ref idet::Detector::prepare_binding.
     */
    bool bind_io = false;

    /**
     * @brief Binarization threshold for text probability maps.
     *
     * Used by text detectors to convert probability maps into binary masks.
     */
    float bin_thresh = 0.3f;

    /**
     * @brief Box confidence threshold for accepting detections.
     *
     * Detections below this threshold may be discarded during postprocessing.
     */
    float box_thresh = 0.5f;

    /**
     * @brief Unclip ratio for expanding detected text boxes.
     *
     * Typically used by DBNet-style postprocessing to expand contours/polygons.
     */
    float unclip = 1.0f;

    /**
     * @brief Maximum image size used for resizing before inference.
     *
     * Engines may resize the longest side (or apply a similar heuristic) to be <= this value.
     * Exact behavior is implementation-defined.
     */
    int max_img_size = 960;

    /** @brief Minimum ROI width (in pixels) for keeping a detection. */
    int min_roi_size_w = 5;

    /** @brief Minimum ROI height (in pixels) for keeping a detection. */
    int min_roi_size_h = 5;

    /**
     * @brief Fixed input grid dimension override (rows x cols).
     *
     * When non-zero, engines may use a fixed input dimension policy rather than dynamic sizing.
     * Interpretation is implementation-defined and should be validated via @ref DetectorConfig::validate.
     *
     * Default is `{0, 0}` to indicate "not set / use engine default".
     */
    GridSpec fixed_input_dim{0, 0};

    /**
     * @brief Tiling grid dimension (rows x cols).
     *
     * When tiling is enabled, the image may be split into `rows * cols` tiles and processed separately.
     */
    GridSpec tiles_dim{1, 1};

    /**
     * @brief Tile overlap ratio in [0, 1).
     *
     * Overlap is used to reduce boundary artifacts and improve continuity across tiles.
     */
    float tile_overlap = 0.1f;

    /**
     * @brief IoU threshold for Non-Maximum Suppression (NMS).
     *
     * Used to merge overlapping detections produced by tiling or multi-head outputs.
     */
    float nms_iou = 0.3f;

    /**
     * @brief Fast IoU option for NMS / overlap checks.
     *
     * true -> AABB IoU approximation (faster, less accurate for rotated quads).
     * false -> polygon IoU (exact for convex quads, slower).
     */
    bool use_fast_iou = false;
};

/**
 * @brief NUMA-aware memory policy hint for runtime setup.
 *
 * This setting controls how the runtime prefers to place or bind memory on NUMA systems.
 * Actual behavior is platform- and implementation-dependent. On non-NUMA platforms, this
 * may have no effect.
 */
enum class NumaMemPolicy {
    /** Prefer lowest latency (e.g., local allocations and locality-first behavior). */
    Latency = 0,
    /** Prefer throughput (may allow broader placement to reduce contention). */
    Throughput = 1,
    /** Prefer strict placement/binding where supported; may fail if constraints cannot be met. */
    Strict = 2
};

/**
 * @brief Runtime policy controlling threading, binding, and global runtime behavior.
 *
 * This structure configures execution characteristics, typically affecting ONNX Runtime and
 * tile-parallel execution.
 *
 * @note
 * Actual effect depends on the selected execution provider(s) and the internal runtime implementation.
 */
struct RuntimePolicy {
    /** @brief ONNX Runtime intra-op thread count (operator-level parallelism). */
    int ort_intra_threads = 1;

    /** @brief ONNX Runtime inter-op thread count (graph-level parallelism). */
    int ort_inter_threads = 1;

    /**
     * @brief OpenMP thread count used for tile-parallel execution.
     *
     * Relevant when tiling is enabled and the library uses OpenMP to process tiles concurrently.
     */
    int tile_omp_threads = 1;

    /**
     * @brief Enables "soft" memory binding policies when applicable.
     *
     * This may affect NUMA locality and memory allocation strategies in runtime setup.
     * Exact behavior is implementation-defined.
     */
    bool soft_mem_bind = true;

    /** @brief Memory placement policy hint for NUMA-capable systems. */
    NumaMemPolicy numa_mem_policy = NumaMemPolicy::Latency;

    /**
     * @brief Suppresses OpenCV internal threading globally.
     *
     * Often used to avoid thread oversubscription when the application already controls parallelism.
     *
     * @warning
     * If this toggles a global OpenCV setting, it may affect other OpenCV users within the same process.
     */
    bool suppress_opencv = true; // globally
};

/**
 * @brief Configuration for creating and updating an @ref idet::Detector instance.
 *
 * This structure binds together:
 * - the high-level task and concrete engine kind,
 * - inference options,
 * - runtime policy,
 * - model path and logging verbosity.
 *
 * Typical workflow:
 *  1) Fill the struct (or call @ref setup).
 *  2) Call @ref validate (recommended).
 *  3) Create a detector via @ref idet::Detector::create or @ref create_detector.
 *
 * The config can also be applied to an existing detector via @ref idet::Detector::update_config.
 */
struct DetectorConfig final {
    /** @brief Selected high-level task. Must match @ref engine. */
    Task task = Task::None;

    /** @brief Selected engine kind (model/pipeline). Must match @ref task. */
    EngineKind engine = EngineKind::None;

    /** @brief Inference and postprocessing options. */
    InferenceOptions infer{};

    /** @brief Runtime threading and global policy options. */
    RuntimePolicy runtime{};

    /** @brief Filesystem path to the model file (e.g., ONNX). */
    std::string model_path{};

    /** @brief Enables verbose logging in library internals (implementation-defined). */
    bool verbose = true;

    /**
     * @brief Validates the configuration for internal consistency and supported values.
     *
     * @return @ref Status::Ok() if the configuration is valid, otherwise a non-OK status describing
     *         the first detected issue.
     */
    [[nodiscard]] Status validate() const noexcept;

    /**
     * @brief Convenience factory to build a minimal config for a given task and model path.
     *
     * @param task Desired task category.
     * @param model_path Filesystem path to the model.
     * @return DetectorConfig with basic fields set. Additional tuning may be required.
     *
     * @note
     * The chosen engine kind may be inferred or left to defaults depending on the implementation.
     */
    static DetectorConfig setup(Task task, std::string model_path);
};

namespace detail {
/** @brief Internal detector vtable type (not part of the public API). */
struct DetectorVTable;
} // namespace detail

/**
 * @brief Main detector facade providing a stable public API.
 *
 * This class uses an opaque implementation pointer (`impl_`) and an internal vtable (`vtbl_`)
 * to hide implementation details and keep the public interface stable.
 *
 * Creation:
 * - Use @ref create (recommended, non-throwing) or @ref create_detector.
 * - Alternatively, use the throwing convenience constructor.
 *
 * Lifetime:
 * - A default-constructed detector is empty/invalid.
 * - Use `if (detector)` to check validity.
 *
 * Copying:
 * - Copy is disabled; move is supported.
 *
 * @thread_safety
 * Thread safety depends on the underlying engine implementation and runtime configuration.
 * Unless explicitly documented otherwise, assume that calling non-const operations concurrently
 * on the same instance is not safe.
 */
class IDET_API Detector final {
  public:
    /** @brief Constructs an empty (invalid) detector. */
    Detector() noexcept = default;

    /**
     * @brief Destroys the detector and releases implementation resources.
     *
     * The destructor is defined out-of-line to keep the public header lightweight and to hide
     * implementation details.
     */
    ~Detector() noexcept;

    /**
     * @brief Move-constructs a detector, transferring ownership of implementation.
     * @param other Source detector (moved-from becomes empty/invalid).
     */
    Detector(Detector&& other) noexcept;

    /**
     * @brief Move-assigns a detector, releasing current resources and taking over @p other.
     * @param other Source detector (moved-from becomes empty/invalid).
     * @return Reference to this instance.
     */
    Detector& operator=(Detector&& other) noexcept;

    /** @brief Copy construction is disabled (detector owns an opaque implementation). */
    Detector(const Detector&) = delete;

    /** @brief Copy assignment is disabled. */
    Detector& operator=(const Detector&) = delete;

    /**
     * @brief Convenience throwing constructor that creates a detector from @p config.
     *
     * Internally calls @ref Detector::create. If creation fails, throws @c std::runtime_error with
     * the error status message.
     *
     * @param config Detector configuration.
     * @throws std::runtime_error if detector creation fails.
     *
     * @warning
     * This constructor introduces an exception boundary. Prefer @ref create for exception-free code.
     */
    explicit Detector(const DetectorConfig& config) {
        auto r = Detector::create(config);
        if (!r.ok()) throw std::runtime_error(r.status().message);
        *this = std::move(r.value());
    }

    /**
     * @brief Checks whether this detector instance is valid and ready for use.
     * @return True if the internal implementation is present, otherwise false.
     */
    explicit operator bool() const noexcept;

    /**
     * @brief Returns the task category of this detector.
     * @return Task associated with the configured engine, or @ref Task::None if invalid.
     */
    Task task() const noexcept;

    /**
     * @brief Returns the engine kind of this detector.
     * @return Engine kind, or @ref EngineKind::None if invalid.
     */
    EngineKind engine() const noexcept;

    /**
     * @brief Resets the detector to an empty state and releases held resources.
     *
     * After reset, `operator bool()` is expected to return false.
     */
    void reset() noexcept;

    /**
     * @brief Creates a detector instance from the given configuration.
     *
     * @param config Detector configuration.
     * @return Result containing a valid detector on success, or an error status on failure.
     *
     * @note
     * It is recommended to call @ref DetectorConfig::validate before creation.
     */
    static Result<Detector> create(const DetectorConfig& config) noexcept;

    /**
     * @brief Updates the configuration of an existing detector instance.
     *
     * The implementation may reconfigure preprocessing/postprocessing thresholds, tiling, runtime
     * options, or engine-specific parameters. Some changes may require rebinding I/O or
     * reinitializing runtime resources.
     *
     * @param config New configuration to apply.
     * @return @ref Status::Ok() on success, otherwise an error status.
     */
    Status update_config(const DetectorConfig& config) noexcept;

    /**
     * @brief Prepares bound I/O (and optionally per-context resources) for a fixed input size.
     *
     * Engines that support bound inference can pre-allocate buffers for a given input resolution.
     * This may reduce per-call overhead for repeated inference on same-sized images.
     *
     * @param width Input width in pixels.
     * @param height Input height in pixels.
     * @param contexts Number of independent contexts to prepare (e.g., per-thread contexts).
     * @return @ref Status::Ok() on success, otherwise an error status.
     *
     * @note
     * The meaning of "contexts" is implementation-defined. Commonly it represents the number of
     * independent inference contexts/bindings that can be used via @ref detect_bound.
     */
    Status prepare_binding(int width, int height, int contexts) noexcept;

    /**
     * @brief Runs detection on the provided image using an unbound (or internally managed) context.
     *
     * @param image Input image. Must be a valid @ref idet::Image view.
     * @return Result containing detected quadrilaterals on success, or an error status on failure.
     *
     * @note
     * Depending on the runtime configuration, this call may allocate temporary buffers.
     */
    Result<VecQuad> detect(const Image& image) noexcept;

    /**
     * @brief Runs detection using a pre-bound context index.
     *
     * Intended for high-throughput and/or multi-threaded scenarios where bindings are prepared
     * in advance via @ref prepare_binding and each thread uses its own @p ctx_idx.
     *
     * @param image Input image. Must be compatible with the prepared binding dimensions.
     * @param ctx_idx Context index in range `[0, contexts)` as configured by @ref prepare_binding.
     * @return Result containing detected quadrilaterals on success, or an error status on failure.
     */
    Result<VecQuad> detect_bound(const Image& image, int ctx_idx) noexcept;

  private:
    /**
     * @brief Opaque pointer to the implementation object (engine backend).
     *
     * Owned by the detector and released in the destructor/reset.
     */
    void* impl_ = nullptr;

    /**
     * @brief Pointer to an internal vtable describing implementation operations.
     *
     * The vtable allows calling engine-specific implementations without exposing their types.
     */
    const detail::DetectorVTable* vtbl_ = nullptr;
};

/**
 * @brief Convenience free function for creating a detector.
 *
 * @param config Detector configuration.
 * @return Result containing a valid detector on success, or an error status on failure.
 *
 * @note
 * This is equivalent to calling @ref idet::Detector::create.
 */
[[nodiscard]] IDET_API inline Result<Detector> create_detector(const DetectorConfig& config) noexcept {
    return Detector::create(config);
}

/**
 * @brief Applies the runtime policy to the current process/runtime environment.
 *
 * This function configures runtime-related global and per-runtime settings such as:
 * - ONNX Runtime thread counts (intra-op/inter-op),
 * - OpenMP affinity and binding (places/proc_bind) when applicable,
 * - optional memory binding policies,
 * - optional suppression of OpenCV internal threading (if enabled).
 *
 * @param policy Runtime policy parameters.
 * @param verbose Enables logging of applied settings (implementation-defined).
 * @return @ref Status::Ok() if the policy is applied successfully, otherwise an error status.
 *
 * @warning
 * Some settings may be process-global (e.g., OpenCV threading and some OpenMP configuration),
 * potentially affecting other components in the same process.
 */
[[nodiscard]] IDET_API Status setup_runtime_policy(const RuntimePolicy& policy, bool verbose = true) noexcept;

} // namespace idet

/** @} */ // end of group idet_api
