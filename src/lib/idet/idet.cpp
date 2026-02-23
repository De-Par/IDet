/**
 * @file idet.cpp
 * @ingroup idet
 * @brief Implementation of the public IDet detector facade and configuration validation.
 *
 * @details
 * This translation unit implements:
 * - @ref idet::DetectorConfig::setup and @ref idet::DetectorConfig::validate
 * - The public @ref idet::Detector PImpl/vtable facade (ABI-stable public surface)
 * - A private implementation class (`detail::DetectorImpl`) that owns the engine instance
 *   and orchestrates preprocessing, tiling, filtering, and NMS.
 *
 * ABI stability strategy:
 * - Public header does not expose implementation types.
 * - The detector holds `void* impl_` and an internal vtable pointer, so layout stays stable.
 *
 * Exception safety:
 * - Public APIs are largely `noexcept`.
 * - Internal code may throw (allocations, STL, third-party libs).
 * - This TU catches exceptions at the vtable boundary and converts them into @ref idet::Status.
 */

#include "idet.h"

#include "algo/nms.h"
#include "algo/tiling.h"
#include "engine/engine_factory.h"
#include "internal/cv_bgr.h"
#include "internal/opencv_headers.h" // IWYU pragma: keep
#include "platform/runtime_policy_setup.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <new>
#include <string>
#include <utility>

namespace idet {

namespace {

/**
 * @brief Checks whether a detection passes minimum width/height constraints.
 *
 * @details
 * Computes an axis-aligned bounding box around the quadrilateral and applies
 * minimum size thresholds in pixels. Thresholds are interpreted as:
 * - if `min_w <= 0` then width constraint is disabled,
 * - if `min_h <= 0` then height constraint is disabled.
 *
 * @param d Detection containing 4 corner points.
 * @param min_w Minimum allowed width in pixels (disabled if <= 0).
 * @param min_h Minimum allowed height in pixels (disabled if <= 0).
 * @return True if constraints pass or are disabled, otherwise false.
 */
static inline bool passes_min_size_(const algo::Detection& d, int min_w, int min_h) noexcept {
    if (min_w <= 0 && min_h <= 0) return true;

    float minx = d.pts[0].x, miny = d.pts[0].y, maxx = d.pts[0].x, maxy = d.pts[0].y;
    for (int i = 1; i < 4; ++i) {
        minx = std::min(minx, d.pts[i].x);
        miny = std::min(miny, d.pts[i].y);
        maxx = std::max(maxx, d.pts[i].x);
        maxy = std::max(maxy, d.pts[i].y);
    }
    const float w = std::max(0.0f, maxx - minx);
    const float h = std::max(0.0f, maxy - miny);
    if (min_w > 0 && w < float(min_w)) return false;
    if (min_h > 0 && h < float(min_h)) return false;
    return true;
}

/**
 * @brief Converts internal detections into the public API quadrilateral list.
 *
 * @details
 * The public API exposes only geometry (quads). Scores remain internal to keep
 * the external surface minimal and stable.
 *
 * @param dets Vector of internal detection objects.
 * @return Public @ref idet::VecQuad where each element is a @ref idet::Quad.
 */
static inline VecQuad to_public_quads_(const std::vector<algo::Detection>& dets) {
    VecQuad out;
    out.reserve(dets.size());
    for (const auto& d : dets) {
        Quad q{};
        for (int i = 0; i < 4; ++i) {
            q[i].x = d.pts[i].x;
            q[i].y = d.pts[i].y;
        }
        out.push_back(q);
    }
    return out;
}

} // namespace

/// @brief Builds a minimal detector configuration for a given task and model path.
DetectorConfig DetectorConfig::setup(Task task, std::string model_path) {
    DetectorConfig c;
    c.model_path = std::move(model_path);
    if (task == Task::Text) {
        c.task = Task::Text;
        c.engine = EngineKind::DBNet;
        // default: prefer exact polygon IoU for text (quads can be rotated)
        c.infer.use_fast_iou = false;
    } else if (task == Task::Face) {
        c.task = Task::Face;
        c.engine = EngineKind::SCRFD;
        // default: faces are rectangles in this pipeline -> fast AABB IoU is usually enough
        c.infer.use_fast_iou = true;
    } else {
        c.task = Task::None;
        c.engine = EngineKind::None;
        c.infer.use_fast_iou = false;
    }
    return c;
}

/// @brief Validates configuration invariants and engine-specific parameter constraints.
Status DetectorConfig::validate() const noexcept {
    if (task == Task::None) return Status::Invalid("DetectorConfig: task==None");
    if (engine == EngineKind::None) return Status::Invalid("DetectorConfig: engine==None");

    const Task et = engine_task(engine);
    if (et == Task::None) return Status::Unsupported("DetectorConfig: unknown engine");
    if (et != task) return Status::Invalid("DetectorConfig: engine/task mismatch");

    if (infer.tiles_dim.rows <= 0 || infer.tiles_dim.cols <= 0)
        return Status::Invalid("DetectorConfig: tiles_dim must be > 0");
    if (!(infer.tile_overlap >= 0.0f && infer.tile_overlap < 1.0f))
        return Status::Invalid("DetectorConfig: tile_overlap must be in [0,1)");

    if (infer.min_roi_size_w < 0 || infer.min_roi_size_h < 0)
        return Status::Invalid("DetectorConfig: min_roi_size must be >= 0");

    if (infer.bind_io && (infer.fixed_input_dim.rows <= 0 || infer.fixed_input_dim.cols <= 0))
        return Status::Invalid("DetectorConfig: bind_io requires fixed_input_dim (HxW) with values > 0");

    if (engine == EngineKind::DBNet) {
        if (!(infer.bin_thresh > 0.f && infer.bin_thresh < 1.f))
            return Status::Invalid("DBNet: bin_thresh must be in (0,1)");
        if (!(infer.box_thresh > 0.f && infer.box_thresh < 1.f))
            return Status::Invalid("DBNet: box_thresh must be in (0,1)");
        if (!(infer.unclip > 0.f)) return Status::Invalid("DBNet: unclip must be > 0");
    } else if (engine == EngineKind::SCRFD) {
        if (!(infer.box_thresh > 0.f && infer.box_thresh < 1.f))
            return Status::Invalid("SCRFD: box_thresh must be in (0,1)");
    }

    return Status::Ok();
}

namespace detail {

/**
 * @brief Private detector implementation owning the engine and executing the pipeline.
 *
 * @details
 * Responsibilities:
 * - validate and initialize the engine (eagerly in create(), lazily if needed)
 * - convert input @ref idet::Image to a BGR `cv::Mat` representation
 * - dispatch to single-pass or tiled inference
 * - apply common postprocessing:
 *   - minimum ROI size filtering
 *   - polygon NMS or score sorting
 *
 * @note
 * This class is not part of the public ABI. It is accessed only via an internal vtable.
 */
class DetectorImpl final {
  public:
    /// @brief Constructs the implementation with an initial configuration snapshot.
    explicit DetectorImpl(DetectorConfig cfg) : cfg_(std::move(cfg)) {}

    /// @brief Returns the configured task.
    Task task() const noexcept {
        return cfg_.task;
    }

    /// @brief Returns the configured engine kind.
    EngineKind engine() const noexcept {
        return cfg_.engine;
    }

    /**
     * @brief Validates config and creates the underlying engine instance.
     *
     * @return @ref idet::Status::Ok() on success, otherwise a non-OK status.
     *
     * @note Engine creation is delegated to @ref idet::engine::create_engine.
     */
    Status init_engine() noexcept {
        const Status s = cfg_.validate();
        if (!s.ok()) return s;

        auto r = engine::create_engine(cfg_);
        if (!r.ok()) return r.status();

        engine_ = std::move(r.value());
        if (!engine_) return Status::Internal("DetectorImpl: create_engine returned null");

        return Status::Ok();
    }

    /**
     * @brief Applies a "hot" configuration update without recreating the detector.
     *
     * @details
     * Immutable for a detector instance:
     * - task, engine kind, model path
     * - runtime policy (threading/affinity/session options)
     *
     * Mutable:
     * - inference options
     * - verbosity
     *
     * @param cfg New configuration.
     * @return @ref idet::Status::Ok() on success, otherwise a non-OK status.
     */
    Status update_config(const DetectorConfig& cfg) noexcept {
        if (cfg.task != cfg_.task) return Status::Invalid("update_config: task cannot change");
        if (cfg.engine != cfg_.engine) return Status::Invalid("update_config: engine cannot change");
        if (cfg.model_path != cfg_.model_path) return Status::Invalid("update_config: model_path cannot change");

        const auto& a = cfg_.runtime;
        const auto& b = cfg.runtime;
        if (b.ort_intra_threads != a.ort_intra_threads || b.ort_inter_threads != a.ort_inter_threads ||
            b.tile_omp_threads != a.tile_omp_threads || b.soft_mem_bind != a.soft_mem_bind ||
            b.suppress_opencv != a.suppress_opencv) {
            return Status::Invalid("update_config: runtime cannot change (recreate detector)");
        }

        cfg_.infer = cfg.infer;
        cfg_.verbose = cfg.verbose;

        if (!engine_) return Status::Invalid("update_config: engine not initialized");
        return engine_->update_hot(cfg_);
    }

    /**
     * @brief Prepares bound I/O resources for a fixed input resolution and number of contexts.
     *
     * @param w Input width in pixels.
     * @param h Input height in pixels.
     * @param contexts Number of independent binding contexts (normalized to >= 1).
     * @return @ref idet::Status::Ok() on success, otherwise a non-OK status.
     */
    Status prepare_binding(int w, int h, int contexts) noexcept {
        if (!engine_) return Status::Invalid("prepare_binding: engine not initialized");
        if (w <= 0 || h <= 0) return Status::Invalid("prepare_binding: non-positive w/h");
        if (contexts <= 0) contexts = 1;

        const Status s = engine_->setup_binding(w, h, contexts);
        if (s.ok()) binding_ready_ = true;
        return s;
    }

    /// @brief Public entry point for unbound (or internally managed) inference.
    Result<VecQuad> detect(const Image& img) noexcept {
        return run_(img, /*force_bound=*/false, /*ctx=*/0, /*explicit_bound_call=*/false);
    }

    /**
     * @brief Public entry point for bound inference using an explicit context index.
     *
     * @param img Input image.
     * @param ctx Context index (must be >= 0).
     * @return Result of detections or an error status.
     */
    Result<VecQuad> detect_bound(const Image& img, int ctx) noexcept {
        if (ctx < 0) return Result<VecQuad>::Err(Status::Invalid("detect_bound: ctx < 0"));
        return run_(img, /*force_bound=*/true, ctx, /*explicit_bound_call=*/true);
    }

  private:
    /**
     * @brief Executes the end-to-end pipeline and returns public quadrilateral results.
     *
     * @details
     * Steps:
     *  1) Ensure engine is initialized.
     *  2) Convert input image to BGR `cv::Mat`.
     *  3) Decide single vs tiled execution.
     *  4) Enforce binding requirements for bound inference.
     *  5) Run inference (single or tiled).
     *  6) Apply common postprocessing:
     *     - min-size filtering
     *     - NMS (or score sort if NMS disabled)
     */
    Result<VecQuad> run_(const Image& img, bool force_bound, int ctx, bool explicit_bound_call) noexcept {
        if (!engine_) {
            const Status s = init_engine();
            if (!s.ok()) return Result<VecQuad>::Err(s);
        }

        // Convert public Image into a BGR cv::Mat view (implementation defined).
        auto bm_res = internal::BgrMat::from(Image(img));
        if (!bm_res.ok()) return Result<VecQuad>::Err(bm_res.status());
        const cv::Mat& bgr = std::move(bm_res.value().mat());

        const bool tiled = (cfg_.infer.tiles_dim.rows * cfg_.infer.tiles_dim.cols) > 1;
        const bool want_bound = force_bound || (cfg_.infer.bind_io && binding_ready_);

        if (want_bound && !binding_ready_) {
            return Result<VecQuad>::Err(Status::Invalid(explicit_bound_call
                                                            ? "detect_bound: binding not prepared"
                                                            : "detect: bind_io enabled but binding not prepared"));
        }

        Result<std::vector<algo::Detection>> r =
            tiled ? run_tiled_(bgr, want_bound, ctx, explicit_bound_call) : run_single_(bgr, want_bound, ctx);

        if (!r.ok()) return Result<VecQuad>::Err(r.status());

        auto dets = std::move(r.value());

        // Common min-size filter.
        if (cfg_.infer.min_roi_size_w > 0 || cfg_.infer.min_roi_size_h > 0) {
            std::vector<algo::Detection> filtered;
            filtered.reserve(dets.size());
            for (auto& d : dets) {
                if (passes_min_size_(d, cfg_.infer.min_roi_size_w, cfg_.infer.min_roi_size_h))
                    filtered.push_back(std::move(d));
            }
            dets.swap(filtered);
        }

        // Common NMS (disabled when threshold <= 0).
        if (cfg_.infer.nms_iou > 0.0f && dets.size() > 1) {
            dets = algo::nms_poly(dets, cfg_.infer.nms_iou, cfg_.infer.use_fast_iou);
        }

        return Result<VecQuad>::Ok(to_public_quads_(dets));
    }

    /// @brief Runs inference on a single image (no tiling).
    Result<std::vector<algo::Detection>> run_single_(const cv::Mat& bgr, bool bound, int ctx) noexcept {
        return bound ? engine_->infer_bound(bgr, ctx) : engine_->infer_unbound(bgr);
    }

    /**
     * @brief Runs tiled inference and merges detections.
     *
     * @note
     * If the user explicitly called `detect_bound(ctx)`, bound tiling must not
     * parallelize across other contexts to preserve the "single explicit ctx" contract.
     */
    Result<std::vector<algo::Detection>> run_tiled_(const cv::Mat& bgr, bool bound, int ctx,
                                                    bool explicit_bound_call) noexcept {
        const bool parallel_bound = bound ? (!explicit_bound_call) : false;

        return algo::infer_tiled(*engine_, bgr, bound, ctx, parallel_bound, cfg_.infer.tiles_dim,
                                 cfg_.infer.tile_overlap, cfg_.runtime.tile_omp_threads);
    }

  private:
    /** @brief Snapshot of configuration used by this detector instance. */
    DetectorConfig cfg_;

    /** @brief Owned engine backend implementation (DBNet, SCRFD, ...). */
    std::unique_ptr<idet::engine::IEngine> engine_;

    /** @brief Whether bound I/O has been prepared successfully. */
    bool binding_ready_ = false;
};

} // namespace detail

namespace detail {

/**
 * @brief Internal vtable describing operations on the opaque detector implementation pointer.
 *
 * @details
 * This vtable provides a stable call surface from the public @ref idet::Detector
 * into the hidden `detail::DetectorImpl` without exposing implementation types
 * in the public header.
 *
 * All function pointers are @c noexcept and must translate exceptions into
 * @ref idet::Status / @ref idet::Result errors.
 */
struct DetectorVTable {
    void (*destroy)(void*) noexcept;
    Status (*update)(void*, const DetectorConfig&) noexcept;
    Status (*prepare_binding)(void*, int, int, int) noexcept;
    Result<VecQuad> (*detect)(void*, const Image&) noexcept;
    Result<VecQuad> (*detect_bound)(void*, const Image&, int) noexcept;

    Task (*task)(const void*) noexcept;
    EngineKind (*engine)(const void*) noexcept;
};

/// @brief The concrete vtable instance used by all detectors.
static const DetectorVTable kVt{
    // destroy
    [](void* p) noexcept { delete static_cast<detail::DetectorImpl*>(p); },

    // update
    [](void* p, const DetectorConfig& cfg) noexcept -> Status {
        try {
            return static_cast<detail::DetectorImpl*>(p)->update_config(cfg);
        } catch (const std::exception& e) {
            return Status::Internal(std::string("update_config threw: ") + e.what());
        } catch (...) {
            return Status::Internal("update_config threw (unknown)");
        }
    },

    // prepare_binding
    [](void* p, int w, int h, int c) noexcept -> Status {
        try {
            return static_cast<detail::DetectorImpl*>(p)->prepare_binding(w, h, c);
        } catch (const std::exception& e) {
            return Status::Internal(std::string("prepare_binding threw: ") + e.what());
        } catch (...) {
            return Status::Internal("prepare_binding threw (unknown)");
        }
    },

    // detect
    [](void* p, const Image& img) noexcept -> Result<VecQuad> {
        try {
            return static_cast<detail::DetectorImpl*>(p)->detect(img);
        } catch (const std::exception& e) {
            return Result<VecQuad>::Err(Status::Internal(std::string("detect threw: ") + e.what()));
        } catch (...) {
            return Result<VecQuad>::Err(Status::Internal("detect threw (unknown)"));
        }
    },

    // detect_bound
    [](void* p, const Image& img, int ctx) noexcept -> Result<VecQuad> {
        try {
            return static_cast<detail::DetectorImpl*>(p)->detect_bound(img, ctx);
        } catch (const std::exception& e) {
            return Result<VecQuad>::Err(Status::Internal(std::string("detect_bound threw: ") + e.what()));
        } catch (...) {
            return Result<VecQuad>::Err(Status::Internal("detect_bound threw (unknown)"));
        }
    },

    // task
    [](const void* p) noexcept -> Task { return static_cast<const detail::DetectorImpl*>(p)->task(); },

    // engine
    [](const void* p) noexcept -> EngineKind { return static_cast<const detail::DetectorImpl*>(p)->engine(); },
};

} // namespace detail

/// @brief Destructor releases implementation resources via @ref reset.
Detector::~Detector() noexcept {
    reset();
}

/// @brief Move constructor transfers ownership of the implementation pointer and vtable.
Detector::Detector(Detector&& other) noexcept : impl_(other.impl_), vtbl_(other.vtbl_) {
    other.impl_ = nullptr;
    other.vtbl_ = nullptr;
}

/// @brief Move assignment releases current resources, then takes ownership from @p other.
Detector& Detector::operator=(Detector&& other) noexcept {
    if (this != &other) {
        reset();
        impl_ = other.impl_;
        vtbl_ = other.vtbl_;
        other.impl_ = nullptr;
        other.vtbl_ = nullptr;
    }
    return *this;
}

/// @brief Returns true if this detector holds a valid implementation and vtable.
Detector::operator bool() const noexcept {
    return impl_ != nullptr && vtbl_ != nullptr;
}

/// @brief Returns the configured task if valid, otherwise @ref idet::Task::None.
Task Detector::task() const noexcept {
    return (vtbl_ && impl_) ? vtbl_->task(impl_) : Task::None;
}

/// @brief Returns the configured engine kind if valid, otherwise @ref idet::EngineKind::None.
EngineKind Detector::engine() const noexcept {
    return (vtbl_ && impl_) ? vtbl_->engine(impl_) : EngineKind::None;
}

/// @brief Destroys the implementation object and clears internal pointers.
void Detector::reset() noexcept {
    if (impl_ && vtbl_ && vtbl_->destroy) vtbl_->destroy(impl_);
    impl_ = nullptr;
    vtbl_ = nullptr;
}

/**
 * @brief Creates a detector instance (allocates implementation and initializes the engine).
 *
 * @details
 * This factory validates the config, constructs the hidden implementation object,
 * initializes the engine backend, and returns a fully usable @ref idet::Detector.
 *
 * @param cfg Detector configuration.
 * @return Result::Ok(detector) on success, or Result::Err(status) on failure.
 */
Result<Detector> Detector::create(const DetectorConfig& cfg) noexcept {
    const Status vs = cfg.validate();
    if (!vs.ok()) return Result<Detector>::Err(vs);

    Detector d;
    detail::DetectorImpl* p = nullptr;

    try {
        p = new (std::nothrow) detail::DetectorImpl(cfg);
        if (!p) return Result<Detector>::Err(Status::OutOfMemory("Detector::create: alloc failed"));

        const Status is = p->init_engine();
        if (!is.ok()) {
            delete p;
            return Result<Detector>::Err(is);
        }
    } catch (const std::exception& e) {
        delete p;
        return Result<Detector>::Err(Status::Invalid(std::string("Detector::create: ctor failed: ") + e.what()));
    } catch (...) {
        delete p;
        return Result<Detector>::Err(Status::Internal("Detector::create: ctor failed (unknown)"));
    }

    d.impl_ = p;
    d.vtbl_ = &detail::kVt;
    return Result<Detector>::Ok(std::move(d));
}

/// @brief Updates configuration via the internal vtable boundary.
Status Detector::update_config(const DetectorConfig& cfg) noexcept {
    if (!impl_ || !vtbl_) return Status::Invalid("Detector::update_config: invalid detector");
    return vtbl_->update(impl_, cfg);
}

/// @brief Prepares binding resources via the internal vtable boundary.
Status Detector::prepare_binding(int width, int height, int contexts) noexcept {
    if (!impl_ || !vtbl_) return Status::Invalid("Detector::prepare_binding: invalid detector");
    if (width <= 0 || height <= 0) return Status::Invalid("Detector::prepare_binding: non-positive w/h");
    if (contexts <= 0) contexts = 1;
    return vtbl_->prepare_binding(impl_, width, height, contexts);
}

/// @brief Runs detection via the internal vtable boundary.
Result<VecQuad> Detector::detect(const Image& image) noexcept {
    if (!impl_ || !vtbl_) return Result<VecQuad>::Err(Status::Invalid("Detector::detect: invalid detector"));
    return vtbl_->detect(impl_, image);
}

/// @brief Runs bound detection via the internal vtable boundary using a context index.
Result<VecQuad> Detector::detect_bound(const Image& image, int ctx_idx) noexcept {
    if (!impl_ || !vtbl_) return Result<VecQuad>::Err(Status::Invalid("Detector::detect_bound: invalid detector"));
    return vtbl_->detect_bound(impl_, image, ctx_idx);
}

/**
 * @brief Applies the requested runtime policy (thread/CPU/memory binding).
 *
 * @details
 * This is a thin public wrapper that delegates to the platform-specific implementation.
 * On non-supported platforms, the implementation may return @ref idet::Status::Ok()
 * without applying any binding.
 *
 * @param policy Runtime policy (CPU set, NUMA node set, binding knobs).
 * @param verbose If true, prints diagnostic details (best-effort).
 * @return Status::Ok() if applied (or not supported but safely ignored), otherwise an error status.
 */
IDET_API Status setup_runtime_policy(const RuntimePolicy& policy, bool verbose) noexcept {
    return platform::setup_runtime_policy_impl(policy, verbose);
}

} // namespace idet
