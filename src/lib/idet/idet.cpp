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
#include "internal/bgr_image.h"
#include "platform/runtime_policy_setup.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <thread>
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
    for (std::size_t i = 1; i < d.pts.size(); ++i) {
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
        out.emplace_back();
        Quad& q = out.back();
        for (std::size_t i = 0; i < q.size(); ++i) {
            q[i].x = d.pts[i].x;
            q[i].y = d.pts[i].y;
        }
    }
    return out;
}

static inline bool fixed_dim_unset_(const GridSpec& g) noexcept {
    return g.rows == 0 && g.cols == 0;
}

static inline bool fixed_dim_set_(const GridSpec& g) noexcept {
    return g.rows > 0 && g.cols > 0;
}

static inline bool same_grid_(const GridSpec& a, const GridSpec& b) noexcept {
    return a.rows == b.rows && a.cols == b.cols;
}

static inline int tile_count_(const GridSpec& g) noexcept {
    if (g.rows <= 0 || g.cols <= 0) return 0;
    constexpr int kMaxInt = std::numeric_limits<int>::max();
    if (g.rows > kMaxInt / g.cols) return kMaxInt;
    return g.rows * g.cols;
}

static inline int auto_binding_contexts_(const DetectorConfig& cfg) noexcept {
    const int tiles = tile_count_(cfg.infer.tiles_dim);
    if (tiles <= 1) return 1;
    return std::max(1, std::min(cfg.runtime.tile_omp_threads, tiles));
}

} // namespace

/// @brief Builds a minimal detector configuration for a given task and model path.
///
/// The default engine for each task is resolved through the engine registry, so adding
/// a new domain only requires registering an entry in @ref engine_factory.cpp. Per-task
/// defaults that are not part of the generic config (e.g. AABB-vs-polygon IoU) are
/// still set here, since they are application-level rather than engine-level.
DetectorConfig DetectorConfig::setup(Task task, std::string model_path) {
    DetectorConfig c;
    c.model_path = std::move(model_path);

    c.task = task;
    c.engine = engine::engine_default_for_task(task);

    switch (task) {
    case Task::Text:
        // Text quads can be rotated -> exact polygon IoU is more accurate.
        c.infer.use_fast_iou = false;
        break;
    case Task::Face:
        // Faces are axis-aligned rectangles in this pipeline -> fast AABB IoU suffices.
        c.infer.use_fast_iou = true;
        break;
    case Task::Cloth:
        // YOLO outputs are axis-aligned rectangles; AABB IoU matches NMS conventions used
        // by the YOLO family.
        c.infer.use_fast_iou = true;
        // YOLO conf threshold conventions are slightly lower than DBNet/SCRFD defaults.
        c.infer.box_thresh = 0.25f;
        break;
    default:
        c.task = Task::None;
        c.engine = EngineKind::None;
        c.infer.use_fast_iou = false;
        break;
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
    if (tile_count_(infer.tiles_dim) <= 0 || tile_count_(infer.tiles_dim) > 4096)
        return Status::Invalid("DetectorConfig: tiles_dim product must be in [1,4096]");
    if (!(infer.tile_overlap >= 0.0f && infer.tile_overlap < 1.0f))
        return Status::Invalid("DetectorConfig: tile_overlap must be in [0,1)");
    if (!(infer.nms_iou >= 0.0f && infer.nms_iou <= 1.0f))
        return Status::Invalid("DetectorConfig: nms_iou must be in [0,1]");
    if (infer.max_img_size <= 0) return Status::Invalid("DetectorConfig: max_img_size must be > 0");

    if (infer.min_roi_size_w < 0 || infer.min_roi_size_h < 0)
        return Status::Invalid("DetectorConfig: min_roi_size must be >= 0");
    if (!(fixed_dim_unset_(infer.fixed_input_dim) || fixed_dim_set_(infer.fixed_input_dim)))
        return Status::Invalid("DetectorConfig: fixed_input_dim must be unset or positive HxW");
    if (infer.bind_io && !fixed_dim_set_(infer.fixed_input_dim))
        return Status::Invalid("DetectorConfig: bind_io requires fixed_input_dim");

    if (runtime.ort_intra_threads <= 0 || runtime.ort_inter_threads <= 0 || runtime.tile_omp_threads <= 0) {
        return Status::Invalid("DetectorConfig: runtime thread counts must be > 0");
    }

    // Engine-specific validation lives in the engine registry to keep this file
    // independent of any particular engine family. See engine_factory.cpp.
    return engine::engine_validate_specific(*this);
}

namespace detail {

/**
 * @brief Private detector implementation owning the engine and executing the pipeline.
 *
 * @details
 * Responsibilities:
 * - validate and initialize the engine (eagerly in create(), lazily if needed)
 * - convert input @ref idet::Image to an internal BGR image representation
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
     * @return @c idet::Status::Ok() on success, otherwise a non-OK status.
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

        return ensure_config_binding_();
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
     * @return @c idet::Status::Ok() on success, otherwise a non-OK status.
     */
    Status update_config(const DetectorConfig& cfg) noexcept {
        const Status vs = cfg.validate();
        if (!vs.ok()) return vs;

        if (cfg.task != cfg_.task) return Status::Invalid("update_config: task cannot change");
        if (cfg.engine != cfg_.engine) return Status::Invalid("update_config: engine cannot change");
        if (cfg.model_path != cfg_.model_path) return Status::Invalid("update_config: model_path cannot change");

        // Runtime is treated as immutable (must match engine.cpp::check_hot_update_).
        // Any change here would require recreating the detector to reapply ORT threadpool /
        // affinity / NUMA policy.
        const auto& a = cfg_.runtime;
        const auto& b = cfg.runtime;
        if (b.ort_intra_threads != a.ort_intra_threads || b.ort_inter_threads != a.ort_inter_threads ||
            b.tile_omp_threads != a.tile_omp_threads || b.soft_mem_bind != a.soft_mem_bind ||
            b.numa_mem_policy != a.numa_mem_policy || b.suppress_opencv != a.suppress_opencv) {
            return Status::Invalid("update_config: runtime cannot change (recreate detector)");
        }

        if (!engine_) return Status::Invalid("update_config: engine not initialized");

        const bool fixed_changed = !same_grid_(cfg.infer.fixed_input_dim, cfg_.infer.fixed_input_dim);

        const Status us = engine_->update_hot(cfg);
        if (!us.ok()) return us;

        cfg_.infer = cfg.infer;
        cfg_.verbose = cfg.verbose;

        if (cfg_.infer.bind_io && fixed_changed) {
            engine_->unset_binding();
        }
        return ensure_config_binding_();
    }

    /**
     * @brief Prepares bound I/O resources for a fixed input resolution and number of contexts.
     *
     * @param w Input width in pixels.
     * @param h Input height in pixels.
     * @param contexts Number of independent binding contexts (normalized to >= 1).
     * @return @c idet::Status::Ok() on success, otherwise a non-OK status.
     */
    Status prepare_binding(int w, int h, int contexts) noexcept {
        if (!engine_) return Status::Invalid("prepare_binding: engine not initialized");
        if (w <= 0 || h <= 0) return Status::Invalid("prepare_binding: non-positive w/h");
        if (contexts <= 0) contexts = 1;

        return engine_->setup_binding(w, h, contexts);
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
     *  2) Convert input image to an internal BGR view/holder.
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

        // Convert public Image into an OpenCV-free BGR view/holder.
        auto bm_res = internal::BgrImage::from(Image(img));
        if (!bm_res.ok()) return Result<VecQuad>::Err(bm_res.status());
        const internal::BgrImageView& bgr = bm_res.value().view();

        const bool want_bound = force_bound || cfg_.infer.bind_io;

        if (want_bound && !binding_ready()) {
            return Result<VecQuad>::Err(Status::Invalid(explicit_bound_call
                                                            ? "detect_bound: binding not prepared"
                                                            : "detect: bind_io enabled but binding not prepared"));
        }

        const bool tiled = tile_count_(cfg_.infer.tiles_dim) > 1;

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
    Result<std::vector<algo::Detection>> run_single_(const internal::BgrImageView& bgr, bool bound, int ctx) noexcept {
        return bound ? engine_->infer_bound(bgr, ctx) : engine_->infer_unbound(bgr);
    }

    /**
     * @brief Runs tiled inference and merges detections.
     *
     * @note
     * If the user explicitly called `detect_bound(ctx)`, bound tiling must not
     * parallelize across other contexts to preserve the "single explicit ctx" contract.
     */
    Result<std::vector<algo::Detection>> run_tiled_(const internal::BgrImageView& bgr, bool bound, int ctx,
                                                    bool explicit_bound_call) noexcept {
        const bool parallel_bound = bound ? (!explicit_bound_call) : false;

        return algo::infer_tiled(*engine_, bgr, bound, ctx, parallel_bound, cfg_.infer.tiles_dim,
                                 cfg_.infer.tile_overlap, cfg_.runtime.tile_omp_threads);
    }

    bool binding_ready() const noexcept {
        return engine_ != nullptr && engine_->binding_ready();
    }

    Status ensure_config_binding_() noexcept {
        if (!cfg_.infer.bind_io) return Status::Ok();

        const GridSpec& fixed = cfg_.infer.fixed_input_dim;
        if (!fixed_dim_set_(fixed)) {
            return Status::Invalid("DetectorImpl: bind_io requires fixed_input_dim");
        }

        const int desired_contexts = auto_binding_contexts_(cfg_);
        if (binding_ready() && engine_->bound_w() == fixed.cols && engine_->bound_h() == fixed.rows &&
            engine_->bound_contexts() >= desired_contexts) {
            return Status::Ok();
        }
        return prepare_binding(fixed.cols, fixed.rows, desired_contexts);
    }

  private:
    /** @brief Snapshot of configuration used by this detector instance. */
    DetectorConfig cfg_;

    /** @brief Owned engine backend implementation (DBNet, SCRFD, ...). */
    std::unique_ptr<idet::engine::IEngine> engine_;
};

/**
 * @brief Background single-flight detector runner.
 *
 * @details
 * The worker owns the detector and serializes all detection calls through one background
 * thread. This gives pipeline users an explicit, bounded handoff point without sharing a
 * detector instance between application threads.
 */
struct DetectorWorkerImpl final {
    Detector detector;
    DetectorWorkerOptions options;

    mutable std::mutex mu;
    std::condition_variable cv;
    std::thread thread;

    bool stop = false;
    bool has_job = false;
    Image job;
    std::atomic<DetectorWorkerState> worker_state{DetectorWorkerState::Idle};
    std::optional<Result<VecQuad>> completed;

    DetectorWorkerImpl(Detector&& det, DetectorWorkerOptions opt) : detector(std::move(det)), options(opt) {}

    ~DetectorWorkerImpl() noexcept {
        shutdown();
    }

    DetectorWorkerImpl(const DetectorWorkerImpl&) = delete;
    DetectorWorkerImpl& operator=(const DetectorWorkerImpl&) = delete;

    Status start() noexcept {
        try {
            thread = std::thread([this]() { this->run_loop(); });
            return Status::Ok();
        } catch (const std::bad_alloc&) {
            return Status::OutOfMemory("DetectorWorker: thread allocation failed");
        } catch (const std::exception& e) {
            return Status::Internal(std::string("DetectorWorker: start failed: ") + e.what());
        } catch (...) {
            return Status::Internal("DetectorWorker: start failed (unknown)");
        }
    }

    void shutdown() noexcept {
        {
            std::lock_guard<std::mutex> lock(mu);
            stop = true;
            cv.notify_all();
        }
        if (thread.joinable()) thread.join();
    }

    Status submit(const Image& image) noexcept {
        if (!image) return Status::Invalid("DetectorWorker::submit: invalid image");

        auto can_accept_locked = [&]() noexcept -> Status {
            if (stop) return Status::Invalid("DetectorWorker::submit: worker is stopping");
            const DetectorWorkerState st = worker_state.load(std::memory_order_acquire);
            if (st == DetectorWorkerState::Running) {
                return Status::Invalid("DetectorWorker::submit: worker is already running");
            }
            if (st == DetectorWorkerState::Ready || completed.has_value()) {
                return Status::Invalid("DetectorWorker::submit: previous result not consumed");
            }
            return Status::Ok();
        };

        {
            std::lock_guard<std::mutex> lock(mu);
            const Status st = can_accept_locked();
            if (!st.ok()) return st;
        }

        Image owned_or_view;
        if (options.copy_input) {
            const ImageView& v = image.view();
            auto cp = Image::copy_from(v.format, v.width, v.height, v.data, v.stride_bytes);
            if (!cp.ok()) return cp.status();
            owned_or_view = std::move(cp.value());
        } else {
            owned_or_view = image;
        }

        {
            std::lock_guard<std::mutex> lock(mu);
            const Status st = can_accept_locked();
            if (!st.ok()) return st;

            job = std::move(owned_or_view);
            has_job = true;
            worker_state.store(DetectorWorkerState::Running, std::memory_order_release);
        }
        cv.notify_one();
        return Status::Ok();
    }

    DetectorWorkerState state() const noexcept {
        return worker_state.load(std::memory_order_acquire);
    }

    Result<VecQuad> take_result() noexcept {
        std::lock_guard<std::mutex> lock(mu);
        if (worker_state.load(std::memory_order_acquire) == DetectorWorkerState::Running) {
            return Result<VecQuad>::Err(Status::Invalid("DetectorWorker::take_result: task still running"));
        }
        if (!completed.has_value()) {
            return Result<VecQuad>::Err(Status::NotFound("DetectorWorker::take_result: no completed result"));
        }

        Result<VecQuad> out = std::move(*completed);
        completed.reset();
        worker_state.store(DetectorWorkerState::Idle, std::memory_order_release);
        return out;
    }

  private:
    void run_loop() noexcept {
        for (;;) {
            Image current;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&]() { return stop || has_job; });
                if (stop && !has_job) return;

                current = std::move(job);
                job = Image{};
                has_job = false;
            }

            Result<VecQuad> r = options.use_bound ? detector.detect_bound(current, options.binding_context_index)
                                                  : detector.detect(current);

            {
                std::lock_guard<std::mutex> lock(mu);
                completed.emplace(std::move(r));
            }
            worker_state.store(DetectorWorkerState::Ready, std::memory_order_release);
        }
    }
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

/*
 * @brief Creates a detector instance (allocates implementation and initializes the engine).
 *
 * @details
 * This factory validates the config, constructs the hidden implementation object,
 * initializes the engine backend, and returns a fully usable @ref idet::Detector.
 *
 * @param cfg Detector configuration.
 * @return Result::Ok(detector) on success, or Result::Err(status) on failure.
 */
Result<Detector> Detector::create(const DetectorConfig& config) noexcept {
    const Status vs = config.validate();
    if (!vs.ok()) return Result<Detector>::Err(vs);

    Detector d;
    std::unique_ptr<detail::DetectorImpl> p;

    try {
        p.reset(new (std::nothrow) detail::DetectorImpl(config));
        if (!p) return Result<Detector>::Err(Status::OutOfMemory("Detector::create: alloc failed"));

        const Status is = p->init_engine();
        if (!is.ok()) return Result<Detector>::Err(is);
    } catch (const std::exception& e) {
        return Result<Detector>::Err(Status::Invalid(std::string("Detector::create: ctor failed: ") + e.what()));
    } catch (...) {
        return Result<Detector>::Err(Status::Internal("Detector::create: ctor failed (unknown)"));
    }

    d.impl_ = p.release();
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

/// @brief Stops the background worker and releases resources.
DetectorWorker::~DetectorWorker() noexcept {
    reset();
}

/// @brief Move constructor transfers the worker implementation pointer.
DetectorWorker::DetectorWorker(DetectorWorker&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

/// @brief Move assignment releases current resources and takes ownership from @p other.
DetectorWorker& DetectorWorker::operator=(DetectorWorker&& other) noexcept {
    if (this != &other) {
        reset();
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

/// @brief Returns whether this worker owns an implementation.
DetectorWorker::operator bool() const noexcept {
    return impl_ != nullptr;
}

/// @brief Creates a detector worker and starts its background thread.
Result<DetectorWorker> DetectorWorker::create(const DetectorConfig& config,
                                              const DetectorWorkerOptions& options) noexcept {
    if (options.binding_contexts <= 0) {
        return Result<DetectorWorker>::Err(Status::Invalid("DetectorWorker::create: binding_contexts must be > 0"));
    }
    if (options.binding_context_index < 0 || options.binding_context_index >= options.binding_contexts) {
        return Result<DetectorWorker>::Err(
            Status::Invalid("DetectorWorker::create: binding_context_index out of range"));
    }
    if (options.use_bound && (options.binding_width <= 0 || options.binding_height <= 0)) {
        return Result<DetectorWorker>::Err(
            Status::Invalid("DetectorWorker::create: bound mode requires positive binding_width/binding_height"));
    }

    auto det_res = Detector::create(config);
    if (!det_res.ok()) return Result<DetectorWorker>::Err(det_res.status());

    Detector det = std::move(det_res.value());
    if (options.use_bound) {
        Status bs = det.prepare_binding(options.binding_width, options.binding_height, options.binding_contexts);
        if (!bs.ok()) return Result<DetectorWorker>::Err(bs);
    }

    std::unique_ptr<detail::DetectorWorkerImpl> p;
    try {
        p.reset(new (std::nothrow) detail::DetectorWorkerImpl(std::move(det), options));
        if (!p) return Result<DetectorWorker>::Err(Status::OutOfMemory("DetectorWorker::create: alloc failed"));

        Status ss = p->start();
        if (!ss.ok()) return Result<DetectorWorker>::Err(ss);
    } catch (const std::exception& e) {
        return Result<DetectorWorker>::Err(Status::Internal(std::string("DetectorWorker::create: ") + e.what()));
    } catch (...) {
        return Result<DetectorWorker>::Err(Status::Internal("DetectorWorker::create: unknown exception"));
    }

    DetectorWorker w;
    w.impl_ = p.release();
    return Result<DetectorWorker>::Ok(std::move(w));
}

/// @brief Submits one image to the background worker.
Status DetectorWorker::submit(const Image& image) noexcept {
    if (!impl_) return Status::Invalid("DetectorWorker::submit: invalid worker");
    return impl_->submit(image);
}

/// @brief Returns the current worker state.
DetectorWorkerState DetectorWorker::state() const noexcept {
    return impl_ ? impl_->state() : DetectorWorkerState::Idle;
}

/// @brief Takes the completed result from the worker.
Result<VecQuad> DetectorWorker::take_result() noexcept {
    if (!impl_) return Result<VecQuad>::Err(Status::Invalid("DetectorWorker::take_result: invalid worker"));
    return impl_->take_result();
}

/// @brief Stops the background worker and clears the implementation pointer.
void DetectorWorker::reset() noexcept {
    delete impl_;
    impl_ = nullptr;
}

/*
 * @brief Applies the requested runtime policy (thread/CPU/memory binding).
 *
 * @details
 * This is a thin public wrapper that delegates to the platform-specific implementation.
 * On non-supported platforms, the implementation may return @c idet::Status::Ok()
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
