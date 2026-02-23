/**
 * @file engine.cpp
 * @ingroup idet_engine
 * @brief Base ORT engine implementation: hot-update checks and session creation helpers.
 *
 * @details
 * Implements @ref idet::engine::IEngine helpers:
 * - immutable/hot-update validation contract (@ref idet::engine::IEngine::check_hot_update_),
 * - common hot-update field application (@ref idet::engine::IEngine::apply_hot_common_),
 * - ORT session creation from filesystem path or embedded model blob
 *   (@ref idet::engine::IEngine::create_session_),
 * - process-wide ORT environment singleton wiring (@ref idet::engine::IEngine::global_env_).
 *
 * Notes:
 * - ORT session options are configured from @ref idet::DetectorConfig::runtime.
 * - Embedded model selection uses @ref idet::internal::get_model_blob (when enabled).
 * - Affinity verification may be performed after session creation (best-effort diagnostics).
 */

#include "engine/engine.h"

#include "internal/embed_model.h"
#include "platform/cross_topology.h"

#include <exception>
#include <new>
#include <string>

namespace idet::engine {

/**
 * @brief Base engine constructor.
 *
 * @details
 * Stores a configuration snapshot and initializes the ONNX Runtime environment reference.
 * The environment is provided by @ref IEngine::global_env_ and is shared process-wide.
 *
 * @param cfg Engine configuration snapshot.
 * @param log_id Optional log identifier string for ORT environment initialization.
 *
 * @note
 * The log id affects only the first call that constructs the global environment singleton.
 * Subsequent engine instances will reuse the same environment.
 */
IEngine::IEngine(const DetectorConfig& cfg, const char* log_id) : cfg_(cfg), env_(global_env_(log_id)) {}

/**
 * @brief Validate whether the proposed configuration can be applied as a hot update.
 *
 * @details
 * A hot update is intended to modify only parameters that do NOT require recreating
 * the ONNX Runtime session or reinitializing runtime policies.
 *
 * Immutable fields (must NOT change):
 * - @ref idet::DetectorConfig::task
 * - @ref idet::DetectorConfig::engine
 * - @ref idet::DetectorConfig::model_path
 * - @ref idet::DetectorConfig::runtime (all fields)
 *
 * Mutable fields (allowed in hot update, applied by @ref apply_hot_common_ and derived engines):
 * - @ref idet::DetectorConfig::infer (thresholds, tiling parameters, etc.)
 * - @ref idet::DetectorConfig::verbose
 *
 * @param next Proposed next configuration.
 * @return Status::Ok() if hot update is allowed; otherwise an error status with a reason.
 *
 * @note
 * The specific semantics of mutable fields are engine-defined. For example, a concrete
 * engine may reject some inference parameters if they imply a shape/binding change.
 */
Status IEngine::check_hot_update_(const DetectorConfig& next) const noexcept {
    if (next.task != cfg_.task) return Status::Invalid("update_hot: task cannot change");
    if (next.engine != cfg_.engine) return Status::Invalid("update_hot: engine cannot change");
    if (next.model_path != cfg_.model_path) return Status::Invalid("update_hot: model_path cannot change");

    // Runtime policy is treated as immutable here because it typically affects ORT threadpools,
    // affinity/memory policy, and other process-wide knobs that are unsafe to change without
    // recreating the session (and often without re-applying placement policy).
    const auto& a = cfg_.runtime;
    const auto& b = next.runtime;

    if (b.ort_intra_threads != a.ort_intra_threads || b.ort_inter_threads != a.ort_inter_threads ||
        b.tile_omp_threads != a.tile_omp_threads || b.soft_mem_bind != a.soft_mem_bind ||
        b.numa_mem_policy != a.numa_mem_policy || b.suppress_opencv != a.suppress_opencv) {
        return Status::Invalid("update_hot: runtime cannot change (recreate detector)");
    }

    return Status::Ok();
}

/**
 * @brief Apply common parts of a hot configuration update to the stored config.
 *
 * @details
 * Updates the configuration fields that are shared between engines and are considered
 * safe to change without recreating the ORT session.
 *
 * Concrete engines typically call:
 *  1) @ref check_hot_update_ (validate)
 *  2) @ref apply_hot_common_ (apply shared fields)
 *  3) Update their own cached parameters, thresholds, precomputed constants, etc.
 *
 * @param next Proposed next configuration.
 */
void IEngine::apply_hot_common_(const DetectorConfig& next) noexcept {
    cfg_.infer = next.infer;
    cfg_.verbose = next.verbose;
}

/**
 * @brief Create and configure the ONNX Runtime session for the engine.
 *
 * @details
 * This method configures basic ORT session options and loads the model either:
 * - from a filesystem path (@p model_path), or
 * - from an embedded binary blob selected by @p engine_kind (see internal/embed_model.h).
 *
 * Current session options:
 * - Graph optimization: ORT_ENABLE_ALL
 * - Execution mode: ORT_SEQUENTIAL
 * - CPU memory arena: enabled (recommended for performance in most CPU workloads)
 * - Memory pattern: enabled (helps when shapes are stable; ORT uses it opportunistically)
 * - Intra-op threads: cfg_.runtime.ort_intra_threads (if > 0)
 * - Inter-op threads: cfg_.runtime.ort_inter_threads (if > 0)
 *
 * Error handling:
 * - Converts exceptions into @ref idet::Status to avoid exceptions crossing API boundaries.
 *
 * @param model_path Path to ONNX model on disk. If empty, embedded model is used.
 * @param engine_kind Engine kind used to select embedded model blob when @p model_path is empty.
 * @return Status::Ok() on success; otherwise an error status describing the failure.
 *
 * @note
 * This method sets @ref IEngine::session_ to a valid session on success.
 *
 * @warning
 * If @p model_path is empty and the binary was not built with the required embedded model,
 * the function returns an error.
 */
Status IEngine::create_session_(const std::string& model_path, EngineKind engine_kind) noexcept {
    try {
        so_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

        // Arena + mem pattern are typical CPU-performance defaults.
        // ORT may ignore mem pattern in cases where it is not applicable.
        so_.EnableCpuMemArena();
        so_.EnableMemPattern();

        // Session log severity (0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL).
        so_.SetLogSeverityLevel(3);

        if (cfg_.runtime.ort_intra_threads > 0) so_.SetIntraOpNumThreads(cfg_.runtime.ort_intra_threads);
        if (cfg_.runtime.ort_inter_threads > 0) so_.SetInterOpNumThreads(cfg_.runtime.ort_inter_threads);

        if (!model_path.empty()) {
            session_ = Ort::Session(env_, model_path.c_str(), so_);
        } else {
            const auto blob = idet::internal::get_model_blob(engine_kind);
            if (blob.empty()) {
                return Status::Invalid("create_session: empty model path and no embedded model provided");
            }
            session_ = Ort::Session(env_, blob.data, blob.size, so_);
        }

        // Best-effort diagnostic: confirm current threads are within the expected affinity mask.
        // This is not a functional requirement for ORT, but helps catch misordered policy setup.
        const auto vr_aff = idet::platform::verify_all_threads_affinity_subset(cfg_.verbose);
        if (!vr_aff.ok()) return vr_aff;

        return Status::Ok();

    } catch (const std::bad_alloc&) {
        return Status::OutOfMemory("create_session: bad_alloc");
    } catch (const Ort::Exception& e) {
        return Status::Invalid(std::string("create_session: ORT exception: ") + e.what());
    } catch (const std::exception& e) {
        return Status::Invalid(std::string("create_session: ") + e.what());
    } catch (...) {
        return Status::Internal("create_session: unknown");
    }
}

} // namespace idet::engine
