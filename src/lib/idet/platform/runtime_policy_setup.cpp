/**
 * @file runtime_policy_setup.cpp
 * @ingroup idet_platform
 * @brief Runtime policy application implementation (affinity + OpenMP/OpenCV coordination).
 *
 * @details
 * Implements @ref idet::platform::setup_runtime_policy_impl:
 * - computes a conservative desired concurrency from ORT intra/inter and tile OpenMP threads,
 * - applies process/thread affinity via @ref idet::platform::apply_process_placement_policy,
 * - optionally prints topology and runs affinity/NUMA diagnostics,
 * - configures OpenMP environment/runtime via @ref idet::platform::configure_openmp_affinity,
 * - optionally suppresses OpenCV internal threading (cv::setNumThreads(1)).
 *
 * @warning
 * Must run early (before ORT session creation / OpenMP initialization) for best effect.
 */

#include "platform/runtime_policy_setup.h"

#include "internal/opencv_headers.h" // IWYU pragma: keep
#include "platform/cross_topology.h"
#include "platform/omp_config.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <string>

namespace idet::platform {

namespace {

/**
 * @brief Clamps a user-provided thread count to a safe positive value.
 *
 * @details
 * Many runtime knobs accept an integer thread count. This helper normalizes
 * non-positive values to 1 to avoid undefined behavior and to keep the rest
 * of the policy logic deterministic.
 *
 * @param v Requested thread count (may be <= 0).
 * @return A positive thread count (>= 1).
 */
static inline std::size_t clamp_threads_(int v) noexcept {
    return (v > 0) ? static_cast<std::size_t>(v) : 1u;
}

} // namespace

/**
 * @brief Applies process-wide runtime settings for CPU binding, OpenMP, and OpenCV.
 *
 * @details
 * Implementation strategy:
 * 1) Derive an upper bound for "desired concurrency" from:
 *    - ORT intra-op and inter-op thread counts, and
 *    - OpenMP thread count used for tile-parallel execution.
 * 2) Apply deterministic CPU placement (and optional NUMA policy) as early as possible.
 * 3) Optionally print detected topology.
 * 4) Run best-effort diagnostics (affinity subset and page locality checks where supported).
 * 5) Configure OpenMP environment/runtime so OpenMP workers stay inside the bound CPU mask.
 * 6) Optionally suppress OpenCV internal threading to avoid oversubscription.
 *
 * @param policy Runtime policy parameters (ORT threads, OpenMP config, memory policy toggles).
 * @param verbose If true, prints topology and configuration diagnostics.
 * @return @ref idet::Status::Ok() on success, or an error status on failure.
 *
 * @note
 * This function should be called early, before creating ONNX Runtime sessions and before
 * entering OpenMP parallel regions, to avoid thread pool initialization with an undesired
 * affinity and/or memory policy.
 */
idet::Status setup_runtime_policy_impl(const idet::RuntimePolicy& policy, bool verbose) noexcept {
    Status status = Status::Ok();

    try {
        const std::size_t ort_intra_th = clamp_threads_(policy.ort_intra_threads);
        const std::size_t ort_inter_th = clamp_threads_(policy.ort_inter_threads);
        const std::size_t tile_omp_th = clamp_threads_(policy.tile_omp_threads);

        /**
         * @details
         * Compute a conservative estimate of "peak concurrency" requested by the configuration.
         *
         * Rationale:
         * - In tiling mode, OpenMP often provides most parallelism => desired ~= tile_omp_th.
         * - In non-tiling mode, ORT thread pools dominate => desired ~= max(intra, inter).
         * - If both ORT intra and inter are > 1, concurrent activity can exceed either value;
         *   a simple upper bound is intra + inter.
         *
         * The final desired thread budget is chosen as a conservative sum:
         *   desired_threads = tile_omp_th + ort_peak
         * so that tiling workers and ORT workers are less likely to oversubscribe a tight CPU mask.
         *
         * @note
         * This is an estimate. The true number of runnable threads depends on ORT execution patterns,
         * OpenMP runtime behavior, and operator-specific parallelism.
         */
        std::size_t ort_peak = 1;
        if (ort_intra_th > 1 && ort_inter_th > 1) {
            ort_peak = ort_intra_th + ort_inter_th;
        } else {
            ort_peak = std::max<std::size_t>(ort_intra_th, ort_inter_th);
        }
        const std::size_t desired_threads = tile_omp_th + ort_peak;

        /**
         * @details
         * Bind CPUs (and optionally apply best-effort NUMA policy).
         *
         * IMPORTANT: must be executed before initializing the OpenMP runtime and before creating ORT
         * thread pools / sessions. Otherwise, runtimes may cache thread counts and/or affinity masks.
         */
        auto br = apply_process_placement_policy(policy, desired_threads);
        if (!br.ok()) return br;
        if (!br.message.empty()) {
            // Non-fatal warnings may be propagated via Status message.
            status.message = std::string("warning: ") + br.message;
        }

        /**
         * @details
         * Topology printout is done after applying placement so that diagnostics reflect the final
         * process-available set and chosen placement policy.
         */
        if (verbose) {
            auto topo = detect_topology();
            print_topology(topo);
        }

        /**
         * @details
         * Diagnostic verification:
         * - Ensure all current threads have affinity inside the allowed/selected CPU set.
         * - Optionally verify page placement against Mems_allowed_list (Linux-only).
         *
         * @note
         * These checks can be expensive; they are primarily intended for debugging and validation.
         */
        {
            auto vr_aff = verify_all_threads_affinity_subset(verbose);
            if (!vr_aff.ok()) return vr_aff;

            auto vr_pg = verify_buffer_pages_on_nodes(0.95, verbose);
            if (!vr_pg.ok()) return vr_pg;
        }

        /**
         * @details
         * Configure OpenMP affinity and determinism defaults.
         *
         * The helper is expected to set OMP_* environment variables before any OpenMP runtime calls
         * (or as early as feasible) to avoid runtime initialization with unexpected values.
         *
         * @note
         * Some OpenMP runtimes may already be initialized by other libraries; in such cases only
         * a subset of settings may take effect.
         */
        configure_openmp_affinity(tile_omp_th, verbose);

        /**
         * @details
         * Optionally suppress OpenCV internal threading to avoid contention with ORT/OpenMP.
         *
         * The library keeps OpenCV SIMD optimizations enabled but forces OpenCV to use a single
         * thread (OpenCV thread pool disabled).
         *
         * @warning
         * OpenCV threading settings are global and affect all OpenCV usage in the process.
         */
        if (policy.suppress_opencv) {
            cv::setUseOptimized(true); // keep SIMD optimizations
            cv::setNumThreads(1);      // disable OpenCV thread pool
        }

        return status;

    } catch (const std::exception& e) {
        return Status::Internal(std::string("setup_runtime_policy threw: ") + e.what());
    } catch (...) {
        return Status::Internal("setup_runtime_policy threw (unknown)");
    }
}

} // namespace idet::platform
