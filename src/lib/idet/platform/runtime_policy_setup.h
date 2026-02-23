/**
 * @file runtime_policy_setup.h
 * @ingroup idet_platform
 * @brief Runtime policy application: CPU affinity + OpenMP/OpenCV threading coordination.
 *
 * @details
 * Declares the internal platform entry point that applies @ref idet::RuntimePolicy to the
 * current process in a best-effort, "do it early" manner.
 *
 * Responsibilities (typical implementation flow):
 * - Detect CPU topology and the effective CPU set available to the process (cpuset/cgroups/affinity-aware).
 * - Choose a deterministic CPU subset sized for the expected concurrency and apply it as the affinity mask
 *   for all current threads (and rely on inheritance for future threads).
 * - Configure OpenMP affinity/placement so that OpenMP worker threads stay within the selected CPU mask.
 * - Optionally suppress OpenCV internal thread pools to avoid oversubscription.
 *
 * Why this exists:
 * - ORT and OpenMP runtimes tend to initialize thread pools lazily but "early enough" that once initialized
 *   they may ignore later environment/affinity changes.
 * - A single, centralized setup point makes it easier to guarantee the intended call order and to debug
 *   effective runtime configuration.
 *
 * @note
 * This function should be called as early as possible in the process lifetime:
 * before creating ONNX Runtime sessions, before entering any OpenMP parallel regions, and ideally before
 * other libraries initialize their own thread pools.
 *
 * @warning
 * Many of the configured knobs are process-global (environment variables, OpenCV global settings).
 * Calling this function can affect other components within the same process.
 */

#pragma once

#include "idet.h"
#include "status.h"

namespace idet::platform {

/**
 * @brief Applies the runtime policy to the current process (internal implementation).
 *
 * @details
 * This function is the platform-layer implementation behind the public API
 * @ref idet::setup_runtime_policy.
 *
 * Expected side effects (implementation-defined, best-effort):
 * - May apply CPU affinity to all current threads (Linux: via /proc/self/task + sched_setaffinity).
 * - May configure OpenMP via @ref idet::platform::configure_openmp_affinity (typically through env vars and/or
 *   OpenMP runtime calls), aiming to keep OpenMP workers inside the process CPU mask.
 * - May call @c cv::setNumThreads(1) when @ref idet::RuntimePolicy::suppress_opencv is enabled.
 *
 * Error reporting:
 * - Returns @ref idet::Status::Invalid when user-provided policy values are inconsistent or cannot be applied.
 * - Returns @ref idet::Status::Internal on unexpected failures (platform calls, missing prerequisites).
 *
 * @param policy Runtime policy containing ORT/OpenMP thread counts and affinity hints.
 * @param verbose If true, prints detected topology, applied CPU mask, and verification diagnostics.
 *
 * @return @ref idet::Status::Ok() on success; otherwise a non-OK status with a diagnostic message.
 *
 * @par Threading contract
 * This function may:
 * - set CPU affinity mask for the process and all currently existing threads,
 * - set OpenMP-related environment variables and/or OpenMP runtime settings,
 * - suppress OpenCV internal threading globally when requested.
 *
 * @warning
 * Must be called before creating ORT sessions if you rely on strict affinity and deterministic thread
 * placement. If ORT/OpenMP thread pools are initialized before this call, only partial application may occur.
 */
idet::Status setup_runtime_policy_impl(const idet::RuntimePolicy& policy, bool verbose) noexcept;

} // namespace idet::platform
