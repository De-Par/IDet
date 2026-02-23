/**
 * @file omp_config.h
 * @ingroup idet_platform
 * @brief OpenMP affinity configuration helpers.
 *
 * This header declares utilities for configuring OpenMP placement and binding in a deterministic
 * and performance-oriented way.
 *
 * Design goals:
 * - Configure OpenMP deterministically and early in the process lifetime.
 * - Avoid initializing the OpenMP runtime before affinity-related environment variables are set.
 * - Keep OpenMP worker threads within a CPU affinity mask established by a separate binding step
 *   (for example, @c bind_for_threads()).
 *
 * Runtime variability:
 * - OpenMP affinity behavior depends on the OpenMP runtime (GNU libgomp, LLVM libomp, Intel runtime, etc.).
 * - Not all values of @c OMP_PLACES / @c OMP_PROC_BIND behave identically across runtimes and platforms.
 *
 * @note
 * This is an internal header and is not part of the stable public API.
 */

#pragma once

#include <cstddef>
#include <string>

namespace idet::platform {

/**
 * @brief Configures OpenMP thread placement and binding policy.
 *
 * This function configures OpenMP affinity using environment variables and/or runtime APIs
 * (implementation-defined). The intent is to prevent thread oversubscription and keep OpenMP
 * execution inside a pre-established CPU mask.
 *
 * Recommended call order:
 *  1) @c detect_topology()
 *  2) @c bind_for_threads(...)
 *  3) @ref configure_openmp_affinity(...)
 *  4) Create ORT sessions / enter OpenMP parallel regions
 *
 * @param tile_omp_threads Number of OpenMP threads intended for tile processing.
 *        If 0, the implementation should derive a default from the effective CPU count of the
 *        current process affinity mask (implementation-defined).
 * @param verbose If true, prints/logs the effective OpenMP configuration (implementation-defined).
 *
 * @warning
 * Many OpenMP settings are process-global. Calling this function after the OpenMP runtime has been
 * initialized may have no effect or only partial effect.
 *
 * @note
 * The meaning of "effective CPU count" depends on the platform binding mechanism and the current
 * process/thread affinity state at the time of the call.
 */
void configure_openmp_affinity(std::size_t tile_omp_threads, bool verbose);

} // namespace idet::platform
