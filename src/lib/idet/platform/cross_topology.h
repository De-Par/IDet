/**
 * @file cross_topology.h
 * @ingroup idet_platform
 * @brief CPU topology discovery, process/thread CPU binding, and NUMA diagnostics (Linux-first).
 *
 * This header declares Linux-first utilities for:
 * - process-aware CPU topology discovery (cpuset/cgroups/affinity-aware),
 * - deterministic CPU placement for a requested concurrency,
 * - optional best-effort NUMA memory policy setup (Linux + libnuma),
 * - diagnostics:
 *   - verify current threads' affinity is within an allowed CPU set,
 *   - verify sampled pages of a buffer reside on allowed NUMA nodes (Linux).
 *
 * Correctness notes for OpenMP + ONNX Runtime:
 * - Call @ref apply_process_placement_policy as early as possible:
 *   before any OpenMP parallel region and before creating ORT sessions (thread pools).
 * - CPU affinity is applied to all currently existing threads; future threads usually inherit it.
 * - On Linux, memory policy is per-thread; apply early and follow first-touch allocation to benefit.
 *
 * @note
 * This is an internal header and is not part of the stable public API.
 */

#pragma once

#include "idet.h"
#include "status.h"

#include <cstddef>
#include <vector>

namespace idet::platform {

/**
 * @brief Per-socket (CPU package) topology summary.
 */
struct SocketInfo {
    /** @brief Linux @c physical_package_id; -1 if unknown/unavailable. */
    int socket_id = -1;

    /** @brief Logical CPU count in this socket (system view). */
    unsigned logical_cores = 0;

    /** @brief Best-effort physical core count in this socket. */
    unsigned physical_cores = 0;

    /** @brief All logical CPUs in this socket (system view). */
    std::vector<int> logical_cpu_ids;

    /** @brief Subset of CPUs available to this process (cpuset/affinity). */
    std::vector<int> available_cpu_ids;

    /**
     * @brief SMT sibling groups per physical core (best-effort).
     *
     * Each entry is a list of CPU IDs that are siblings of one physical core.
     * Empty if the platform cannot provide grouping.
     */
    std::vector<std::vector<int>> core_siblings;
};

/**
 * @brief Process-aware machine topology summary.
 */
struct Topology {
    /** @brief Total logical CPUs visible to the OS (online). */
    unsigned total_logical = 0;

    /** @brief Best-effort total physical core count. */
    unsigned total_physical = 0;

    /** @brief Number of CPU sockets detected. */
    unsigned socket_count = 0;

    /** @brief Online CPU IDs (system view). */
    std::vector<int> all_cpu_ids;

    /** @brief Effective CPUs allowed for this process (cpuset/affinity). */
    std::vector<int> available_cpu_ids;

    /** @brief Per-socket details. */
    std::vector<SocketInfo> sockets;
};

/**
 * @brief Detects CPU topology and the process-available CPU set.
 *
 * @return Populated topology summary. Fields may be partially populated on platforms
 *         that do not expose full topology information.
 */
Topology detect_topology();

/**
 * @brief Prints a human-readable topology summary (to stdout).
 *
 * @param topology Topology information returned by @ref detect_topology.
 */
void print_topology(const Topology& topology);

/**
 * @brief Applies deterministic CPU placement for desired concurrency and optional NUMA policy.
 *
 * CPU selection policy (best-effort):
 * - Prefer a single socket if it can host all desired threads.
 * - Otherwise compactly spill to additional sockets (use as few as possible).
 * - Physical-first ordering where possible: fill one thread per physical core before using SMT siblings.
 *
 * Affinity application:
 * - Applies the chosen CPU mask to all currently existing threads (via @c /proc/self/task on Linux).
 *
 * NUMA policy (Linux + libnuma, best-effort, per-thread):
 * - Enabled only if @c runtime_policy.soft_mem_bind is true.
 * - @c runtime_policy.numa_mem_policy selects the policy:
 *   - @c Latency: prefer local NUMA node (or first allowed)
 *   - @c Throughput: interleave across allowed nodes (or prefer a broader placement)
 *   - @c Strict: enforce membind to the allowed node mask (may fail if constraints cannot be met)
 *
 * @param runtime_policy Runtime policy controlling NUMA and related settings.
 * @param desired_threads Desired level of concurrency for which CPUs should be selected and bound.
 * @return @ref idet::Status::Ok() on success, otherwise an error status describing the failure.
 *
 * @note
 * On non-Linux platforms this is expected to be a no-op and return @ref idet::Status::Ok().
 */
idet::Status apply_process_placement_policy(const idet::RuntimePolicy& runtime_policy, std::size_t desired_threads);

/**
 * @brief Diagnostic: verifies all current threads' affinity is a subset of @p allowed_cpus.
 *
 * @param allowed_cpus Allowed CPU IDs.
 * @param verbose If true, prints a summary on success and detailed info on errors.
 * @return @ref idet::Status::Ok() if all threads are within the allowed set, otherwise an error status.
 */
idet::Status verify_all_threads_affinity_subset(const std::vector<int>& allowed_cpus, bool verbose = true);

/**
 * @brief Diagnostic: verifies all current threads' affinity is within the current process allowed CPUs.
 *
 * Uses @ref detect_topology().available_cpu_ids (falls back to @ref Topology::all_cpu_ids if empty).
 *
 * @param verbose If true, prints a summary on success and detailed info on errors.
 * @return @ref idet::Status::Ok() if all threads are within the allowed set, otherwise an error status.
 */
idet::Status verify_all_threads_affinity_subset(bool verbose = true);

/**
 * @brief Diagnostic: verifies sampled pages of a user-provided buffer reside on allowed NUMA nodes (Linux).
 *
 * @param base Base pointer of the buffer.
 * @param bytes Buffer size in bytes.
 * @param allowed_nodes NUMA node IDs considered valid.
 * @param min_ratio Minimum fraction of sampled pages that must be on allowed nodes.
 * @param verbose If true, prints distribution statistics.
 * @return @ref idet::Status::Ok() if the sampled distribution satisfies the constraint, otherwise an error status.
 */
idet::Status verify_buffer_pages_on_nodes(void* base, std::size_t bytes, const std::vector<int>& allowed_nodes,
                                          double min_ratio = 0.95, bool verbose = true);

/**
 * @brief Convenience diagnostic: allocates and first-touches a test buffer and checks it against Mems_allowed_list.
 *
 * The function:
 * - allocates @p bytes bytes on heap,
 * - performs a first-touch write pass,
 * - parses allowed nodes from @c /proc/self/status: @c Mems_allowed_list,
 * - calls @ref verify_buffer_pages_on_nodes with that node list.
 *
 * @param min_ratio Minimum fraction of sampled pages that must be on allowed nodes.
 * @param verbose If true, prints distribution statistics.
 * @param bytes Buffer size to allocate and test.
 * @return @ref idet::Status::Ok() on success, otherwise an error status.
 *
 * @note
 * Useful for quick smoke tests; prefer the explicit overload for production buffers.
 */
idet::Status verify_buffer_pages_on_nodes(double min_ratio = 0.95, bool verbose = true,
                                          std::size_t bytes = 256ull * 1024 * 1024);

} // namespace idet::platform
