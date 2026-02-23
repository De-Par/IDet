/**
 * @file cross_topology.cpp
 * @ingroup idet_platform
 * @brief CPU topology discovery, affinity application, and NUMA diagnostics implementation.
 *
 * @details
 * Implements APIs declared in @ref cross_topology.h:
 * - process-aware topology detection (online CPUs + cpuset/affinity constraints),
 * - deterministic CPU selection (single-socket preference, physical-first ordering),
 * - applying CPU affinity to all existing threads (/proc/self/task on Linux),
 * - diagnostics:
 *   - verify threads' affinity subset,
 *   - verify sampled buffer pages locality via move_pages() (Linux).
 *
 * NUMA policy:
 * - Optional best-effort policy via libnuma when built with USE_LIBNUMA and headers available.
 *
 * Portability:
 * - Linux: full functionality (sysfs + /proc + sched + move_pages).
 * - macOS: best-effort topology via sysctl; no CPU affinity/NUMA diagnostics here.
 */

#include "cross_topology.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
    #if defined(__has_include) && __has_include(<filesystem>)
        #include <filesystem>
namespace fs = std::filesystem;
    #else
        #error "[ERROR] C++17 <filesystem> is required on Linux for sysfs enumeration"
    #endif

    #include <dirent.h>
    #include <numaif.h> // move_pages
    #include <sched.h>
    #include <unistd.h>

#elif defined(__APPLE__)
    #include <numeric>
    #include <sys/sysctl.h>
    #include <thread>
#else
    #error "[ERROR] Compilation available only for Linux or macOS"
#endif

/**
 * libnuma availability:
 * - @c USE_LIBNUMA is a build-time toggle (project-specific).
 * - Even if enabled, the header must be present and the runtime must support NUMA.
 */
#if defined(__linux__) && defined(USE_LIBNUMA) && defined(__has_include) && __has_include(<numa.h>)
    #include <numa.h>
    #define HAS_LIBNUMA 1
#else
    #define HAS_LIBNUMA 0
#endif

namespace idet::platform {

namespace {

// ----------------------------- tiny helpers -----------------------------

/**
 * @brief Formats a sorted list of IDs into a compact range representation.
 *
 * @details
 * Example: [0,1,2,3,8,10,11] -> "0-3,8,10-11"
 */
static std::string format_id_list(const std::vector<int>& ids_raw) {
    if (ids_raw.empty()) return "none";

    std::vector<int> v = ids_raw;
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());

    std::ostringstream oss;
    int range_start = v[0];
    int prev = v[0];

    auto flush_range = [&](bool first) {
        if (!first) oss << ",";
        if (range_start == prev)
            oss << range_start;
        else
            oss << range_start << "-" << prev;
    };

    bool first = true;
    for (std::size_t i = 1; i < v.size(); ++i) {
        const int x = v[i];
        if (x == prev + 1) {
            prev = x;
            continue;
        }
        flush_range(first);
        first = false;
        range_start = prev = x;
    }
    flush_range(first);

    return oss.str();
}

#if defined(__linux__)

/**
 * @brief Returns a copy of the string with leading/trailing ASCII whitespace removed.
 *
 * @note
 * This is a small utility intended for parsing procfs/sysfs text files.
 */
static std::string trim_copy(std::string s) {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!s.empty() && is_space(static_cast<unsigned char>(s.front())))
        s.erase(s.begin());
    while (!s.empty() && is_space(static_cast<unsigned char>(s.back())))
        s.pop_back();
    return s;
}

/**
 * @brief Parses Linux-style CPU/node list strings like "0-3,8,10-11" into a sorted unique list.
 *
 * @details
 * Accepts:
 * - comma-separated items,
 * - each item is either a single integer or a closed range "a-b".
 *
 * Invalid items are ignored (best-effort parsing).
 */
static std::vector<int> parse_cpu_list_string(const std::string& raw) {
    std::string s = trim_copy(raw);
    std::vector<int> ids;

    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (item.empty()) continue;

        const auto dash = item.find('-');
        if (dash == std::string::npos) {
            try {
                ids.push_back(std::stoi(item));
            } catch (...) {
            }
        } else {
            try {
                const int a = std::stoi(trim_copy(item.substr(0, dash)));
                const int b = std::stoi(trim_copy(item.substr(dash + 1)));
                if (a <= b) {
                    for (int x = a; x <= b; ++x)
                        ids.push_back(x);
                }
            } catch (...) {
            }
        }
    }

    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

/**
 * @brief Reads a single integer from a sysfs file (first line).
 *
 * @return True on success, false if the file is missing/unreadable or parsing fails.
 */
static bool read_int_from_file(const fs::path& p, int& out) {
    std::ifstream f(p);
    if (!f) return false;
    std::string s;
    std::getline(f, s);
    s = trim_copy(s);
    if (s.empty()) return false;
    try {
        out = std::stoi(s);
        return true;
    } catch (...) {
        return false;
    }
}

/**
 * @brief Enumerates online CPU IDs using sysfs.
 *
 * @details
 * First tries @c /sys/devices/system/cpu/online (preferred).
 * If unavailable, falls back to directory enumeration of @c cpuN directories.
 */
static std::vector<int> linux_all_cpu_ids() {
    std::string online;
    {
        std::ifstream f("/sys/devices/system/cpu/online");
        if (f) std::getline(f, online);
    }
    if (!trim_copy(online).empty()) return parse_cpu_list_string(online);

    std::vector<int> ids;
    const fs::path cpu_root = "/sys/devices/system/cpu";
    for (const auto& entry : fs::directory_iterator(cpu_root)) {
        if (!entry.is_directory()) continue;

        const std::string name = entry.path().filename().string();
        if (name.rfind("cpu", 0) != 0 || name.size() <= 3) continue;

        const std::string digits = name.substr(3);
        bool ok = !digits.empty();
        for (unsigned char ch : digits)
            ok = ok && (std::isdigit(ch) != 0);
        if (!ok) continue;

        try {
            ids.push_back(std::stoi(digits));
        } catch (...) {
        }
    }

    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

/**
 * @brief Returns the current process affinity mask as a list of CPU IDs.
 *
 * @details
 * Uses @c sched_getaffinity(2). When available, uses the dynamic CPU set API (@c CPU_ALLOC)
 * to handle systems with more than @c CPU_SETSIZE CPUs.
 */
static std::vector<int> linux_affinity_cpu_ids() {
    long nconf = sysconf(_SC_NPROCESSORS_CONF);
    int max_cpu = (nconf > 0) ? static_cast<int>(nconf) : CPU_SETSIZE;
    if (max_cpu <= 0) max_cpu = CPU_SETSIZE;

    #if defined(CPU_ALLOC)
    const size_t setsize = CPU_ALLOC_SIZE(max_cpu);
    cpu_set_t* set = CPU_ALLOC(max_cpu);
    if (!set) return {};
    CPU_ZERO_S(setsize, set);

    if (sched_getaffinity(0, setsize, set) != 0) {
        CPU_FREE(set);
        return {};
    }

    std::vector<int> cpus;
    cpus.reserve(static_cast<std::size_t>(max_cpu));
    for (int c = 0; c < max_cpu; ++c) {
        if (CPU_ISSET_S(c, setsize, set)) cpus.push_back(c);
    }
    CPU_FREE(set);

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
    #else
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return {};
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &set)) cpus.push_back(c);
    return cpus;
    #endif
}

/**
 * @brief Reads @c Cpus_allowed_list from @c /proc/self/status as a fallback source of process-allowed CPUs.
 *
 * @note
 * In containerized environments, @c Cpus_allowed_list often reflects cpuset/cgroup constraints.
 */
static std::vector<int> linux_available_cpu_ids_via_proc() {
    std::ifstream f("/proc/self/status");
    if (!f) return {};
    std::string line;
    const std::string key = "Cpus_allowed_list:";
    while (std::getline(f, line)) {
        if (line.rfind(key, 0) == 0) {
            const std::string list = trim_copy(line.substr(key.size()));
            return parse_cpu_list_string(list);
        }
    }
    return {};
}

/**
 * @brief Determines the effective CPU set available to this process.
 *
 * @details
 * Preference order:
 * 1) @c sched_getaffinity(2)
 * 2) @c /proc/self/status: Cpus_allowed_list
 * 3) sysfs online CPU list
 */
static std::vector<int> linux_available_cpu_ids() {
    auto a = linux_affinity_cpu_ids();
    if (!a.empty()) return a;

    auto p = linux_available_cpu_ids_via_proc();
    if (!p.empty()) return p;

    return linux_all_cpu_ids();
}

/**
 * @brief Applies a CPU affinity mask to a specific thread (TID).
 *
 * @note
 * Uses @c sched_setaffinity(2). When the thread disappears between enumeration and the call,
 * ESRCH is treated as non-fatal for "apply-to-all-threads" workflows.
 */
static idet::Status linux_set_affinity_tid(pid_t tid, const std::vector<int>& cpus) {
    if (cpus.empty()) return idet::Status::Invalid("linux_set_affinity_tid: empty CPU list");

    const int max_id = *std::max_element(cpus.begin(), cpus.end());
    if (max_id < 0) return idet::Status::Invalid("linux_set_affinity_tid: negative CPU id");

    #if defined(CPU_ALLOC)
    const int max_cpu = max_id + 1;
    const size_t setsize = CPU_ALLOC_SIZE(max_cpu);
    cpu_set_t* set = CPU_ALLOC(max_cpu);
    if (!set) return idet::Status::Invalid("linux_set_affinity_tid: CPU_ALLOC failed");

    CPU_ZERO_S(setsize, set);
    for (int c : cpus)
        if (c >= 0) CPU_SET_S(c, setsize, set);

    const int rc = sched_setaffinity(tid, setsize, set);
    const int err = errno;
    CPU_FREE(set);

    if (rc != 0) {
        // ESRCH = thread died between enumeration and set; not fatal for "apply all threads" use-case.
        if (err == ESRCH) return idet::Status::Ok();
        std::ostringstream oss;
        oss << "sched_setaffinity(tid=" << tid << ") failed: " << std::strerror(err);
        return idet::Status::Invalid(oss.str());
    }
    return idet::Status::Ok();
    #else
    if (max_id >= CPU_SETSIZE) {
        std::ostringstream oss;
        oss << "linux_set_affinity_tid: CPU id " << max_id << " exceeds CPU_SETSIZE=" << CPU_SETSIZE;
        return idet::Status::Invalid(oss.str());
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus)
        if (c >= 0) CPU_SET(c, &set);

    const int rc = sched_setaffinity(tid, sizeof(set), &set);
    const int err = errno;
    if (rc != 0) {
        if (err == ESRCH) return idet::Status::Ok();
        std::ostringstream oss;
        oss << "sched_setaffinity(tid=" << tid << ") failed: " << std::strerror(err);
        return idet::Status::Invalid(oss.str());
    }
    return idet::Status::Ok();
    #endif
}

/**
 * @brief Applies a CPU affinity mask to all currently existing threads in the process.
 *
 * @details
 * Enumerates @c /proc/self/task, parses each directory name as a TID, and applies the mask.
 * If enumeration fails, falls back to applying affinity to the current thread only.
 *
 * @return The first encountered failure (if any). ESRCH is ignored as non-fatal.
 */
static idet::Status linux_set_affinity_all_threads(const std::vector<int>& cpus) {
    DIR* d = opendir("/proc/self/task");
    if (!d) {
        // Fallback: at least apply to the current thread.
        return linux_set_affinity_tid(0, cpus);
    }

    idet::Status out = idet::Status::Ok();

    while (auto* e = readdir(d)) {
        if (!e->d_name[0] || e->d_name[0] == '.') continue;

        char* endp = nullptr;
        const long tid = std::strtol(e->d_name, &endp, 10);
        if (!endp || *endp != '\0' || tid <= 0) continue;

        const auto r = linux_set_affinity_tid(static_cast<pid_t>(tid), cpus);
        if (!r.ok() && out.ok()) out = r; // keep first failure
    }

    closedir(d);
    return out;
}

/**
 * @brief Returns the socket (package) id for a logical CPU id (Linux sysfs).
 *
 * @return Socket id, or -1 if the information is unavailable.
 */
static int linux_cpu_socket_id(int cpu_id) {
    const fs::path topo_dir = fs::path("/sys/devices/system/cpu") / ("cpu" + std::to_string(cpu_id)) / "topology";
    int sid = -1;
    if (!read_int_from_file(topo_dir / "physical_package_id", sid)) return -1;
    return sid;
}

/**
 * @brief Returns the core id for a logical CPU id (Linux sysfs).
 *
 * @return Core id, or -1 if the information is unavailable.
 */
static int linux_cpu_core_id(int cpu_id) {
    const fs::path topo_dir = fs::path("/sys/devices/system/cpu") / ("cpu" + std::to_string(cpu_id)) / "topology";
    int cid = -1;
    if (!read_int_from_file(topo_dir / "core_id", cid)) return -1;
    return cid;
}

/**
 * @brief Builds a "physical-first" CPU order for CPUs that belong to a single socket.
 *
 * @details
 * Groups CPUs by core id (SMT siblings), then performs round-robin selection across cores:
 * - first pass picks one CPU from each core,
 * - next passes pick SMT siblings.
 *
 * This yields a compact order that prefers spreading across physical cores before SMT.
 */
static std::vector<int> physical_first_order(const std::vector<int>& cpus_in_socket) {
    std::map<int, std::vector<int>> core_to_cpus;
    for (int cpu : cpus_in_socket) {
        int core = linux_cpu_core_id(cpu);
        if (core < 0) core = 1000000 + cpu; // unique fallback
        core_to_cpus[core].push_back(cpu);
    }

    for (auto& [core, v] : core_to_cpus) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    std::vector<std::vector<int>> cores;
    cores.reserve(core_to_cpus.size());
    for (auto& [core, v] : core_to_cpus)
        cores.push_back(v);

    std::size_t max_sibs = 0;
    for (auto& v : cores)
        max_sibs = std::max(max_sibs, v.size());

    std::vector<int> ordered;
    ordered.reserve(cpus_in_socket.size());

    for (std::size_t pass = 0; pass < max_sibs; ++pass) {
        for (auto& sibs : cores) {
            if (pass < sibs.size()) ordered.push_back(sibs[pass]);
        }
    }
    return ordered;
}

// -------- NUMA helpers that do NOT require libnuma (sysfs + /proc) --------

/**
 * @brief Parses @c Mems_allowed_list from @c /proc/self/status.
 *
 * @details
 * This reflects the allowed NUMA nodes for memory allocations for the process under current
 * cpuset/cgroup constraints (if any).
 */
static std::vector<int> linux_allowed_mems_nodes() {
    std::ifstream f("/proc/self/status");
    if (!f) return {};

    std::string line;
    const std::string key = "Mems_allowed_list:";
    while (std::getline(f, line)) {
        if (line.rfind(key, 0) == 0) {
            const std::string list = trim_copy(line.substr(key.size()));
            return parse_cpu_list_string(list);
        }
    }
    return {};
}

/**
 * @brief Returns a map node_id -> cpus for all NUMA nodes visible in sysfs.
 */
static std::map<int, std::vector<int>> linux_numa_node_to_cpus() {
    std::map<int, std::vector<int>> out;
    const fs::path nodes_root = "/sys/devices/system/node";
    if (!fs::exists(nodes_root)) return out;

    for (const auto& e : fs::directory_iterator(nodes_root)) {
        if (!e.is_directory()) continue;

        const auto name = e.path().filename().string(); // node0
        if (name.rfind("node", 0) != 0) continue;

        const std::string idstr = name.substr(4);
        bool ok = !idstr.empty();
        for (unsigned char ch : idstr)
            ok = ok && (std::isdigit(ch) != 0);
        if (!ok) continue;

        int node_id = -1;
        try {
            node_id = std::stoi(idstr);
        } catch (...) {
            continue;
        }

        std::ifstream f(e.path() / "cpulist");
        std::string s;
        if (f) std::getline(f, s);

        auto cpus = parse_cpu_list_string(s);
        if (!cpus.empty()) out.emplace(node_id, std::move(cpus));
    }
    return out;
}

/**
 * @brief Determines which NUMA nodes overlap with the given CPU list.
 *
 * @return Sorted unique list of node ids that contain at least one CPU from @p cpus.
 */
static std::vector<int> nodes_for_cpus(const std::vector<int>& cpus) {
    const auto mapn = linux_numa_node_to_cpus();
    if (mapn.empty()) return {};

    std::set<int> S(cpus.begin(), cpus.end());
    std::vector<int> nodes;

    for (const auto& [node, ncpus] : mapn) {
        bool overlap = false;
        for (int c : ncpus) {
            if (S.count(c)) {
                overlap = true;
                break;
            }
        }
        if (overlap) nodes.push_back(node);
    }

    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    return nodes;
}

// -------- libnuma policy (optional) --------

    #if HAS_LIBNUMA

/**
 * @brief Applies a best-effort "soft" NUMA memory policy using libnuma (per-thread).
 *
 * @details
 * This is used by @ref apply_process_placement_policy when @c soft_mem_bind is enabled.
 * The policy is best-effort:
 * - @c Latency: choose a preferred node (try local CPU's node, else first in list).
 * - @c Throughput: prefer one node if there is only one; otherwise interleave across nodes.
 * - @c Strict: membind to the mask (can be restrictive).
 *
 * @warning
 * libnuma policies are typically per-thread. Apply early (before allocations) and rely on
 * first-touch allocation for best results.
 */
static idet::Status linux_apply_soft_mempolicy(const std::vector<int>& nodes, idet::NumaMemPolicy policy) {
    if (nodes.empty()) return idet::Status::Ok();
    if (numa_available() < 0) return idet::Status::Invalid("libnuma: numa_available() < 0");

    bitmask* mask = numa_allocate_nodemask();
    if (!mask) return idet::Status::Invalid("libnuma: numa_allocate_nodemask failed");

    numa_bitmask_clearall(mask);
    for (int n : nodes)
        if (n >= 0) numa_bitmask_setbit(mask, n);

    auto pick_latency_node = [&]() -> int {
        // Choose node of current CPU if it belongs to our nodes set; otherwise first node.
        const int cpu = sched_getcpu();
        if (cpu >= 0) {
            const int n = numa_node_of_cpu(cpu);
            if (n >= 0) {
                for (int x : nodes)
                    if (x == n) return n;
            }
        }
        return nodes[0];
    };

    switch (policy) {
    case idet::NumaMemPolicy::Latency:
        numa_set_preferred(pick_latency_node());
        break;

    case idet::NumaMemPolicy::Throughput:
        if (nodes.size() == 1)
            numa_set_preferred(nodes[0]);
        else
            numa_set_interleave_mask(mask);
        break;

    case idet::NumaMemPolicy::Strict:
        numa_set_membind(mask);
        break;
    }

    numa_free_nodemask(mask);
    return idet::Status::Ok();
}
    #endif // HAS_LIBNUMA

// -------- topology detection --------

/**
 * @brief Linux topology discovery implementation.
 *
 * @details
 * - @ref Topology::all_cpu_ids comes from sysfs online CPU ids.
 * - @ref Topology::available_cpu_ids comes from affinity/cpuset constraints and is intersected with all_cpu_ids.
 * - Per-socket aggregation is performed using sysfs topology (physical_package_id, core_id).
 * - If socket information is not available, a single-socket fallback is used.
 */
static Topology detect_linux() {
    Topology topo;

    topo.all_cpu_ids = linux_all_cpu_ids();
    topo.available_cpu_ids = linux_available_cpu_ids();

    // Ensure available ⊆ all (online).
    {
        std::set<int> all_set(topo.all_cpu_ids.begin(), topo.all_cpu_ids.end());
        std::vector<int> filtered;
        filtered.reserve(topo.available_cpu_ids.size());
        for (int c : topo.available_cpu_ids)
            if (all_set.count(c)) filtered.push_back(c);
        topo.available_cpu_ids.swap(filtered);
    }

    topo.total_logical = static_cast<unsigned>(topo.all_cpu_ids.size());

    struct Agg {
        std::set<int> core_ids;
        std::vector<int> cpus;
    };
    std::map<int, Agg> per_socket;

    for (int cpu_id : topo.all_cpu_ids) {
        const int socket_id = linux_cpu_socket_id(cpu_id);
        const int core_id = linux_cpu_core_id(cpu_id);
        if (socket_id < 0) continue;

        auto& agg = per_socket[socket_id];
        agg.cpus.push_back(cpu_id);
        if (core_id >= 0) agg.core_ids.insert(core_id);
    }

    // Fallback: no socket info.
    if (per_socket.empty()) {
        SocketInfo s;
        s.socket_id = 0;
        s.logical_cpu_ids = topo.all_cpu_ids;
        s.available_cpu_ids = topo.available_cpu_ids.empty() ? topo.all_cpu_ids : topo.available_cpu_ids;
        s.logical_cores = static_cast<unsigned>(s.logical_cpu_ids.size());
        s.physical_cores = s.logical_cores;

        topo.sockets.push_back(std::move(s));
        topo.socket_count = 1;
        topo.total_physical = topo.total_logical;
        return topo;
    }

    const std::set<int> avail_set(topo.available_cpu_ids.begin(), topo.available_cpu_ids.end());

    for (auto& [sid, agg] : per_socket) {
        std::sort(agg.cpus.begin(), agg.cpus.end());
        agg.cpus.erase(std::unique(agg.cpus.begin(), agg.cpus.end()), agg.cpus.end());

        SocketInfo s;
        s.socket_id = sid;
        s.logical_cpu_ids = agg.cpus;

        for (int c : s.logical_cpu_ids)
            if (avail_set.count(c)) s.available_cpu_ids.push_back(c);

        s.logical_cores = static_cast<unsigned>(s.logical_cpu_ids.size());
        s.physical_cores = agg.core_ids.empty() ? s.logical_cores : static_cast<unsigned>(agg.core_ids.size());

        // Core sibling groups for diagnostics/selection. Prefer process-available CPUs when possible.
        const auto& base = !s.available_cpu_ids.empty() ? s.available_cpu_ids : s.logical_cpu_ids;
        std::map<int, std::vector<int>> core_to;
        for (int cpu : base) {
            int core = linux_cpu_core_id(cpu);
            if (core < 0) core = 1000000 + cpu;
            core_to[core].push_back(cpu);
        }
        for (auto& [core, v] : core_to) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            s.core_siblings.push_back(v);
        }

        topo.sockets.push_back(std::move(s));
    }

    topo.socket_count = static_cast<unsigned>(topo.sockets.size());
    topo.total_physical = 0;
    for (auto& s : topo.sockets)
        topo.total_physical += s.physical_cores;

    return topo;
}

#endif // __linux__

#if defined(__APPLE__)

/**
 * @brief Reads a uint64 sysctl value by name.
 */
static bool sysctl_get_uint64(const char* name, uint64_t& out) {
    size_t size = sizeof(uint64_t);
    return sysctlbyname(name, &out, &size, nullptr, 0) == 0;
}

/**
 * @brief Reads an unsigned sysctl value by name, or returns @p defv if missing.
 */
static unsigned sysctl_get_uint_or(const char* name, unsigned defv) {
    uint64_t tmp = 0;
    if (!sysctl_get_uint64(name, tmp)) return defv;
    return static_cast<unsigned>(tmp);
}

/**
 * @brief macOS topology discovery implementation (coarse).
 *
 * @details
 * macOS does not expose Linux-like socket/core topology and cpuset constraints in the same way.
 * This implementation:
 * - uses sysctl to query logical and physical CPU counts,
 * - creates a simple contiguous CPU id list [0..logical-1],
 * - estimates "packages" (sockets) via @c hw.packages if available; otherwise assumes 1.
 *
 * @note
 * The per-socket CPU assignment is a simple contiguous partition for reporting purposes.
 */
static Topology detect_macos() {
    Topology topo;

    unsigned logical = sysctl_get_uint_or("hw.logicalcpu_max", 0);
    if (!logical)
        logical = sysctl_get_uint_or("hw.logicalcpu", static_cast<unsigned>(std::thread::hardware_concurrency()));

    unsigned physical = sysctl_get_uint_or("hw.physicalcpu_max", 0);
    if (!physical) physical = sysctl_get_uint_or("hw.physicalcpu", logical);

    topo.total_logical = logical ? logical : 1u;
    topo.total_physical = physical ? physical : topo.total_logical;

    topo.all_cpu_ids.resize(topo.total_logical);
    std::iota(topo.all_cpu_ids.begin(), topo.all_cpu_ids.end(), 0);
    topo.available_cpu_ids = topo.all_cpu_ids;

    unsigned packages = sysctl_get_uint_or("hw.packages", 0);
    if (!packages) packages = 1;
    topo.socket_count = packages;

    std::vector<unsigned> per_sock_logical(packages, topo.total_logical / packages);
    for (unsigned i = 0; i < topo.total_logical % packages; ++i)
        per_sock_logical[i]++;

    std::vector<unsigned> per_sock_physical(packages, topo.total_physical / packages);
    for (unsigned i = 0; i < topo.total_physical % packages; ++i)
        per_sock_physical[i]++;

    int cursor = 0;
    for (unsigned sid = 0; sid < packages; ++sid) {
        SocketInfo s;
        s.socket_id = static_cast<int>(sid);
        s.logical_cores = per_sock_logical[sid];
        s.physical_cores = per_sock_physical[sid];
        for (unsigned k = 0; k < s.logical_cores && cursor < static_cast<int>(topo.total_logical); ++k)
            s.logical_cpu_ids.push_back(cursor++);
        s.available_cpu_ids = s.logical_cpu_ids;
        topo.sockets.push_back(std::move(s));
    }

    return topo;
}

#endif // __APPLE__

} // namespace

// ----------------------------- Public API -----------------------------

Topology detect_topology() {
#if defined(__linux__)
    return detect_linux();
#elif defined(__APPLE__)
    return detect_macos();
#else
    Topology t;
    t.total_logical = 1;
    t.total_physical = 1;
    t.socket_count = 1;
    t.all_cpu_ids = {0};
    t.available_cpu_ids = {0};
    SocketInfo s;
    s.socket_id = 0;
    s.logical_cores = 1;
    s.physical_cores = 1;
    s.logical_cpu_ids = {0};
    s.available_cpu_ids = {0};
    t.sockets.push_back(std::move(s));
    return t;
#endif
}

void print_topology(const Topology& topo) {
    std::cout << "\n=== CPU Topology ===\n";
    std::cout << "Sockets:        " << topo.socket_count << "\n";
    std::cout << "Total logical:  " << topo.total_logical << "\n";
    std::cout << "Total physical: " << topo.total_physical << "\n";
    std::cout << "All CPU IDs:       " << format_id_list(topo.all_cpu_ids) << " (" << topo.all_cpu_ids.size() << ")\n";
    std::cout << "Available CPU IDs: " << format_id_list(topo.available_cpu_ids) << " ("
              << topo.available_cpu_ids.size() << ")\n";

    std::cout << "\n=== Per-socket ===\n";
    for (const auto& s : topo.sockets) {
        std::cout << "Socket_id=" << s.socket_id << " | logical=" << s.logical_cores
                  << " | physical=" << s.physical_cores << "\n";
        std::cout << "    All CPU IDs:       " << format_id_list(s.logical_cpu_ids) << " (" << s.logical_cpu_ids.size()
                  << ")\n";
        std::cout << "    Available CPU IDs: " << format_id_list(s.available_cpu_ids) << " ("
                  << s.available_cpu_ids.size() << ")\n";
    }
    std::cout << "\n" << std::flush;
}

idet::Status apply_process_placement_policy(const idet::RuntimePolicy& runtime_policy, std::size_t desired_threads) {
#if !defined(__linux__)
    (void)runtime_policy;
    (void)desired_threads;
    return idet::Status::Ok();
#else
    if (desired_threads == 0) {
        return idet::Status::Internal("apply_process_placement_policy: desired_threads must be > 0");
    }

    const auto topology = detect_topology();
    const auto& global_avail = !topology.available_cpu_ids.empty() ? topology.available_cpu_ids : topology.all_cpu_ids;

    if (global_avail.empty()) {
        return idet::Status::Internal("apply_process_placement_policy: no CPUs available to this process");
    }
    if (desired_threads > global_avail.size()) {
        std::ostringstream oss;
        oss << "apply_process_placement_policy: desired_threads=" << desired_threads << " but only "
            << global_avail.size() << " CPUs available (cpuset/affinity). Refuse oversubscription.";
        return idet::Status::Invalid(oss.str());
    }

    // Build socket candidates: intersect socket CPUs with global_avail, then apply a physical-first ordering.
    const std::set<int> G(global_avail.begin(), global_avail.end());

    struct SockCand {
        int socket_id = -1;
        std::vector<int> avail_ordered;
        bool contains_current_cpu = false;
    };

    const int current_cpu = sched_getcpu();

    std::vector<SockCand> cands;
    cands.reserve(topology.sockets.size());

    for (const auto& s : topology.sockets) {
        const auto& src = !s.available_cpu_ids.empty() ? s.available_cpu_ids : s.logical_cpu_ids;

        std::vector<int> avail;
        avail.reserve(src.size());
        for (int c : src)
            if (G.count(c)) avail.push_back(c);

        std::sort(avail.begin(), avail.end());
        avail.erase(std::unique(avail.begin(), avail.end()), avail.end());
        if (avail.empty()) continue;

        SockCand sc;
        sc.socket_id = s.socket_id;
        sc.contains_current_cpu =
            (current_cpu >= 0) && (std::find(avail.begin(), avail.end(), current_cpu) != avail.end());
        sc.avail_ordered = physical_first_order(avail);
        cands.push_back(std::move(sc));
    }

    // If socket decomposition is missing/unreliable, treat the entire available set as a single candidate.
    if (cands.empty()) {
        SockCand sc;
        sc.socket_id = 0;
        sc.contains_current_cpu = (current_cpu >= 0) && (std::find(global_avail.begin(), global_avail.end(),
                                                                   current_cpu) != global_avail.end());
        sc.avail_ordered = physical_first_order(global_avail);
        cands.push_back(std::move(sc));
    }

    // Prefer the socket that contains the current CPU, then prefer larger candidates (more CPUs available).
    std::sort(cands.begin(), cands.end(), [](const SockCand& a, const SockCand& b) {
        if (a.contains_current_cpu != b.contains_current_cpu) return a.contains_current_cpu > b.contains_current_cpu;
        if (a.avail_ordered.size() != b.avail_ordered.size()) return a.avail_ordered.size() > b.avail_ordered.size();
        return a.socket_id < b.socket_id;
    });

    std::vector<int> chosen_cpus;
    chosen_cpus.reserve(desired_threads);

    // 1) Single-socket placement if any socket can host all desired threads.
    for (const auto& sc : cands) {
        if (sc.avail_ordered.size() >= desired_threads) {
            chosen_cpus.assign(sc.avail_ordered.begin(),
                               sc.avail_ordered.begin() + static_cast<std::ptrdiff_t>(desired_threads));
            break;
        }
    }

    // 2) Compact spill across sockets: fill candidates in sorted preference order until enough CPUs are selected.
    if (chosen_cpus.empty()) {
        for (const auto& sc : cands) {
            for (int c : sc.avail_ordered) {
                if (chosen_cpus.size() == desired_threads) break;
                chosen_cpus.push_back(c);
            }
            if (chosen_cpus.size() == desired_threads) break;
        }
        if (chosen_cpus.size() != desired_threads) {
            return idet::Status::Internal(
                "apply_process_placement_policy: could not gather enough CPUs; inconsistent topology/cpuset?");
        }
    }

    // 3) Apply CPU affinity to ALL current threads in the process.
    auto r = linux_set_affinity_all_threads(chosen_cpus);
    if (!r.ok()) return r;

    // 4) Optional NUMA policy (best-effort; per-thread) using libnuma when available.
    if (runtime_policy.soft_mem_bind) {
    #if HAS_LIBNUMA
        auto nodes = nodes_for_cpus(chosen_cpus);

        // Filter to Mems_allowed_list if present (container/cpuset constrained environments).
        const auto allowed = linux_allowed_mems_nodes();
        if (!allowed.empty()) {
            const std::set<int> AN(allowed.begin(), allowed.end());
            std::vector<int> filtered;
            filtered.reserve(nodes.size());
            for (int n : nodes)
                if (AN.count(n)) filtered.push_back(n);
            nodes.swap(filtered);
        }

        if (!nodes.empty()) {
            auto mr = linux_apply_soft_mempolicy(nodes, runtime_policy.numa_mem_policy);
            if (!mr.ok()) return mr;
        }
    #else
        return idet::Status::Invalid(
            "apply_process_placement_policy: soft_mem_bind requested, but built without libnuma "
            "(define USE_LIBNUMA and install libnuma-dev)");
    #endif
    }
    return idet::Status::Ok();
#endif
}

// ----------------------------- Diagnostics -----------------------------

idet::Status verify_all_threads_affinity_subset(const std::vector<int>& allowed_cpus, bool verbose) {
#if !defined(__linux__)
    (void)allowed_cpus;
    (void)verbose;
    return idet::Status::Ok();
#else
    if (allowed_cpus.empty()) return idet::Status::Invalid("verify_all_threads_affinity_subset: empty allowed_cpus");

    const std::set<int> A(allowed_cpus.begin(), allowed_cpus.end());
    const int max_id = *std::max_element(allowed_cpus.begin(), allowed_cpus.end());
    if (max_id < 0) return idet::Status::Invalid("verify_all_threads_affinity_subset: negative CPU id");

    long nconf = sysconf(_SC_NPROCESSORS_CONF);
    int max_cpu = (nconf > 0) ? static_cast<int>(nconf) : (max_id + 1);
    max_cpu = std::max(max_cpu, max_id + 1);
    if (max_cpu <= 0) max_cpu = CPU_SETSIZE;

    DIR* d = opendir("/proc/self/task");
    if (!d) return idet::Status::Invalid("verify_all_threads_affinity_subset: cannot open /proc/self/task");

    int checked = 0;

    #if defined(CPU_ALLOC)
    const size_t setsize = CPU_ALLOC_SIZE(max_cpu);
    cpu_set_t* cur = CPU_ALLOC(max_cpu);
    if (!cur) {
        closedir(d);
        return idet::Status::Invalid("verify_all_threads_affinity_subset: CPU_ALLOC failed");
    }

    while (auto* e = readdir(d)) {
        if (!e->d_name[0] || e->d_name[0] == '.') continue;

        char* endp = nullptr;
        const long tid = std::strtol(e->d_name, &endp, 10);
        if (!endp || *endp != '\0' || tid <= 0) continue;

        CPU_ZERO_S(setsize, cur);
        if (sched_getaffinity(static_cast<pid_t>(tid), setsize, cur) != 0) {
            if (errno == ESRCH) continue; // thread died
            CPU_FREE(cur);
            closedir(d);
            std::ostringstream oss;
            oss << "verify_all_threads_affinity_subset: sched_getaffinity(tid=" << tid
                << ") failed: " << std::strerror(errno);
            return idet::Status::Invalid(oss.str());
        }

        for (int c = 0; c < max_cpu; ++c) {
            if (CPU_ISSET_S(c, setsize, cur) && !A.count(c)) {
                CPU_FREE(cur);
                closedir(d);
                std::ostringstream oss;
                oss << "verify_all_threads_affinity_subset: tid=" << tid << " has CPU " << c << " outside allowed set";
                return idet::Status::Invalid(oss.str());
            }
        }
        ++checked;
    }

    CPU_FREE(cur);
    #else
    if (max_id >= CPU_SETSIZE) {
        closedir(d);
        std::ostringstream oss;
        oss << "verify_all_threads_affinity_subset: CPU id " << max_id << " exceeds CPU_SETSIZE=" << CPU_SETSIZE
            << " without CPU_ALLOC";
        return idet::Status::Invalid(oss.str());
    }

    cpu_set_t cur;
    while (auto* e = readdir(d)) {
        if (!e->d_name[0] || e->d_name[0] == '.') continue;

        char* endp = nullptr;
        const long tid = std::strtol(e->d_name, &endp, 10);
        if (!endp || *endp != '\0' || tid <= 0) continue;

        CPU_ZERO(&cur);
        if (sched_getaffinity(static_cast<pid_t>(tid), sizeof(cur), &cur) != 0) {
            if (errno == ESRCH) continue;
            closedir(d);
            std::ostringstream oss;
            oss << "verify_all_threads_affinity_subset: sched_getaffinity(tid=" << tid
                << ") failed: " << std::strerror(errno);
            return idet::Status::Invalid(oss.str());
        }

        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &cur) && !A.count(c)) {
                closedir(d);
                std::ostringstream oss;
                oss << "verify_all_threads_affinity_subset: tid=" << tid << " has CPU " << c << " outside allowed set";
                return idet::Status::Invalid(oss.str());
            }
        }
        ++checked;
    }
    #endif

    closedir(d);

    if (verbose) {
        std::cout << "[verify_affinity] OK. checked_threads=" << checked << " allowed_cpus=["
                  << format_id_list(allowed_cpus) << "] (" << allowed_cpus.size() << ")\n"
                  << std::flush;
    }

    return idet::Status::Ok();
#endif
}

idet::Status verify_all_threads_affinity_subset(bool verbose) {
#if !defined(__linux__)
    (void)verbose;
    return idet::Status::Ok();
#else
    const auto topo = detect_topology();
    const auto allowed = !topo.available_cpu_ids.empty() ? topo.available_cpu_ids : topo.all_cpu_ids;
    return verify_all_threads_affinity_subset(allowed, verbose);
#endif
}

idet::Status verify_buffer_pages_on_nodes(void* base, std::size_t bytes, const std::vector<int>& allowed_nodes,
                                          double min_ratio, bool verbose) {
#if !defined(__linux__)
    (void)base;
    (void)bytes;
    (void)allowed_nodes;
    (void)min_ratio;
    (void)verbose;
    return idet::Status::Ok();
#else
    if (!base || bytes == 0) return idet::Status::Invalid("verify_buffer_pages_on_nodes: null/empty buffer");
    if (allowed_nodes.empty()) return idet::Status::Invalid("verify_buffer_pages_on_nodes: empty allowed_nodes");
    if (min_ratio < 0.0 || min_ratio > 1.0)
        return idet::Status::Invalid("verify_buffer_pages_on_nodes: min_ratio must be in [0;1]");

    const std::set<int> AN(allowed_nodes.begin(), allowed_nodes.end());

    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) page = 4096;

    const std::size_t page_sz = static_cast<std::size_t>(page);
    const std::size_t n_pages = (bytes + page_sz - 1) / page_sz;
    if (n_pages == 0) return idet::Status::Invalid("verify_buffer_pages_on_nodes: buffer smaller than a page");

    // Sample at most 4096 pages to keep overhead bounded for large buffers.
    const std::size_t max_samples = 4096;
    const std::size_t stride = (n_pages > max_samples) ? (n_pages / max_samples) : 1;

    std::vector<void*> addrs;
    addrs.reserve((n_pages + stride - 1) / stride);
    for (std::size_t i = 0; i < n_pages; i += stride) {
        addrs.push_back(static_cast<char*>(base) + i * page_sz);
    }

    std::vector<int> status(addrs.size(), -1);

    // Query page locations (do not move): nodes=nullptr, flags=0.
    if (::move_pages(0, static_cast<unsigned long>(addrs.size()), addrs.data(), nullptr, status.data(), 0) != 0) {
        std::ostringstream oss;
        oss << "verify_buffer_pages_on_nodes: move_pages(query) failed: " << std::strerror(errno) << " (errno=" << errno
            << ")";
        return idet::Status::Invalid(oss.str());
    }

    // Counters. Note: negative values are per-page failures (negative errno codes).
    std::unordered_map<int, std::size_t> cnt; // node_id or negative errno -> samples
    std::size_t in_allowed = 0;
    std::size_t out_allowed = 0;
    std::size_t neg = 0;
    std::size_t valid = 0;

    // Dominant node among valid samples (>=0).
    int dominant_node = -1;
    std::size_t dominant_cnt = 0;

    for (int st : status) {
        std::size_t& c = cnt[st];
        c++;
        if (st >= 0) {
            valid++;
            if (AN.count(st))
                in_allowed++;
            else
                out_allowed++;

            // Track dominant valid node.
            if (c > dominant_cnt) {
                dominant_cnt = c;
                dominant_node = st;
            }
        } else {
            neg++;
        }
    }

    const std::size_t samples = status.size();

    /**
     * The locality ratio is computed over all samples (including negative statuses). This makes the
     * check conservative in the presence of per-page failures.
     */
    const double ratio = (samples == 0) ? 0.0 : (static_cast<double>(in_allowed) / static_cast<double>(samples));
    const double dominant_ratio = (valid == 0) ? 0.0 : (static_cast<double>(dominant_cnt) / static_cast<double>(valid));

    auto explain_neg = [](int st) -> const char* {
        // move_pages per-page failures are negative errno codes.
        switch (-st) {
        case EACCES:
            return "EACCES (no permission / restricted mapping)";
        case EFAULT:
            return "EFAULT (bad address)";
        case EINVAL:
            return "EINVAL (invalid addr or flags)";
        case ENODEV:
            return "ENODEV (node not online / not supported)";
        case ENOENT:
            return "ENOENT (page not present)";
        case EPERM:
            return "EPERM (permission)";
        default:
            return "neg_errno (per-page failure)";
        }
    };

    if (verbose) {
        constexpr std::size_t top_n = 4; // must be > 0

        std::cout << "[verify_pages]\n"
                  << "  buffer         : base=" << base << " bytes=" << bytes << "\n"
                  << "  paging         : page_sz=" << page_sz << " total_pages=" << n_pages << "\n"
                  << "  sampling       : max_samples=" << max_samples << " stride=" << stride << " samples=" << samples
                  << "\n"
                  << "  allowed_nodes  : [" << format_id_list(allowed_nodes) << "] (" << allowed_nodes.size() << ")\n"
                  << "  summary        : valid=" << valid << " in_allowed=" << in_allowed
                  << " out_allowed=" << out_allowed << " neg=" << neg << " ratio=" << ratio
                  << " min_ratio=" << min_ratio << "\n";

        if (dominant_node >= 0) {
            std::cout << "  selected_node  : node=" << dominant_node << " cnt=" << dominant_cnt
                      << " share_valid=" << dominant_ratio << "\n";
        } else {
            std::cout << "  selected_node  : <none> (no valid pages)\n";
        }

        // Build separate lists: nodes (>=0) and negative statuses (<0).
        std::vector<std::pair<int, std::size_t>> nodes;
        std::vector<std::pair<int, std::size_t>> negs;
        nodes.reserve(cnt.size());
        negs.reserve(cnt.size());

        for (const auto& kv : cnt) {
            if (kv.first >= 0)
                nodes.push_back(kv);
            else
                negs.push_back(kv);
        }

        // Sort by count desc, then id asc.
        auto by_cnt_desc = [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        };
        std::sort(nodes.begin(), nodes.end(), by_cnt_desc);
        std::sort(negs.begin(), negs.end(), by_cnt_desc);

        // Print TOP-N nodes.
        const std::size_t show_nodes = std::min(top_n, nodes.size());
        std::size_t shown_sum = 0;

        std::cout << "  top_nodes      : (top " << show_nodes << " of " << nodes.size() << ")\n";
        for (std::size_t i = 0; i < show_nodes; ++i) {
            const int node = nodes[i].first;
            const std::size_t c = nodes[i].second;
            shown_sum += c;

            const bool allowed = (AN.count(node) != 0);
            const double share_samples = (samples == 0) ? 0.0 : (double)c / (double)samples;
            const double share_valid = (valid == 0) ? 0.0 : (double)c / (double)valid;

            std::cout << "    - node " << node << " : " << c << " share_samples=" << share_samples
                      << " share_valid=" << share_valid << (allowed ? " (allowed)" : " (NOT allowed)")
                      << (node == dominant_node ? " [dominant]" : "") << "\n";
        }

        if (nodes.size() > show_nodes) {
            const std::size_t others_nodes = nodes.size() - show_nodes;
            const std::size_t others_cnt = (valid >= shown_sum) ? (valid - shown_sum) : 0;
            const double others_share = (valid == 0) ? 0.0 : (double)others_cnt / (double)valid;

            std::cout << "    - others (" << others_nodes << " nodes) : " << others_cnt
                      << " share_valid=" << others_share << "\n";
        }

        // Print TOP-N negative statuses.
        if (!negs.empty()) {
            const std::size_t show_negs = std::min(top_n, negs.size());
            std::size_t shown_negs_sum = 0;

            std::cout << "  neg_statuses   : (top " << show_negs << " of " << negs.size() << ")\n";
            for (std::size_t i = 0; i < show_negs; ++i) {
                const int st = negs[i].first;
                const std::size_t c = negs[i].second;
                shown_negs_sum += c;

                std::cout << "    - status " << st << " : " << c << " (" << explain_neg(st) << ")\n";
            }

            if (negs.size() > show_negs) {
                const std::size_t others = (neg >= shown_negs_sum) ? (neg - shown_negs_sum) : 0;
                std::cout << "    - other neg statuses : " << others << "\n";
            }
        }
        std::cout << std::flush;
    }

    if (ratio < min_ratio) {
        std::ostringstream oss;
        oss << "verify_buffer_pages_on_nodes: locality ratio " << ratio << " < min_ratio " << min_ratio
            << " (in_allowed=" << in_allowed << ", out_allowed=" << out_allowed << ", neg=" << neg
            << ", samples=" << samples << ", selected_node=" << dominant_node
            << ", selected_share_valid=" << dominant_ratio << ")";
        return idet::Status::Invalid(oss.str());
    }

    return idet::Status::Ok();
#endif
}

idet::Status verify_buffer_pages_on_nodes(double min_ratio, bool verbose, std::size_t bytes) {
#if !defined(__linux__)
    (void)min_ratio;
    (void)verbose;
    (void)bytes;
    return idet::Status::Ok();
#else
    std::vector<unsigned char> buf(bytes);
    // First-touch to materialize pages and establish a NUMA placement.
    std::fill(buf.begin(), buf.end(), static_cast<unsigned char>(1));

    const auto allowed_nodes = linux_allowed_mems_nodes();
    if (allowed_nodes.empty()) {
        return idet::Status::Invalid("verify_buffer_pages_on_nodes: Mems_allowed_list is empty/unavailable");
    }

    return verify_buffer_pages_on_nodes(buf.data(), buf.size(), allowed_nodes, min_ratio, verbose);
#endif
}

} // namespace idet::platform
