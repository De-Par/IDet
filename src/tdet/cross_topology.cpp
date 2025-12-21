#include "cross_topology.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#error "[ERROR] C++17 <filesystem> is required"
#endif

#if defined(__linux__)
#include <cstring>
#include <sched.h>
#include <unistd.h>

#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

// Optional NUMA (Linux + libnuma + -DUSE_LIBNUMA)
#if defined(__linux__) && defined(USE_LIBNUMA) && __has_include(<numa.h>)
#include <numa.h>
#include <numaif.h>
#define HAS_LIBNUMA 1
#else
#define HAS_LIBNUMA 0
#endif

// Parse strings like "0-3,5,7-8" into sorted unique vector
static std::vector<int> parse_cpu_list_string(const std::string& s) {
    std::vector<int> cpus;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        auto dash = item.find('-');
        if (dash == std::string::npos) {
            try {
                cpus.push_back(std::stoi(item));
            } catch (...) {
            }
        } else {
            try {
                int a = std::stoi(item.substr(0, dash));
                int b = std::stoi(item.substr(dash + 1));
                if (a <= b) {
                    for (int x = a; x <= b; ++x)
                        cpus.push_back(x);
                }
            } catch (...) {
            }
        }
    }

    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    return cpus;
}

// Format vector<int> as "0-3,5,7-8"
static std::string format_cpu_list(const std::vector<int>& cpus_raw) {
    if (cpus_raw.empty()) return "";

    std::vector<int> v = cpus_raw;
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());

    std::ostringstream oss;
    int range_start = v[0];
    int prev = v[0];

    auto flush_range = [&](bool first) {
        if (!first) oss << ",";
        if (range_start == prev) {
            oss << range_start;
        } else {
            oss << range_start << "-" << prev;
        }
    };

    bool first = true;
    for (size_t i = 1; i < v.size(); ++i) {
        int x = v[i];
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

// ----------- Linux implementation -----------

#if defined(__linux__)

static bool read_int_from_file(const fs::path& p, int& out) {
    std::ifstream f(p);
    if (!f) return false;
    std::string s;
    std::getline(f, s);
    if (s.empty()) return false;
    try {
        out = std::stoi(s);
        return true;
    } catch (...) {
        return false;
    }
}

// Online CPUs (OS-visible), using /sys/devices/system/cpu/online or directory enumeration
static std::vector<int> linux_all_cpu_ids() {
    std::string online;
    {
        std::ifstream f("/sys/devices/system/cpu/online");
        if (f) std::getline(f, online);
    }
    if (!online.empty()) return parse_cpu_list_string(online);

    std::vector<int> ids;
    const fs::path cpu_root = "/sys/devices/system/cpu";
    for (const auto& entry : fs::directory_iterator(cpu_root)) {
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("cpu", 0) != 0) continue;
        if (name.size() <= 3) continue;

        const std::string digits = name.substr(3);
        if (!std::all_of(digits.begin(), digits.end(), ::isdigit)) continue;
        try {
            ids.push_back(std::stoi(digits));
        } catch (...) {
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

// CPUs available to current process (cpuset/affinity), via /proc/self/status Cpus_allowed_list
static std::vector<int> linux_available_cpu_ids() {
    std::ifstream f("/proc/self/status");
    if (!f) return linux_all_cpu_ids();

    std::string line;
    const std::string key = "Cpus_allowed_list:";
    while (std::getline(f, line)) {
        if (line.rfind(key, 0) == 0) {
            std::string list = line.substr(key.size());
            auto pos = list.find_first_not_of(" \t");
            if (pos != std::string::npos) list = list.substr(pos);
            return parse_cpu_list_string(list);
        }
    }
    return linux_all_cpu_ids();
}

// Set process CPU affinity to given CPU list. Handles large CPU IDs when CPU_ALLOC_* is available
static bool linux_set_affinity_to_cpulist(const std::vector<int>& cpus, std::string* err = nullptr) {
    if (cpus.empty()) {
        if (err) *err = "CPU list is empty";
        return false;
    }

    int max_id = *std::max_element(cpus.begin(), cpus.end());
    if (max_id < 0) {
        if (err) *err = "CPU list contains negative IDs";
        return false;
    }

#if defined(CPU_ALLOC)
    int max_cpu = max_id + 1;
    size_t setsize = CPU_ALLOC_SIZE(max_cpu);
    cpu_set_t* set = CPU_ALLOC(max_cpu);
    if (!set) {
        if (err) *err = "CPU_ALLOC failed";
        return false;
    }
    CPU_ZERO_S(setsize, set);
    for (int c : cpus) {
        if (c < 0) continue;
        CPU_SET_S(c, setsize, set);
    }

    int rc = sched_setaffinity(0, setsize, set);
    CPU_FREE(set);

    if (rc != 0) {
        if (err) *err = std::string("sched_setaffinity failed: ") + std::strerror(errno);
        return false;
    }
    return true;
#else
    if (max_id >= CPU_SETSIZE) {
        if (err) {
            *err = "CPU id " + std::to_string(max_id) + " exceeds CPU_SETSIZE=" + std::to_string(CPU_SETSIZE) +
                   " and dynamic CPU sets are not available";
        }
        return false;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus) {
        if (c < 0) continue;
        CPU_SET(c, &set);
    }

    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        if (err) *err = std::string("sched_setaffinity failed: ") + std::strerror(errno);
        return false;
    }
    return true;
#endif
}

#if HAS_LIBNUMA
// NUMA node -> cpus mapping from sysfs
static std::map<int, std::vector<int>> linux_numa_node_to_cpus() {
    std::map<int, std::vector<int>> mapn;
    const fs::path nodes_root = "/sys/devices/system/node";
    if (!fs::exists(nodes_root)) return mapn;

    for (const auto& e : fs::directory_iterator(nodes_root)) {
        if (!e.is_directory()) continue;
        const auto name = e.path().filename().string(); // node0
        if (name.rfind("node", 0) != 0) continue;

        const std::string idstr = name.substr(4);
        if (idstr.empty() || !std::all_of(idstr.begin(), idstr.end(), ::isdigit)) continue;
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
        if (!cpus.empty()) mapn.emplace(node_id, std::move(cpus));
    }
    return mapn;
}

// NUMA nodes that contain any of the given CPUs
static std::vector<int> nodes_for_cpus(const std::vector<int>& cpus) {
    auto mapn = linux_numa_node_to_cpus();
    if (mapn.empty()) return {};

    std::set<int> S(cpus.begin(), cpus.end());
    std::vector<int> nodes;

    for (auto& [node, ncpus] : mapn) {
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
#endif // HAS_LIBNUMA

// Detect topology on Linux via sysfs + /proc/self/status
static Topology detect_linux() {
    Topology topo;
    topo.all_cpu_ids = linux_all_cpu_ids();
    topo.available_cpu_ids = linux_available_cpu_ids();

    // Aggregate per socket using /sys/devices/system/cpu/cpu*/topology
    struct Agg {
        std::set<int> core_ids;
        std::vector<int> cpus;
    };
    std::map<int, Agg> per_socket;

    for (int cpu_id : topo.all_cpu_ids) {
        fs::path topo_dir = fs::path("/sys/devices/system/cpu") / ("cpu" + std::to_string(cpu_id)) / "topology";
        int socket_id = -1;
        int core_id = -1;

        if (!read_int_from_file(topo_dir / "physical_package_id", socket_id)) {
            continue; // skip if missing
        }
        read_int_from_file(topo_dir / "core_id", core_id);

        auto& agg = per_socket[socket_id];
        agg.cpus.push_back(cpu_id);
        if (core_id >= 0) agg.core_ids.insert(core_id);
    }

    for (auto& [sid, agg] : per_socket) {
        std::sort(agg.cpus.begin(), agg.cpus.end());
        agg.cpus.erase(std::unique(agg.cpus.begin(), agg.cpus.end()), agg.cpus.end());

        SocketInfo s;
        s.socket_id = sid;
        s.logical_cores = static_cast<unsigned>(agg.cpus.size());
        s.physical_cores = agg.core_ids.empty() ? s.logical_cores : static_cast<unsigned>(agg.core_ids.size());
        s.logical_cpu_ids = agg.cpus;

        // Intersection with process-available CPUs
        std::set<int> avail_set(topo.available_cpu_ids.begin(), topo.available_cpu_ids.end());
        for (int c : s.logical_cpu_ids) {
            if (avail_set.count(c)) s.available_cpu_ids.push_back(c);
        }

        topo.total_logical += s.logical_cores;
        topo.total_physical += s.physical_cores;
        topo.sockets.push_back(std::move(s));
    }

    topo.socket_count = static_cast<unsigned>(topo.sockets.size());

    return topo;
}

#endif // __linux__

// ----------- MacOS implementation -----------

#if defined(__APPLE__)

static bool sysctl_get_uint64(const char* name, uint64_t& out) {
    size_t size = sizeof(uint64_t);
    if (sysctlbyname(name, &out, &size, nullptr, 0) == 0) return true;
    return false;
}

static unsigned sysctl_get_uint_or(const char* name, unsigned defv) {
    uint64_t tmp = 0;
    if (!sysctl_get_uint64(name, tmp)) return defv;
    return static_cast<unsigned>(tmp);
}

// Detect topology on macOS via sysctl
// Per-socket split is approximate (even distribution) because public APIs
// do not expose exact per-CPU/socket mapping without IOKit
static Topology detect_macos() {
    Topology topo;

    unsigned logical = sysctl_get_uint_or("hw.logicalcpu_max", 0);
    if (!logical) logical = sysctl_get_uint_or("hw.logicalcpu", std::thread::hardware_concurrency());

    unsigned physical = sysctl_get_uint_or("hw.physicalcpu_max", 0);
    if (!physical) physical = sysctl_get_uint_or("hw.physicalcpu", logical);

    topo.total_logical = logical ? logical : 1u;
    topo.total_physical = physical ? physical : topo.total_logical;

    topo.all_cpu_ids.resize(topo.total_logical);
    std::iota(topo.all_cpu_ids.begin(), topo.all_cpu_ids.end(), 0);

    // No public CPU affinity mask API on MacOS; assume all are available
    topo.available_cpu_ids = topo.all_cpu_ids;

    unsigned packages = sysctl_get_uint_or("hw.packages", 0);
    if (!packages) packages = 1;
    topo.socket_count = packages;

    // Evenly distribute logical and physical cores across packages
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

        for (unsigned k = 0; k < s.logical_cores && cursor < static_cast<int>(topo.total_logical); ++k) {
            s.logical_cpu_ids.push_back(cursor++);
        }
        s.available_cpu_ids = s.logical_cpu_ids;

        topo.sockets.push_back(std::move(s));
    }

    return topo;
}

#endif // __APPLE__

// ----------- Public API -----------

Topology detect_topology() {
#if defined(__linux__)
    return detect_linux();
#elif defined(__APPLE__)
    return detect_macos();
#else
    Topology t;
    unsigned hw = std::thread::hardware_concurrency();
    t.total_logical = hw ? hw : 1;
    t.total_physical = t.total_logical;
    t.socket_count = 1;

    t.all_cpu_ids.resize(t.total_logical);
    std::iota(t.all_cpu_ids.begin(), t.all_cpu_ids.end(), 0);
    t.available_cpu_ids = t.all_cpu_ids;

    SocketInfo s;
    s.socket_id = 0;
    s.logical_cores = t.total_logical;
    s.physical_cores = t.total_physical;
    s.logical_cpu_ids = t.all_cpu_ids;
    s.available_cpu_ids = s.logical_cpu_ids;
    t.sockets.push_back(std::move(s));

    return t;
#endif
}

// Select compact set of sockets/CPUs for N threads and bind CPU + NUMA policy (only Linux)
bool bind_for_threads(const Topology& topo, unsigned desired_threads, bool verbose, bool soft_memory_bind,
                      std::string* err) {
    if (desired_threads == 0) {
        if (err) *err = "desired_threads must be > 0";
        return false;
    }

    const auto& global_avail = !topo.available_cpu_ids.empty() ? topo.available_cpu_ids : topo.all_cpu_ids;

    if (global_avail.empty()) {
        if (err) *err = "no CPUs visible to process";
        return false;
    }

    if (desired_threads > global_avail.size()) {
        if (err) {
            *err = "requested " + std::to_string(desired_threads) + " threads, but only " +
                   std::to_string(global_avail.size()) + " logical CPUs are available (affinity/cpuset)";
        }
        return false;
    }

#if defined(__linux__)
    std::set<int> G(global_avail.begin(), global_avail.end());

    auto intersect_with_global = [&](const std::vector<int>& src) {
        std::vector<int> out;
        out.reserve(src.size());
        for (int c : src) {
            if (G.count(c)) out.push_back(c);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    };

    std::vector<int> chosen_cpus;
    std::vector<int> chosen_sockets;

    // 1) Try to fit all threads into a single socket
    const SocketInfo* best_single = nullptr;
    std::vector<int> best_single_avail;

    for (const auto& s : topo.sockets) {
        const auto& src = !s.available_cpu_ids.empty() ? s.available_cpu_ids : s.logical_cpu_ids;
        auto avail = intersect_with_global(src);
        if (avail.size() >= desired_threads) {
            best_single = &s;
            best_single_avail = std::move(avail);
            break; // first matching socket is enough
        }
    }

    if (best_single) {
        chosen_sockets.push_back(best_single->socket_id);
        chosen_cpus.assign(best_single_avail.begin(), best_single_avail.begin() + desired_threads);
    } else {
        // 2) Need multiple sockets: pick minimal count of sockets with most CPUs first
        struct SockAvail {
            const SocketInfo* s;
            std::vector<int> avail;
        };
        std::vector<SockAvail> cand;

        for (const auto& s : topo.sockets) {
            const auto& src = !s.available_cpu_ids.empty() ? s.available_cpu_ids : s.logical_cpu_ids;
            auto avail = intersect_with_global(src);
            if (!avail.empty()) {
                cand.push_back({&s, std::move(avail)});
            }
        }

        if (cand.empty()) {
            if (err) *err = "no CPUs available in any socket";
            return false;
        }

        std::sort(cand.begin(), cand.end(),
                  [](const SockAvail& a, const SockAvail& b) { return a.avail.size() > b.avail.size(); });

        for (auto& sa : cand) {
            chosen_sockets.push_back(sa.s->socket_id);
            for (int c : sa.avail) {
                if (chosen_cpus.size() == desired_threads) break;
                chosen_cpus.push_back(c);
            }
            if (chosen_cpus.size() == desired_threads) break;
        }

        if (chosen_cpus.size() != desired_threads) {
            if (err) *err = "internal error: could not gather requested CPUs; check topology/cpuset";
            return false;
        }
    }

    // 3) Bind CPU affinity to chosen CPUs
    if (!linux_set_affinity_to_cpulist(chosen_cpus, err)) {
        return false;
    }

#if HAS_LIBNUMA
    if (numa_available() >= 0) {
        auto nodes = nodes_for_cpus(chosen_cpus);
        if (!nodes.empty()) {
            bitmask* mask = numa_allocate_nodemask();
            numa_bitmask_clearall(mask);
            for (int n : nodes)
                numa_bitmask_setbit(mask, n);

            if (soft_memory_bind) {
                if (nodes.size() == 1) {
                    numa_set_preferred(nodes[0]);
                } else {
                    numa_set_membind(mask);
                }
            } else {
                numa_bind(mask);
            }
            numa_free_nodemask(mask);
        } else {
            numa_set_localalloc();
        }
    }
#else
    (void)soft_memory_bind;
#endif
#else
    (void)topo;
    (void)desired_threads;
    (void)soft_memory_bind;
    if (err) *err = "bind_for_threads is currently implemented only on Linux";
#endif
    // Log results of binding
    if (verbose) {
        std::set<int> chosen_set(chosen_cpus.begin(), chosen_cpus.end());
        std::cout << "\n[bind_for_threads] threads=" << desired_threads << "\n";
        std::cout << "    sockets_ids: [" << format_cpu_list(chosen_sockets) << "]\n";

        // For each socket, print which of its CPUs were selected
        for (const auto& s : topo.sockets) {
            std::vector<int> used;
            for (int c : s.logical_cpu_ids) {
                if (chosen_set.count(c)) used.push_back(c);
            }
            if (!used.empty()) {
                std::cout << "    socket_" << s.socket_id << " cpus_id: [" << format_cpu_list(used) << "]\n";
            }
        }
        std::cout << std::flush;
    }

    // Done: caller can now create N threads/OpenMP region
    return true;
}

// Pretty-printer for debugging/logging
void print_topology(const Topology& topo) {
    std::cout << "=== CPU(s) Topology ===\n";
    std::cout << "Sockets:        " << topo.socket_count << "\n";
    std::cout << "Total logical:  " << topo.total_logical << "\n";
    std::cout << "Total physical: " << topo.total_physical << "\n";
    std::cout << "All CPU IDs:       " << format_cpu_list(topo.all_cpu_ids) << "\n";
    std::cout << "Available CPU IDs: " << format_cpu_list(topo.available_cpu_ids) << "\n";

    std::cout << "\n=== Per-socket ===\n";
    for (const auto& s : topo.sockets) {
        std::cout << "Socket_id=" << s.socket_id << " | logical=" << s.logical_cores
                  << " | physical=" << s.physical_cores << "\n";
        std::cout << "    All CPU IDs:       " << format_cpu_list(s.logical_cpu_ids) << "\n";
        std::cout << "    Available CPU IDs: " << format_cpu_list(s.available_cpu_ids) << "\n";
    }
}