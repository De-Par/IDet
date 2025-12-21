#pragma once
#include <string>
#include <vector>

struct SocketInfo {
    // Socket/package id
    int socket_id = -1;

    // Logical cores in this socket
    unsigned logical_cores = 0;

    // Physical cores in this socket
    unsigned physical_cores = 0;

    // All logical CPU IDs in this socket
    std::vector<int> logical_cpu_ids;

    // Subset available to current process
    std::vector<int> available_cpu_ids;
};

struct Topology {
    // Total logical cores (OS-visible)
    unsigned total_logical = 0;

    // Total physical cores
    unsigned total_physical = 0;

    // Number of sockets
    unsigned socket_count = 0;

    // All CPU IDs (0..N-1 typically)
    std::vector<int> all_cpu_ids;

    // CPUs available to this process
    std::vector<int> available_cpu_ids;

    // Per-socket breakdown
    std::vector<SocketInfo> sockets;
};

Topology detect_topology(void);

void print_topology(const Topology& topo);

bool bind_for_threads(const Topology& topo, unsigned desired_threads, bool soft_memory_bind, bool verbose = true,
                      std::string* err = nullptr);