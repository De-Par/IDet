#include "tdet.h"

#include "cli.h"
#include "cross_topology.h"
#include "omp_config.h"
#include "run_bench.h"
#include "run_single.h"

#include <iostream>

namespace tdet {

static bool bind_cpus_and_mem(Options& opt) {
    // Collect CPU(s) info
    Topology topo = detect_topology();
    if (opt.verbose) print_topology(topo);

    unsigned all_threads = opt.ort_intra_threads * opt.ort_inter_threads * opt.tile_omp_threads;

    // Bind cpu_nodes and memory
    std::string err;
    if (!bind_for_threads(topo, all_threads, opt.verbose, /*soft_memory_bind=*/false, &err)) {
        std::cerr << "[ERROR] bind_for_threads(" << all_threads << ") failed: " << err << "\n";
        return false;
    } else if (!err.empty()) {
        std::cerr << "[WARN] bind_for_threads(" << all_threads << ") warned: " << err << "\n";
    }
    return true;
}

bool InitEnvironment(Options& opt) {
    // Bind cpu_nodes / memory
    if (!bind_cpus_and_mem(opt)) return false;

    // Setup OpenMP
    configure_openmp_affinity(opt.omp_places_cli, opt.omp_bind_cli, opt.tile_omp_threads, opt.verbose);
    return true;
}

bool RunDetection(Options& opt) {
    try {
        if (opt.bench_iters > 0)
            return run_bench(opt);
        else
            return run_single(opt);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Fatal error: " << e.what() << "\n";
        return false;
    }
}

bool ParseArgs(int argc, char** argv, Options& opt) {
    return parse_arguments(argc, argv, opt);
}

void PrintUsage(const char* app) {
    print_usage(app);
}

} // namespace tdet