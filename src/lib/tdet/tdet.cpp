#include "tdet.h"

#include "cli.h"
#include "cross_topology.h"
#include "drawing.h"
#include "face_detector.h"
#include "nms.h"
#include "omp_config.h"
#include "run_bench.h"
#include "run_face.h"
#include "run_single.h"
#include "tiling.h"

#include <iostream>

namespace tdet {

static int extract_tile_threads(const DetectorConfig& base) {
    if (base.kind == DetectorKind::Text) {
        const auto* t = static_cast<const TextDetectorConfig*>(&base);
        return t->threads.tile_omp_threads;
    }
    return 1;
}

static bool bind_cpus_and_mem(DetectorConfig& cfg) {
    // Collect CPU(s) info
    Topology topo = detect_topology();
    if (cfg.output.verbose) print_topology(topo);

    unsigned all_threads = static_cast<unsigned>(cfg.threads.ort_intra_threads) *
                           static_cast<unsigned>(cfg.threads.ort_inter_threads) *
                           static_cast<unsigned>(extract_tile_threads(cfg));

    // Bind cpu_nodes and memory
    std::string err;
    if (!bind_for_threads(topo, all_threads, cfg.output.verbose, /*soft_memory_bind=*/false, &err)) {
        std::cerr << "[ERROR] bind_for_threads(" << all_threads << ") failed: " << err << "\n";
        return false;
    } else if (!err.empty()) {
        std::cerr << "[WARN] bind_for_threads(" << all_threads << ") warned: " << err << "\n";
    }
    return true;
}

bool InitEnvironment(DetectorConfig& cfg) {
    // Bind cpu_nodes / memory
    if (!bind_cpus_and_mem(cfg)) return false;

    // Setup OpenMP (only meaningful for text detector; fallback to defaults)
    if (cfg.kind == DetectorKind::Text) {
        auto* t = static_cast<TextDetectorConfig*>(&cfg);
        configure_openmp_affinity(t->threads.omp_places_cli, t->threads.omp_bind_cli, t->threads.tile_omp_threads,
                                  t->output.verbose);
    }
    return true;
}

bool RunDetection(DetectorConfig& cfg) {
    try {
        if (cfg.kind == DetectorKind::Text) {
            auto* t = static_cast<TextDetectorConfig*>(&cfg);
            if (t->bench.bench_iters > 0)
                return run_bench(*t);
            else
                return run_single(*t);
        } else if (cfg.kind == DetectorKind::Face) {
            auto* f = static_cast<FaceDetectorConfig*>(&cfg);
            if (f->bench.bench_iters > 0)
                return run_face_bench(*f);
            else
                return run_face_single(*f);
        }
        std::cerr << "[ERROR] Unsupported detector config type\n";
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Fatal error: " << e.what() << "\n";
        return false;
    }
}

bool ParseArgs(int argc, char** argv, std::unique_ptr<DetectorConfig>& cfg_out) {
    return parse_arguments(argc, argv, cfg_out);
}

void PrintUsage(const char* app) {
    print_usage(app);
}

} // namespace tdet
