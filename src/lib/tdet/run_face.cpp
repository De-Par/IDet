#include "run_face.h"

#include "drawing.h"
#include "face_detector.h"
#include "nms.h"
#include "tiling.h"
#include "timer.h"

#include <algorithm>
#include <iostream>
#include <numeric>

bool run_face_single(const tdet::FaceDetectorConfig& opt) {
    cv::Mat img = cv::imread(opt.paths.image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << opt.paths.image_path << "\n";
        return false;
    }

    GridSpec g{1, 1};
    bool use_tiles = parse_tiles(opt.tiling.grid, g);

    FaceDetector det(opt);

    double ms = 0.0;
    std::vector<Detection> dets;
    if (use_tiles) {
        dets = infer_tiled_face(img, det, g, opt.tiling.overlap, &ms, opt.threads.tile_omp_threads,
                                opt.tiling.bind_io != 0, opt.infer.fixed_W, opt.infer.fixed_H);
    } else {
        dets = det.detect(img, &ms);
    }

    dets = nms_poly(dets, opt.output.nms_iou);
    std::cout << "Face detect time: " << ms << " ms, dets=" << dets.size() << "\n";

    if (opt.output.is_draw) {
        cv::Mat vis = img.clone();
        draw_and_dump(vis, dets, g.cols, g.rows, /*is_draw=*/true, /*is_dump=*/false);
        cv::imwrite(opt.paths.out_path, vis);
    }

    return true;
}

bool run_face_bench(const tdet::FaceDetectorConfig& opt) {
    cv::Mat img = cv::imread(opt.paths.image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << opt.paths.image_path << "\n";
        return false;
    }

    GridSpec g{1, 1};
    bool use_tiles = parse_tiles(opt.tiling.grid, g);

    FaceDetector det(opt);

    auto run_once = [&](double& ms_out) -> std::vector<Detection> {
        if (use_tiles)
            return infer_tiled_face(img, det, g, opt.tiling.overlap, &ms_out, opt.threads.tile_omp_threads,
                                    opt.tiling.bind_io != 0, opt.infer.fixed_W, opt.infer.fixed_H);
        return det.detect(img, &ms_out);
    };

    // Warmup
    const int warm_n = std::max(1, opt.bench.warmup);
    for (int i = 0; i < warm_n; ++i) {
        double ms = 0.0;
        auto dets = run_once(ms);
        (void)dets;
    }

    const int iters = std::max(3, opt.bench.bench_iters);
    std::vector<double> times;
    times.reserve((size_t)iters);
    std::vector<Detection> dets;
    for (int i = 0; i < iters; ++i) {
        double ms = 0.0;
        dets = run_once(ms);
        dets = nms_poly(dets, opt.output.nms_iou);
        times.push_back(ms);
    }

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = times.empty() ? 0.0 : sum / times.size();
    auto pct = [&](double p) {
        if (times.empty()) return 0.0;
        std::vector<double> t = times;
        size_t k = (size_t)((p / 100.0) * (t.size() - 1));
        std::nth_element(t.begin(), t.begin() + k, t.end());
        return t[k];
    };

    std::cout << "[FACE BENCH]"
              << " | dets=" << dets.size()
              << " | mean=" << mean << " ms"
              << " | p50=" << pct(50) << " ms"
              << " | p90=" << pct(90) << " ms"
              << " | p95=" << pct(95) << " ms"
              << " | p99=" << pct(99) << " ms"
              << " | min=" << *std::min_element(times.begin(), times.end()) << " ms"
              << " | max=" << *std::max_element(times.begin(), times.end()) << " ms\n";

    if (opt.output.is_draw) {
        cv::Mat vis = img.clone();
        draw_and_dump(vis, dets, g.cols, g.rows, /*is_draw=*/true, /*is_dump=*/false);
        cv::imwrite(opt.paths.out_path, vis);
    }
    return true;
}
