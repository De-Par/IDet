#include "run/run_common.h"

#include "drawing.h"
#include "geometry.h"
#include "nms.h"
#include "tiling.h"
#include "timer.h"
#include "internal/progress_bar.h"

#include <algorithm>
#include <iostream>
#include <numeric>

using tdet::DetectorConfig;
using tdet::DetectorKind;

namespace {

void adjust_fixed_dims(DetectorConfig& cfg, const cv::Mat& img, const GridSpec& g, bool use_tiles) {
    if (use_tiles) {
        if (cfg.infer.fixed_W <= 0 || cfg.infer.fixed_H <= 0) {
            int tileW = (img.cols + g.cols - 1) / g.cols;
            int tileH = (img.rows + g.rows - 1) / g.rows;
            int limit = (cfg.infer.limit_side_len > 0) ? cfg.infer.limit_side_len : 640;
            auto [fw, fh] = aspect_fit32(tileW, tileH, limit);
            cfg.infer.fixed_W = fw;
            cfg.infer.fixed_H = fh;
        }
    } else {
        if (cfg.tiling.bind_io) {
            if (cfg.infer.fixed_W <= 0 || cfg.infer.fixed_H <= 0) {
                int limit = (cfg.infer.limit_side_len > 0) ? cfg.infer.limit_side_len : std::max(img.cols, img.rows);
                auto [fw, fh] = aspect_fit32(img.cols, img.rows, limit);
                cfg.infer.fixed_W = fw;
                cfg.infer.fixed_H = fh;
            }
        } else {
            cfg.infer.fixed_W = cfg.infer.fixed_H = 0;
        }
    }
}

std::vector<Detection> run_infer(IDetector& det, const DetectorConfig& cfg, const cv::Mat& img, const GridSpec& g,
                                 bool use_tiles, double* ms_out) {
    double ms = 0.0;
    std::vector<Detection> dets;

    if (use_tiles) {
        dets = infer_tiled_generic(img, det, g, cfg.tiling.overlap, &ms, cfg.threads.tile_omp_threads,
                                   cfg.tiling.bind_io != 0, cfg.infer.fixed_W, cfg.infer.fixed_H);
    } else if (cfg.tiling.bind_io && det.supports_binding() && cfg.infer.fixed_W > 0 && cfg.infer.fixed_H > 0) {
        det.prepare_binding(cfg.infer.fixed_W, cfg.infer.fixed_H, 1);
        dets = det.detect_bound(img, 0, &ms);
    } else {
        dets = det.detect(img, &ms);
    }

    if (ms_out) *ms_out = ms;
    return dets;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    size_t k = (size_t)((p / 100.0) * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

double mean_of(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
}

} // namespace

bool run_detector_single(
    DetectorConfig& cfg,
    const std::function<std::unique_ptr<IDetector>(const DetectorConfig&)>& make_detector) {
    cv::Mat img = cv::imread(cfg.paths.image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << cfg.paths.image_path << "\n";
        return false;
    }

    GridSpec g{1, 1};
    bool use_tiles = parse_tiles(cfg.tiling.grid, g);

    adjust_fixed_dims(cfg, img, g, use_tiles);

    auto det = make_detector(cfg);
    if (!det) {
        std::cerr << "[ERROR] Detector factory returned null\n";
        return false;
    }

    double ms_infer = 0.0;
    auto dets = run_infer(*det, cfg, img, g, use_tiles, &ms_infer);

    Timer t;
    t.tic();
    dets = nms_poly(dets, cfg.output.nms_iou);
    double ms_nms = t.toc_ms();

    std::cout << "Time: " << ms_infer;
    if (use_tiles) std::cout << " + " << ms_nms << " ms (infer + nms)";
    else std::cout << " ms";
    std::cout << ", dets=" << dets.size() << "\n";

    if (cfg.output.is_draw) {
        cv::Mat vis = img.clone();
        draw_and_dump(vis, dets, g.cols, g.rows, /*is_draw=*/true, /*is_dump=*/false);
        cv::imwrite(cfg.paths.out_path, vis);
    } else {
        cv::Mat dummy = img;
        draw_and_dump(dummy, dets, g.cols, g.rows, /*is_draw=*/true, /*is_dump=*/false);
    }
    return true;
}

bool run_detector_bench(
    DetectorConfig& cfg,
    const std::function<std::unique_ptr<IDetector>(const DetectorConfig&)>& make_detector) {
    cv::Mat img = cv::imread(cfg.paths.image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << cfg.paths.image_path << "\n";
        return false;
    }

    GridSpec g{1, 1};
    bool use_tiles = parse_tiles(cfg.tiling.grid, g);
    adjust_fixed_dims(cfg, img, g, use_tiles);

    auto det = make_detector(cfg);
    if (!det) {
        std::cerr << "[ERROR] Detector factory returned null\n";
        return false;
    }

    // Warmup
    const int warm_n = std::max(1, cfg.bench.warmup);
    util::ProgressBar bar;
    bar.setup(warm_n, "Warmup: ", util::Color::yellow, 50);

    for (int i = 0; i < warm_n; ++i) {
        bar.tick();
        double ms = 0.0;
        auto d = run_infer(*det, cfg, img, g, use_tiles, &ms);
        (void)d;
    }
    bar.done();

    const int iters = std::max(3, cfg.bench.bench_iters);
    std::vector<double> infer_times;
    std::vector<double> nms_times;
    infer_times.reserve((size_t)iters);
    nms_times.reserve((size_t)iters);
    std::vector<Detection> dets;

    bar.setup(iters, "Bench:  ", util::Color::green, 50);

    for (int i = 0; i < iters; ++i) {
        bar.tick();
        double ms_infer = 0.0;
        dets = run_infer(*det, cfg, img, g, use_tiles, &ms_infer);
        Timer t;
        t.tic();
        dets = nms_poly(dets, cfg.output.nms_iou);
        double ms_nms = t.toc_ms();

        infer_times.push_back(ms_infer);
        nms_times.push_back(ms_nms);
    }
    bar.done();

    double mean_infer = mean_of(infer_times);
    double mean_nms = mean_of(nms_times);
    std::cout << "[BENCH]"
              << " | dets=" << dets.size()
              << " | mean_infer=" << mean_infer << " ms"
              << " | mean_nms=" << mean_nms << " ms"
              << " | p50=" << percentile(infer_times, 50) << " ms"
              << " | p90=" << percentile(infer_times, 90) << " ms"
              << " | p95=" << percentile(infer_times, 95) << " ms"
              << " | p99=" << percentile(infer_times, 99) << " ms"
              << " | min=" << *std::min_element(infer_times.begin(), infer_times.end()) << " ms"
              << " | max=" << *std::max_element(infer_times.begin(), infer_times.end()) << " ms\n";

    if (cfg.output.is_draw) {
        cv::Mat vis = img.clone();
        draw_and_dump(vis, dets, g.cols, g.rows, /*is_draw=*/true, /*is_dump=*/false);
        cv::imwrite(cfg.paths.out_path, vis);
    }
    return true;
}
