#include "run_bench.h"

#include "dbnet.h"
#include "drawing.h"
#include "geometry.h"
#include "nms.h"
#include "opencv_headers.h"
#include "progress_bar.h"
#include "tiling.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#if defined(_OPENMP)
#include <omp.h>
#endif

// ------------------------------ Helpers ------------------------------ //

static inline double mean_of(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    const double s = std::accumulate(v.begin(), v.end(), 0.0);
    return s / double(v.size());
}

static inline double stdev_of(const std::vector<double>& v, const double mean) {
    if (v.size() < 2) return 0.0;
    long double acc = 0.0L;
    for (double x : v) {
        const long double d = (long double)x - (long double)mean;
        acc += d * d;
    }
    return std::sqrt(double(acc / (v.size() - 1)));
}

static inline double percentile_of(std::vector<double> v, const double p) {
    if (v.empty()) return 0.0;
    if (p <= 0.0) return *std::min_element(v.begin(), v.end());
    if (p >= 100.0) return *std::max_element(v.begin(), v.end());
    const double pos = (p / 100.0) * double(v.size() - 1);
    const size_t k = size_t(std::floor(pos));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

static inline std::string wh_str(const int W, const int H) {
    return (W > 0 && H > 0) ? (std::to_string(W) + "x" + std::to_string(H)) : std::string("auto");
}

// ------------------------------ Bench ------------------------------ //

bool run_bench(const tdet::Options& opt) {
    // Turn off OpenCV threading - manage ourselves
    cv::setNumThreads(1);
    cv::setUseOptimized(true);

    cv::Mat img = cv::imread(opt.image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[ERROR] Cannot read image: " << opt.image_path << "\n";
        return false;
    }

    GridSpec g{1, 1};
    bool use_tiles = parse_tiles(opt.tiles_arg, g);

    /*
        OMP_threads — outside OpenMPthreads : takes effect on tiles, pack of images, postprocessing
            Use --tile_omp / OMP_NUM_THREADS with (OMP_PLACES, OMP_PROC_BIND) to manipulate
    */
    int omp_threads = opt.tile_omp_threads;

    /*
        ORT_intra_threads — internal pull of ONNX Runtime for graph operations (Conv/MatMul etc.)
            inside single node Use SessionOptions.SetIntraOpNumThreads(N) / --threads_intra to
            manipulate

        ORT intra-op threads:
            - with tiling (OpenMP outside) => ORT=1 to avoid nested parallelism
            - without tiling => ORT = user (--threads_intra) or auto(0), ORT_inter=OMP_threads=1
    */
    int intra_threads = opt.ort_intra_threads;

    /*
        ORT_inter_threads — parallelism between nodes of graph. For DBNet type detectors usually not
            affect (use inter_threads=1)
    */
    int inter_threads = opt.ort_inter_threads;

    // Setup detector
    DBNet det(opt.model_path, intra_threads, inter_threads, opt.verbose);
    det.bin_thresh = opt.bin_thresh;
    det.box_thresh = opt.box_thresh;
    det.unclip_ratio = opt.unclip;
    det.limit_side_len = opt.side;
    det.apply_sigmoid = (opt.apply_sigmoid != 0);
    det.min_text_size = opt.min_text_size;

    // Prepare per-mode sizing & binding
    int tileW = 0, tileH = 0;
    if (use_tiles) {
        tileW = (img.cols + g.cols - 1) / g.cols;
        tileH = (img.rows + g.rows - 1) / g.rows;

        // For tiling we keep a fixed per-tile input (performance & stable output sizes).
        if (opt.fixedW > 0 && opt.fixedH > 0) {
            det.fixed_W = opt.fixedW;
            det.fixed_H = opt.fixedH;
        } else {
            auto [fw, fh] = aspect_fit32(tileW, tileH, opt.side);
            det.fixed_W = fw;
            det.fixed_H = fh;
        }

        if (opt.bind_io) det.ensure_pool_size(std::max(1, omp_threads));
    } else {
        if (opt.bind_io) {
            // bind_io => fixed shape
            if (opt.fixedW > 0 && opt.fixedH > 0) {
                det.fixed_W = opt.fixedW;
                det.fixed_H = opt.fixedH;
            } else {
                auto [fw, fh] = aspect_fit32(img.cols, img.rows, opt.side);
                det.fixed_W = fw;
                det.fixed_H = fh;
            }
            det.ensure_pool_size(1);
        } else {
            // unbound + non-tiled => dynamic (best recall)
            det.fixed_W = 0;
            det.fixed_H = 0;
        }
    }

    // Config banner
    std::cerr << std::fixed << std::setprecision(3);
    std::cerr << "[BENCH][CONFIG]"
              << " | image_wh=" << img.cols << "x" << img.rows << " | grid_wh=" << g.cols << "x" << g.rows
              << " | tiles=" << (use_tiles ? "on" : "off")
              << " | tile_wh=" << (use_tiles ? wh_str(tileW, tileH) : "none")
              << " | in_wh=" << wh_str(det.fixed_W, det.fixed_H) << " | overlap=" << opt.tile_overlap
              << " | intra_th=" << intra_threads << " | inter_th=" << inter_threads << " | omp_th=" << omp_threads
              << " | bind_io=" << (opt.bind_io ? 1 : 0) << " | sigmoid=" << (det.apply_sigmoid ? 1 : 0) << "\n";

    // Reserve vec for detections
    std::vector<Detection> dets;

    // Init progress bar
    util::ProgressBar bar;

    // Warmup
    const int warm_n = std::max(1, opt.warmup);
    std::vector<double> warm;
    warm.reserve((size_t)warm_n);
    std::vector<double> warm_nms_times;
    warm_nms_times.reserve((size_t)warm_n);

    bar.setup(warm_n, "Warmup: ", util::Color::yellow, 50);

    for (int i = 0; i < warm_n; ++i) {
        bar.tick();
        double ms_infer = 0.0;

        if (!use_tiles) {
            dets = opt.bind_io ? det.infer_bound(img, 0, &ms_infer) : det.infer_unbound(img, &ms_infer);
        } else {
            dets = opt.bind_io ? infer_tiled_bound(img, det, g, opt.tile_overlap, &ms_infer, omp_threads)
                               : infer_tiled_unbound(img, det, g, opt.tile_overlap, &ms_infer, omp_threads);
        }

        Timer T;
        T.tic();
        dets = nms_poly(dets, opt.nms_iou);
        warm_nms_times.push_back(T.toc_ms());

        warm.push_back(ms_infer);
    }
    bar.done();

#if 0
    if (!warm.empty()) {
        // Stats
        double w_mean = mean_of(warm);
        double w_p50 = percentile_of(warm, 50.0);
        double w_p90 = percentile_of(warm, 90.0);
        double w_p95 = percentile_of(warm, 95.0);
        double w_p99 = percentile_of(warm, 99.0);
        double w_min = *std::min_element(warm.begin(), warm.end());
        double w_max = *std::max_element(warm.begin(), warm.end());
        double w_std = stdev_of(warm, w_mean);

        // Postprocessing
        double avg_nms = mean_of(warm_nms_times);

        std::cerr << "[BENCH][WARMUP]"
                  << " | dets=" << dets.size() << " | mean=" << w_mean << "ms"
                  << " | p50=" << w_p50 << "ms"
                  << " | p90=" << w_p90 << "ms"
                  << " | p95=" << w_p95 << "ms"
                  << " | p99=" << w_p99 << "ms"
                  << " | min=" << w_min << "ms"
                  << " | max=" << w_max << "ms"
                  << " | std=" << w_std << "ms"
                  << " | avg_fps=" << (w_mean > 0.0 ? 1000.0 / w_mean : 0.0) << " | avg_nms=" << avg_nms << "ms\n";
    }
#endif

    // Measure
    const int iters = std::max(3, opt.bench_iters);
    std::vector<double> infer_times;
    infer_times.reserve((size_t)iters);
    std::vector<double> nms_times;
    nms_times.reserve((size_t)iters);

    bar.setup(iters, "Bench:  ", util::Color::green, 50);

    for (int i = 0; i < iters; ++i) {
        bar.tick();
        double ms_infer = 0.0;

        if (!use_tiles) {
            dets = opt.bind_io ? det.infer_bound(img, 0, &ms_infer) : det.infer_unbound(img, &ms_infer);
        } else {
            dets = opt.bind_io ? infer_tiled_bound(img, det, g, opt.tile_overlap, &ms_infer, omp_threads)
                               : infer_tiled_unbound(img, det, g, opt.tile_overlap, &ms_infer, omp_threads);
        }

        Timer T;
        T.tic();
        dets = nms_poly(dets, opt.nms_iou);
        nms_times.push_back(T.toc_ms());

        infer_times.push_back(ms_infer);
    }

    bar.done();

    // Stats
    double avg = mean_of(infer_times);
    double p50 = percentile_of(infer_times, 50.0);
    double p90 = percentile_of(infer_times, 90.0);
    double p95 = percentile_of(infer_times, 95.0);
    double p99 = percentile_of(infer_times, 99.0);
    double tmin = *std::min_element(infer_times.begin(), infer_times.end());
    double tmax = *std::max_element(infer_times.begin(), infer_times.end());
    double stdv = stdev_of(infer_times, avg);

    // Postprocessing
    double avg_nms = mean_of(nms_times);

    std::cerr << "[BENCH][ INFR ]"
              << " | dets=" << dets.size() << " | mean=" << avg << "ms"
              << " | p50=" << p50 << "ms"
              << " | p90=" << p90 << "ms"
              << " | p95=" << p95 << "ms"
              << " | p99=" << p99 << "ms"
              << " | min=" << tmin << "ms"
              << " | max=" << tmax << "ms"
              << " | std=" << stdv << "ms"
              << " | avg_fps=" << (avg > 0.0 ? 1000.0 / avg : 0.0) << " | avg_nms=" << avg_nms << "ms\n\n";

    if (opt.is_draw) {
        cv::Mat vis = img.clone();
        draw_and_dump(vis, dets,
                      /*cols=*/g.cols,
                      /*rows=*/g.rows,
                      /*is_draw=*/true,
                      /*is_dump=*/false);
        cv::imwrite(opt.out_path, vis);
    }

    return true;
}