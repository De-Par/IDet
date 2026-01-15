#include "run_single.h"

#include "dbnet.h"
#include "drawing.h"
#include "geometry.h"
#include "nms.h"
#include "opencv_headers.h"
#include "tiling.h"
#include "timer.h"

#include <iostream>

#if defined(_OPENMP)
    #include <omp.h>
#endif

bool run_single(const tdet::TextDetectorConfig& opt) {
    // Turn off OpenCV threading - manage ourselves
    cv::setUseOptimized(true);
    cv::setNumThreads(1);

    cv::Mat img = cv::imread(opt.paths.image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[ERROR] Cannot read image\n";
        return false;
    }

    GridSpec g{1, 1};
    bool use_tiles = parse_tiles(opt.tiling.grid, g);

    int omp_threads = opt.threads.tile_omp_threads;

    tdet::TextDetectorConfig cfg = opt;
    if (use_tiles && cfg.threads.ort_intra_threads <= 0) cfg.threads.ort_intra_threads = 1;

    if (use_tiles) {
        int tileW = (img.cols + g.cols - 1) / g.cols;
        int tileH = (img.rows + g.rows - 1) / g.rows;
        auto [fw, fh] = aspect_fit32(tileW, tileH, (cfg.infer.limit_side_len > 0 ? cfg.infer.limit_side_len : 640));
        cfg.infer.fixed_W = fw;
        cfg.infer.fixed_H = fh;
    } else {
        if (opt.tiling.bind_io) {
            if (cfg.infer.fixed_W <= 0 || cfg.infer.fixed_H <= 0) {
                auto [fw, fh] = aspect_fit32(img.cols, img.rows, cfg.infer.limit_side_len);
                cfg.infer.fixed_W = fw;
                cfg.infer.fixed_H = fh;
            }
        } else {
            cfg.infer.fixed_W = cfg.infer.fixed_H = 0;
        }
    }

    DBNet det(opt.paths.model_path, cfg, opt.output.verbose);
    if (opt.tiling.bind_io) det.ensure_pool_size(std::max(1, omp_threads));

    std::vector<Detection> dets;
    double ms = 0.0;

    if (!use_tiles) {
        if (opt.tiling.bind_io)
            dets = det.infer_bound(img, 0, &ms);
        else
            dets = det.infer_unbound(img, &ms);

        std::cout << "Time: " << ms << " ms\n";

    } else {
        if (opt.tiling.bind_io)
            dets = infer_tiled_bound(img, det, g, opt.tiling.overlap, &ms, omp_threads);
        else
            dets = infer_tiled_unbound(img, det, g, opt.tiling.overlap, &ms, omp_threads);

        Timer T;
        T.tic();
        dets = nms_poly(dets, opt.output.nms_iou);
        double nms_time = T.toc_ms();

        std::cout << "Time: " << ms << " + " << nms_time << " ms (infer + nms)\n";
    }

    if (opt.output.is_draw) {
        cv::Mat vis = img.clone();
        draw_and_dump(vis, dets,
                      /*cols=*/g.cols,
                      /*rows=*/g.rows,
                      /*is_draw=*/true,
                      /*is_dump=*/true);
        cv::imwrite(opt.paths.out_path, vis);
    } else {
        cv::Mat dummy = img;
        draw_and_dump(dummy, dets,
                      /*cols=*/g.cols,
                      /*rows=*/g.rows,
                      /*is_draw=*/true,
                      /*is_dump=*/true);
    }

    return true;
}
