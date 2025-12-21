#include "run_single.h"

#include "dbnet.h"
#include "drawing.h"
#include "geometry.h"
#include "nms.h"
#include "tiling.h"
#include "timer.h"

#include <iostream>

#if defined(__APPLE__)
#include <opencv2/opencv.hpp>
#else
#include <opencv4/opencv2/opencv.hpp>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

bool run_single(const tdet::Options& opt) {
    // Turn off OpenCV threading - manage ourselves
    cv::setUseOptimized(true);
    cv::setNumThreads(1);

    cv::Mat img = cv::imread(opt.image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[ERROR] Cannot read image\n";
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
       inside single node Use SessionOptions.SetIntraOpNumThreads(N) / --threads_intra to manipulate

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

    if (use_tiles) {
        // derive per-tile fixed HW from --side
        int tileW = (img.cols + g.cols - 1) / g.cols;
        int tileH = (img.rows + g.rows - 1) / g.rows;
        auto [fw, fh] = aspect_fit32(tileW, tileH, (opt.side > 0 ? opt.side : 640));
        det.fixed_W = fw;
        det.fixed_H = fh;

        if (opt.bind_io) det.ensure_pool_size(std::max(1, omp_threads));
    } else {
        if (opt.bind_io) {
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
            det.fixed_W = det.fixed_H = 0;
        }
    }

    std::vector<Detection> dets;
    double ms = 0.0;

    if (!use_tiles) {
        if (opt.bind_io)
            dets = det.infer_bound(img, 0, &ms);
        else
            dets = det.infer_unbound(img, &ms);

        std::cout << "Time: " << ms << " ms\n";

    } else {
        if (opt.bind_io)
            dets = infer_tiled_bound(img, det, g, opt.tile_overlap, &ms, omp_threads);
        else
            dets = infer_tiled_unbound(img, det, g, opt.tile_overlap, &ms, omp_threads);

        Timer T;
        T.tic();
        dets = nms_poly(dets, opt.nms_iou);
        double nms_time = T.toc_ms();

        std::cout << "Time: " << ms << " + " << nms_time << " ms (infer + nms)\n";
    }

    if (opt.is_draw) {
        cv::Mat vis = img.clone();
        draw_and_dump(vis, dets,
                      /*cols=*/g.cols,
                      /*rows=*/g.rows,
                      /*is_draw=*/true,
                      /*is_dump=*/true);
        cv::imwrite(opt.out_path, vis);
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