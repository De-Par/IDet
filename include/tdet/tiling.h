#pragma once

#include <vector>
#include <string>

#if defined(__APPLE__)
    #include <opencv2/core.hpp>
#else
    #include <opencv4/opencv2/core.hpp>
#endif

#include "dbnet.h"


struct GridSpec {
    int rows{1};
    int cols{1};
};

bool parse_tiles(const std::string &s, GridSpec &g);

std::vector<Detection> infer_tiled_unbound(const cv::Mat &img, DBNet &det, const GridSpec &g, const float overlap, double *ms_out, const int tile_omp_threads);

std::vector<Detection> infer_tiled_bound(const cv::Mat &img, DBNet &det, const GridSpec &g, const float overlap, double *ms_out, const int tile_omp_threads);