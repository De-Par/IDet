#pragma once
#include "dbnet.h"
#include "face_detector.h"
#include "opencv_headers.h"

#include <string>
#include <vector>

struct GridSpec {
    int rows{1};
    int cols{1};
};

bool parse_tiles(const std::string& s, GridSpec& g);

std::vector<Detection> infer_tiled_unbound(const cv::Mat& img, DBNet& det, const GridSpec& g, const float overlap,
                                           double* ms_out, const int tile_omp_threads);

std::vector<Detection> infer_tiled_bound(const cv::Mat& img, DBNet& det, const GridSpec& g, const float overlap,
                                         double* ms_out, const int tile_omp_threads);

std::vector<Detection> infer_tiled_face(const cv::Mat& img, FaceDetector& det, const GridSpec& g, const float overlap,
                                        double* ms_out, const int tile_omp_threads, bool bind_io = false,
                                        int fixed_w = 0, int fixed_h = 0);
