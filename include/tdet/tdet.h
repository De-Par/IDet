#pragma once
#include "export.h"

#include <string>

namespace tdet {

struct Options {
    std::string model_path;
    std::string image_path;
    std::string out_path = "out.png";

    float bin_thresh = 0.3f;
    float box_thresh = 0.3f;
    float unclip = 1.0f;

    int side = 960;
    int min_text_size = 3;

    int ort_intra_threads = 1;
    int ort_inter_threads = 1;
    int tile_omp_threads = 1;

    float tile_overlap = 0.1f;
    float nms_iou = 0.3f;
    int apply_sigmoid = 0;
    std::string tiles_arg;

    std::string omp_places_cli;
    std::string omp_bind_cli;

    int bind_io = 0;
    int fixedW = 0;
    int fixedH = 0;
    std::string fixed_wh;

    int bench_iters = 0;
    int warmup = 0;

    bool is_draw = false;
    bool verbose = true;
};

TDET_API bool InitEnvironment(Options& opt);

TDET_API bool RunDetection(Options& opt);

TDET_API bool ParseArgs(int argc, char** argv, Options& opt);

TDET_API void PrintUsage(const char* app);

} // namespace tdet