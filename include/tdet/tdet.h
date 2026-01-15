#pragma once
#include "export.h"

#include <memory>
#include <string>

namespace tdet {

enum class DetectorKind {
    Text,
    Face,
    Unknown
};

struct Paths {
    std::string model_path;
    std::string image_path;
    std::string out_path = "out.png";
};

struct InferenceParams {
    float bin_thresh = 0.3f;
    float box_thresh = 0.3f;
    float unclip = 1.0f;
    int limit_side_len = 960;
    bool apply_sigmoid = false;
    int fixed_W = 0;
    int fixed_H = 0;
};

struct Threading {
    int ort_intra_threads = 1;
    int ort_inter_threads = 1;
    int tile_omp_threads = 1;
    std::string omp_places_cli;
    std::string omp_bind_cli;
};

struct TilingParams {
    float overlap = 0.1f;
    std::string grid;
    int bind_io = 0;
    std::string fixed_wh;
};

struct Benchmarking {
    int bench_iters = 0;
    int warmup = 0;
};

struct OutputParams {
    float nms_iou = 0.3f;
    bool is_draw = false;
    bool verbose = true;
};

struct DetectorConfig {
    DetectorKind kind = DetectorKind::Unknown;
    Paths paths;
    InferenceParams infer;
    Threading threads;
    TilingParams tiling;
    Benchmarking bench;
    OutputParams output;

    virtual ~DetectorConfig() = default;
};

struct TextDetectorConfig : public DetectorConfig {
    int min_text_size = 3;

    TextDetectorConfig() {
        kind = DetectorKind::Text;
    }
};

struct FaceDetectorConfig : public DetectorConfig {
    int min_size_w = 10;
    int min_size_h = 10;

    FaceDetectorConfig() {
        kind = DetectorKind::Face;
        infer.apply_sigmoid = false; // SCRFD outputs are already probabilities
        infer.box_thresh = 0.6f;
        output.nms_iou = 0.4f;
    }
};

TDET_API bool InitEnvironment(DetectorConfig& cfg);

TDET_API bool RunDetection(DetectorConfig& cfg);

TDET_API bool ParseArgs(int argc, char** argv, std::unique_ptr<DetectorConfig>& cfg_out);

TDET_API void PrintUsage(const char* app);

} // namespace tdet
