#include "dbnet.h"
#include "face_detector.h"
#include "opencv_headers.h"
#include "run/run_bench.h"
#include "run/run_common.h"
#include "run/run_face.h"
#include "run/run_single.h"

bool run_single(const tdet::TextDetectorConfig& opt) {
    cv::setUseOptimized(true);
    cv::setNumThreads(1);

    auto cfg = opt;
    return run_detector_single(cfg, [](const tdet::DetectorConfig& cfg) {
        auto* tcfg = dynamic_cast<const tdet::TextDetectorConfig*>(&cfg);
        tdet::TextDetectorConfig local = tcfg ? *tcfg : tdet::TextDetectorConfig{};
        return std::make_unique<DBNet>(cfg.paths.model_path, local, cfg.output.verbose);
    });
}

bool run_bench(const tdet::TextDetectorConfig& opt) {
    cv::setUseOptimized(true);
    cv::setNumThreads(1);

    auto cfg = opt;
    return run_detector_bench(cfg, [](const tdet::DetectorConfig& cfg) {
        auto* tcfg = dynamic_cast<const tdet::TextDetectorConfig*>(&cfg);
        tdet::TextDetectorConfig local = tcfg ? *tcfg : tdet::TextDetectorConfig{};
        return std::make_unique<DBNet>(cfg.paths.model_path, local, cfg.output.verbose);
    });
}

bool run_face_single(const tdet::FaceDetectorConfig& opt) {
    cv::setUseOptimized(true);
    cv::setNumThreads(1);

    auto cfg = opt;
    return run_detector_single(cfg, [](const tdet::DetectorConfig& cfg) {
        auto* fcfg = dynamic_cast<const tdet::FaceDetectorConfig*>(&cfg);
        tdet::FaceDetectorConfig local = fcfg ? *fcfg : tdet::FaceDetectorConfig{};
        return std::make_unique<FaceDetector>(local);
    });
}

bool run_face_bench(const tdet::FaceDetectorConfig& opt) {
    cv::setUseOptimized(true);
    cv::setNumThreads(1);

    auto cfg = opt;
    return run_detector_bench(cfg, [](const tdet::DetectorConfig& cfg) {
        auto* fcfg = dynamic_cast<const tdet::FaceDetectorConfig*>(&cfg);
        tdet::FaceDetectorConfig local = fcfg ? *fcfg : tdet::FaceDetectorConfig{};
        return std::make_unique<FaceDetector>(local);
    });
}
