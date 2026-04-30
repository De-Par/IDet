#include <idet.h>

#include <iostream>
#include <string>
#include <utility>

namespace {

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path = "assets/models/paddleocr/ch_ppocr_v2_det.onnx";
    std::string image_path = "assets/images/text/small.png";

    if (argc > 1) model_path = argv[1];
    if (argc > 2) image_path = argv[2];

    idet::RuntimePolicy policy{};
    policy.ort_intra_threads = 1;
    policy.ort_inter_threads = 1;
    policy.tile_omp_threads = 1;
    policy.suppress_opencv = true;

    const idet::Status policy_status = idet::setup_runtime_policy(policy, false);
    if (!policy_status.ok()) {
        return fail("setup_runtime_policy failed: " + policy_status.message);
    }

    idet::DetectorConfig config = idet::DetectorConfig::setup(idet::Task::Text, model_path);
    config.verbose = false;
    config.runtime = policy;

    auto detector_result = idet::Detector::create(config);
    if (!detector_result.ok()) {
        return fail("Detector::create failed: " + detector_result.status().message);
    }
    idet::Detector detector = std::move(detector_result).value();

    auto image_result = idet::load_image(image_path, idet::PixelFormat::BGR_U8);
    if (!image_result.ok()) {
        return fail("load_image failed: " + image_result.status().message);
    }
    idet::Image image = std::move(image_result).value();

    auto detections = detector.detect(image);
    if (!detections.ok()) {
        return fail("detect failed: " + detections.status().message);
    }

    std::cout << "sync_dets_n: " << detections.value().size() << '\n';
    return 0;
}
