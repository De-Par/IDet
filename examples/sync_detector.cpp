#include <cstddef>
#include <idet.h>
#include <iostream>
#include <string>
#include <utility>

namespace {

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

void print_detection(std::size_t index, const idet::Quad& quad) {
    std::cout << "  [" << index << "]"
              << " tl=(" << quad[0].x << ", " << quad[0].y << ")"
              << " tr=(" << quad[1].x << ", " << quad[1].y << ")"
              << " br=(" << quad[2].x << ", " << quad[2].y << ")"
              << " bl=(" << quad[3].x << ", " << quad[3].y << ")\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string model_path = "assets/models/paddleocr/ch_ppocr_v2_det.onnx";
    std::string image_path = "assets/images/text/small.png";

    if (argc > 1) model_path = argv[1];
    if (argc > 2) image_path = argv[2];

    // This example is intentionally blocking: the caller thread owns the whole request
    // lifecycle, waits for inference to finish, and then prints a detailed result dump.
    // Use it for CLI tools, smoke checks, or pipeline stages that already run on a
    // dedicated inference thread.
    idet::RuntimePolicy policy{};
    policy.ort_intra_threads = 1;
    policy.ort_inter_threads = 1;
    policy.tile_omp_threads = 1;
    policy.suppress_opencv = true;

    const idet::Status policy_status = idet::setup_runtime_policy(policy, false);
    if (!policy_status.ok()) {
        return fail("setup_runtime_policy failed: " + policy_status.message);
    }

    // DetectorConfig carries model/task options plus the local runtime budget consumed by
    // the detector. The process-global parts of the policy were applied explicitly above,
    // so the library does not surprise a larger application by changing OpenMP/OpenCV state
    // inside Detector construction.
    idet::DetectorConfig config = idet::DetectorConfig::setup(idet::Task::Text, model_path);
    config.verbose = false;
    config.runtime = policy;

    auto detector_result = idet::Detector::create(config);
    if (!detector_result.ok()) {
        return fail("Detector::create failed: " + detector_result.status().message);
    }
    idet::Detector detector = std::move(detector_result).value();

    // load_image is an application-boundary helper. It decodes the file and returns an owning
    // idet::Image, so the pixels stay valid for the blocking detect call below.
    auto image_result = idet::load_image(image_path, idet::PixelFormat::BGR_U8);
    if (!image_result.ok()) {
        return fail("load_image failed: " + image_result.status().message);
    }
    idet::Image image = std::move(image_result).value();

    const idet::ImageView& view = image.view();

    std::cout << "sync_detector\n"
              << "  model: " << model_path << '\n'
              << "  image: " << image_path << '\n'
              << "  shape: " << view.width << "x" << view.height << '\n'
              << "  stride_bytes: " << view.stride_bytes << '\n';

    // The synchronous API returns all detections in the caller thread. For text detection
    // each result is a quadrilateral normalized to TL -> TR -> BR -> BL point order.
    auto detections = detector.detect(image);
    if (!detections.ok()) {
        return fail("detect failed: " + detections.status().message);
    }

    const idet::VecQuad& quads = detections.value();
    std::cout << "sync_dets_n: " << quads.size() << '\n';
    for (std::size_t i = 0; i < quads.size(); ++i) {
        print_detection(i, quads[i]);
    }

    return 0;
}
