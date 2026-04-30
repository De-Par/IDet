#include <idet.h>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <utility>

namespace {

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

void do_other_application_work(std::size_t& ticks) {
    ++ticks;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

    idet::DetectorWorkerOptions worker_options{};
    worker_options.copy_input = true;

    auto worker_result = idet::DetectorWorker::create(config, worker_options);
    if (!worker_result.ok()) {
        return fail("DetectorWorker::create failed: " + worker_result.status().message);
    }
    idet::DetectorWorker worker = std::move(worker_result).value();

    auto image_result = idet::load_image(image_path, idet::PixelFormat::BGR_U8);
    if (!image_result.ok()) {
        return fail("load_image failed: " + image_result.status().message);
    }
    idet::Image frame = std::move(image_result).value();

    constexpr int kFramesToProcess = 3;
    int submitted = 0;
    int completed = 0;
    std::size_t app_ticks = 0;

    while (completed < kFramesToProcess) {
        const idet::DetectorWorkerState state = worker.state();

        if (state == idet::DetectorWorkerState::Idle && submitted < kFramesToProcess) {
            const idet::Status submit_status = worker.submit(frame);
            if (!submit_status.ok()) {
                return fail("DetectorWorker::submit failed: " + submit_status.message);
            }
            ++submitted;
            continue;
        }

        if (state == idet::DetectorWorkerState::Ready) {
            auto detections = worker.take_result();
            if (!detections.ok()) {
                return fail("DetectorWorker::take_result failed: " + detections.status().message);
            }
            std::cout << "frame " << completed << " dets_n: " << detections.value().size() << '\n';
            ++completed;
            continue;
        }

        do_other_application_work(app_ticks);
    }

    std::cout << "submitted: " << submitted << ", completed: " << completed << ", app_ticks: " << app_ticks << '\n';
    return 0;
}
