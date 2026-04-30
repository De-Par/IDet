#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <idet.h>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct ExampleConfig {
    std::string model_path = "assets/models/paddleocr/ch_ppocr_v2_det.onnx";
    std::string video_path = "assets/videos/test.yuv";
    int width = 1920;
    int height = 1080;
    int max_frames = 3;
};

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

void do_other_application_work(std::size_t& ticks) {
    // Stand-in for the host application's own hot-loop work: polling sockets,
    // advancing simulation, rendering UI, updating state machines, or scheduling
    // other dependency-heavy tasks. Detection continues on DetectorWorker's
    // background thread while this function runs.
    ++ticks;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

bool parse_positive_int(std::string_view text, int& value) {
    if (text.empty()) return false;

    int parsed = 0;
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end || parsed <= 0) return false;

    value = parsed;
    return true;
}

bool parse_config(int argc, char** argv, ExampleConfig& config, std::string& error) {
    if (argc > 1) config.model_path = argv[1];
    if (argc > 2) config.video_path = argv[2];
    if (argc > 3 && !parse_positive_int(argv[3], config.width)) {
        error = "invalid width: expected a positive integer";
        return false;
    }
    if (argc > 4 && !parse_positive_int(argv[4], config.height)) {
        error = "invalid height: expected a positive integer";
        return false;
    }
    if (argc > 5 && !parse_positive_int(argv[5], config.max_frames)) {
        error = "invalid max_frames: expected a positive integer";
        return false;
    }

    return true;
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
    out = a * b;
    return true;
}

bool compute_i420_frame_size(int width, int height, std::size_t& frame_bytes) {
    if (width <= 0 || height <= 0) return false;

    // I420 stores one full-resolution Y plane and two half-resolution chroma
    // planes. Odd dimensions would make chroma addressing ambiguous for this
    // compact example reader, so reject them up front.
    if ((width % 2) != 0 || (height % 2) != 0) return false;

    std::size_t pixels = 0;
    if (!checked_mul(static_cast<std::size_t>(width), static_cast<std::size_t>(height), pixels)) {
        return false;
    }

    const std::size_t chroma_bytes = pixels / 2;
    if (pixels > std::numeric_limits<std::size_t>::max() - chroma_bytes) return false;

    frame_bytes = pixels + chroma_bytes;
    return true;
}

std::uint8_t clamp_to_u8(int value) {
    return static_cast<std::uint8_t>(std::clamp(value, 0, 255));
}

void convert_i420_to_bgr(const std::uint8_t* i420, int width, int height, std::vector<std::uint8_t>& bgr) {
    const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::uint8_t* y_plane = i420;
    const std::uint8_t* u_plane = y_plane + pixels;
    const std::uint8_t* v_plane = u_plane + pixels / 4;

    const int chroma_width = width / 2;
    const std::size_t row_stride = static_cast<std::size_t>(width) * 3U;

    for (int row = 0; row < height; ++row) {
        const std::size_t y_row = static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
        const std::size_t uv_row = static_cast<std::size_t>(row / 2) * static_cast<std::size_t>(chroma_width);
        std::uint8_t* dst = bgr.data() + static_cast<std::size_t>(row) * row_stride;

        for (int col = 0; col < width; ++col) {
            const int y = static_cast<int>(y_plane[y_row + static_cast<std::size_t>(col)]);
            const int u = static_cast<int>(u_plane[uv_row + static_cast<std::size_t>(col / 2)]);
            const int v = static_cast<int>(v_plane[uv_row + static_cast<std::size_t>(col / 2)]);

            // BT.601 limited-range YUV -> RGB conversion. IDet expects packed
            // BGR_U8 here, so store channels in B, G, R order.
            const int c = y - 16;
            const int d = u - 128;
            const int e = v - 128;
            const int r = (298 * c + 409 * e + 128) >> 8;
            const int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            const int b = (298 * c + 516 * d + 128) >> 8;

            const std::size_t dst_offset = static_cast<std::size_t>(col) * 3U;
            dst[dst_offset + 0] = clamp_to_u8(b);
            dst[dst_offset + 1] = clamp_to_u8(g);
            dst[dst_offset + 2] = clamp_to_u8(r);
        }
    }
}

class I420VideoReader {
  public:
    bool open(const std::string& path, int width, int height, std::string& error) {
        width_ = width;
        height_ = height;
        path_ = path;
        frames_read_ = 0;

        if (!compute_i420_frame_size(width_, height_, frame_bytes_)) {
            error = "invalid I420 geometry: width and height must be positive even numbers";
            return false;
        }
        if (frame_bytes_ > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
            error = "I420 frame is too large for std::istream::read";
            return false;
        }

        std::size_t row_stride = 0;
        if (!checked_mul(static_cast<std::size_t>(width_), 3U, row_stride)) {
            error = "BGR row stride overflow";
            return false;
        }
        row_stride_ = row_stride;

        std::size_t bgr_bytes = 0;
        if (!checked_mul(row_stride_, static_cast<std::size_t>(height_), bgr_bytes)) {
            error = "BGR frame size overflow";
            return false;
        }

        file_.open(path_, std::ios::binary);
        if (!file_) {
            error = "failed to open video: " + path_;
            return false;
        }

        file_.seekg(0, std::ios::end);
        const std::streamoff end_pos = file_.tellg();
        if (end_pos <= 0) {
            error = "video is empty or not seekable: " + path_;
            return false;
        }

        const auto file_bytes = static_cast<std::uint64_t>(end_pos);
        total_frames_ = static_cast<std::size_t>(file_bytes / frame_bytes_);
        trailing_bytes_ = static_cast<std::size_t>(file_bytes % frame_bytes_);
        if (total_frames_ == 0) {
            error = "video is smaller than one complete I420 frame";
            return false;
        }

        file_.clear();
        file_.seekg(0, std::ios::beg);

        yuv_.resize(frame_bytes_);
        bgr_.resize(bgr_bytes);
        return true;
    }

    bool read_next(idet::Image& frame, std::string& error) {
        if (frames_read_ >= total_frames_) {
            error = "no more complete I420 frames";
            return false;
        }

        const auto read_bytes = static_cast<std::streamsize>(frame_bytes_);
        file_.read(reinterpret_cast<char*>(yuv_.data()), read_bytes);
        if (file_.gcount() != read_bytes) {
            error = "failed to read a complete I420 frame";
            return false;
        }

        convert_i420_to_bgr(yuv_.data(), width_, height_, bgr_);

        // The returned image is a non-owning view over this reader's reusable BGR
        // buffer. The worker below uses copy_input=true, so submit() deep-copies the
        // frame before read_next() can reuse the buffer for the following video frame.
        frame = idet::Image::wrap(idet::PixelFormat::BGR_U8, width_, height_, bgr_.data(), row_stride_);
        ++frames_read_;
        return true;
    }

    std::size_t total_frames() const {
        return total_frames_;
    }

    std::size_t trailing_bytes() const {
        return trailing_bytes_;
    }

  private:
    std::string path_;
    std::ifstream file_;
    int width_ = 0;
    int height_ = 0;
    std::size_t frame_bytes_ = 0;
    std::size_t row_stride_ = 0;
    std::size_t total_frames_ = 0;
    std::size_t trailing_bytes_ = 0;
    std::size_t frames_read_ = 0;
    std::vector<std::uint8_t> yuv_;
    std::vector<std::uint8_t> bgr_;
};

} // namespace

int main(int argc, char** argv) {
    ExampleConfig example{};
    std::string error;
    if (!parse_config(argc, argv, example, error)) {
        return fail(error);
    }

    I420VideoReader reader;
    if (!reader.open(example.video_path, example.width, example.height, error)) {
        return fail(error);
    }

    // In a real embedding application this policy is usually derived from a CPU
    // budget owned by the host pipeline. Keep defaults small in examples to avoid
    // oversubscribing other dependencies in the same process.
    idet::RuntimePolicy policy{};
    policy.ort_intra_threads = 1;
    policy.ort_inter_threads = 1;
    policy.tile_omp_threads = 1;
    policy.suppress_opencv = true;

    const idet::Status policy_status = idet::setup_runtime_policy(policy, false);
    if (!policy_status.ok()) {
        return fail("setup_runtime_policy failed: " + policy_status.message);
    }

    idet::DetectorConfig config = idet::DetectorConfig::setup(idet::Task::Text, example.model_path);
    config.verbose = false;
    config.runtime = policy;

    // copy_input=true is the safest hot-loop handoff mode: submit() copies the
    // current frame into worker-owned memory, so the application can immediately
    // reuse or release its capture/conversion buffer.
    idet::DetectorWorkerOptions worker_options{};
    worker_options.copy_input = true;

    auto worker_result = idet::DetectorWorker::create(config, worker_options);
    if (!worker_result.ok()) {
        return fail("DetectorWorker::create failed: " + worker_result.status().message);
    }
    idet::DetectorWorker worker = std::move(worker_result).value();

    const std::size_t frames_to_process = std::min(static_cast<std::size_t>(example.max_frames), reader.total_frames());
    std::size_t loaded = 0;
    std::size_t submitted = 0;
    std::size_t completed = 0;
    std::size_t app_ticks = 0;
    idet::Image pending_frame;
    bool has_pending_frame = false;

    std::cout << "hot_loop_worker\n"
              << "  model: " << example.model_path << '\n'
              << "  video: " << example.video_path << '\n'
              << "  shape: " << example.width << "x" << example.height << " I420\n"
              << "  available_frames: " << reader.total_frames() << '\n'
              << "  trailing_bytes: " << reader.trailing_bytes() << '\n'
              << "  frames_to_process: " << frames_to_process << '\n';

    const auto load_pending_frame = [&]() -> bool {
        if (has_pending_frame || loaded >= frames_to_process) return true;

        if (!reader.read_next(pending_frame, error)) return false;
        std::cout << "prepared frame " << loaded << '\n';
        ++loaded;
        has_pending_frame = true;
        return true;
    };

    while (completed < frames_to_process) {
        idet::DetectorWorkerState state = worker.state();

        if (state == idet::DetectorWorkerState::Ready) {
            auto detections = worker.take_result();
            if (!detections.ok()) {
                return fail("DetectorWorker::take_result failed: " + detections.status().message);
            }
            std::cout << "frame " << completed << " dets_n: " << detections.value().size() << '\n';
            ++completed;
            if (completed >= frames_to_process) break;
            state = idet::DetectorWorkerState::Idle;
        }

        if (state == idet::DetectorWorkerState::Idle && submitted < frames_to_process) {
            if (!load_pending_frame()) {
                return fail("read_next failed: " + error);
            }
            if (!has_pending_frame) {
                return fail("internal pipeline error: no pending frame available for submit");
            }

            const idet::Status submit_status = worker.submit(pending_frame);
            if (!submit_status.ok()) {
                return fail("DetectorWorker::submit failed: " + submit_status.message);
            }
            std::cout << "submitted frame " << submitted << '\n';
            ++submitted;
            has_pending_frame = false;
            continue;
        }

        // Decode/convert at most one next frame while detection is running. The
        // pending frame stays valid because we do not overwrite the reader buffer
        // again until submit() has copied it into worker-owned memory.
        if (state == idet::DetectorWorkerState::Running && !has_pending_frame && loaded < frames_to_process) {
            if (!load_pending_frame()) {
                return fail("read_next failed: " + error);
            }
        }

        do_other_application_work(app_ticks);
    }

    std::cout << "loaded: " << loaded << ", submitted: " << submitted << ", completed: " << completed
              << ", app_ticks: " << app_ticks << '\n';
    return 0;
}
