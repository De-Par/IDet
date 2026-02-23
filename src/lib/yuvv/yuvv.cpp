#include "yuvv.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(__has_include) && __has_include(<opencv4/opencv2/opencv.hpp>)
    #include <opencv4/opencv2/opencv.hpp> // IWYU pragma: keep
#elif defined(__has_include) && __has_include(<opencv2/opencv.hpp>)
    #include <opencv2/opencv.hpp> // IWYU pragma: keep
#else
    #error "[ERROR] OpenCV 'opencv.hpp' header not found"
#endif

/**
 * @file yuvv.cpp
 * @brief Implementation of a lightweight raw YUV player with OpenCV preview
 *
 * Supported formats:
 *  - I420 (YUV420p planar), NV12, NV21 (YUV420 semi-planar)
 *  - YUY2, UYVY (YUV422 packed)
 */

namespace yuvv {

class YuvViewer::Impl {
  public:
    explicit Impl(ViewerConfig cfg) : cfg_(std::move(cfg)) {}

    int run() {
        if (!open_file()) return 2;

        cv::namedWindow(cfg_.window_name, cv::WINDOW_NORMAL);

        try {
            while (true) {
                if (cfg_.max_frames >= 0 && shown_ >= cfg_.max_frames) {
                    if (cfg_.loop)
                        restart();
                    else
                        break;
                }
                if (frame_idx_ >= total_frames_) {
                    if (cfg_.loop)
                        restart();
                    else
                        break;
                }

                const bool should_show = (!paused_) || (paused_ && step_once_);
                if (should_show) {
                    cv::Mat bgr;
                    if (!read_frame_bgr(frame_idx_, bgr)) {
                        std::cerr << "[ERROR] Read failed at frame " << frame_idx_ << "\n";
                        break;
                    }

                    cv::Mat vis = bgr;
                    if (cfg_.overlay_info) {
                        vis = bgr.clone();
                        const std::string text =
                            "frame " + std::to_string(frame_idx_ + 1) + " / " + std::to_string(total_frames_);
                        cv::putText(vis, text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
                    }

                    cv::imshow(cfg_.window_name, vis);
                    last_bgr_ = bgr;

                    if (post_preview_cb_) {
                        post_preview_cb_(make_view(bgr), frame_idx_);
                    }

                    frame_idx_++;
                    shown_++;

                    if (step_once_) {
                        step_once_ = false;
                        paused_ = true;
                    }
                }

                const int delay = paused_ ? 0 : last_key_delay_ms_;
                const int key = cv::waitKey(delay);
                if (key != -1) handle_key(key);

                const double wnd_vis = cv::getWindowProperty(cfg_.window_name, cv::WND_PROP_VISIBLE);
                if (wnd_vis < 1.0) {
                    return 0; // closed by window [X]
                }
            }
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()) != "quit") throw;
        }

        return 0;
    }

    int64_t total_frames() const {
        return total_frames_;
    }

    void set_post_preview_callback(PostPreviewCallback cb) {
        post_preview_cb_ = std::move(cb);
    }

  private:
    // Computes the expected raw frame byte size for a given format
    static size_t frame_size_bytes(int w, int h, YuvFormat fmt) {
        switch (fmt) {
        case YuvFormat::I420:
        case YuvFormat::NV12:
        case YuvFormat::NV21:
            return (size_t)w * (size_t)h * 3 / 2; // 4:2:0
        case YuvFormat::YUY2:
        case YuvFormat::UYVY:
            return (size_t)w * (size_t)h * 2; // 4:2:2 packed
        }
        return 0;
    }

    // Maps the internal format enum to OpenCV conversion code for cv::cvtColor()
    static int cvt_code(YuvFormat fmt) {
        switch (fmt) {
        case YuvFormat::I420:
            return cv::COLOR_YUV2BGR_I420;
        case YuvFormat::NV12:
            return cv::COLOR_YUV2BGR_NV12;
        case YuvFormat::NV21:
            return cv::COLOR_YUV2BGR_NV21;
        case YuvFormat::YUY2:
            return cv::COLOR_YUV2BGR_YUY2;
        case YuvFormat::UYVY:
            return cv::COLOR_YUV2BGR_UYVY;
        }
        return -1;
    }

    // Convert OpenCV matrix structure to custom wrapper
    static BgrFrameView make_view(const cv::Mat& bgr) {
        BgrFrameView v;
        v.w = bgr.cols;
        v.h = bgr.rows;
        v.channels = bgr.channels();
        v.stride_bytes = static_cast<int>(bgr.step);
        v.data = bgr.ptr<uint8_t>(0);
        return v;
    }

    bool open_file() {
        if (file_opened_) return true;

        frame_bytes_ = frame_size_bytes(cfg_.w, cfg_.h, cfg_.fmt);
        if (frame_bytes_ == 0) {
            std::cerr << "[ERROR] Invalid frame size params\n";
            return false;
        }
        if (cvt_code(cfg_.fmt) < 0) {
            std::cerr << "[ERROR] Unsupported YUV format for cvtColor\n";
            return false;
        }

        file_.open(cfg_.file, std::ios::binary);
        if (!file_) {
            std::cerr << "[ERROR] Cannot open file: " << cfg_.file << "\n";
            return false;
        }

        file_.seekg(0, std::ios::end);
        const std::streamoff file_size = file_.tellg();
        if (file_size <= 0) {
            std::cerr << "[ERROR] Empty file\n";
            return false;
        }

        total_frames_ = (int64_t)(file_size / (std::streamoff)frame_bytes_);
        file_.seekg(0, std::ios::beg);

        if (total_frames_ <= 0) {
            std::cerr << "[ERROR] File too small for one frame\n";
            return false;
        }
        if (cfg_.start_frame >= total_frames_) {
            std::cerr << "[ERROR] start_frame " << cfg_.start_frame << " >= total_frames " << total_frames_ << "\n";
            return false;
        }

        buf_.assign(frame_bytes_, 0);

        frame_idx_ = cfg_.start_frame;
        shown_ = 0;
        paused_ = false;
        step_once_ = false;

        last_key_delay_ms_ = std::max(1, (int)(1000.0 / cfg_.fps + 0.5));

        file_opened_ = true;
        return true;
    }

    bool seek_to_frame(int64_t frame_idx) {
        const std::streamoff off = (std::streamoff)frame_idx * (std::streamoff)frame_bytes_;
        file_.clear();
        file_.seekg(off, std::ios::beg);
        return (bool)file_;
    }

    bool read_frame_bgr(int64_t frame_idx, cv::Mat& out_bgr) {
        if (!seek_to_frame(frame_idx)) return false;

        file_.read(reinterpret_cast<char*>(buf_.data()), (std::streamsize)buf_.size());
        if (!file_ || file_.gcount() != (std::streamsize)buf_.size()) return false;

        cv::Mat yuv;
        if (cfg_.fmt == YuvFormat::I420 || cfg_.fmt == YuvFormat::NV12 || cfg_.fmt == YuvFormat::NV21) {
            yuv = cv::Mat(cfg_.h * 3 / 2, cfg_.w, CV_8UC1, buf_.data());
        } else {
            yuv = cv::Mat(cfg_.h, cfg_.w, CV_8UC2, buf_.data());
        }

        cv::cvtColor(yuv, out_bgr, cvt_code(cfg_.fmt));
        return true;
    }

    void restart() {
        frame_idx_ = cfg_.start_frame;
        shown_ = 0;
        paused_ = false;
        step_once_ = false;
    }

    void save_last_frame_png() {
        if (last_bgr_.empty()) {
            std::cerr << "[WARN] No frame to save yet\n";
            return;
        }

        std::ostringstream oss;
        oss << "frame_" << std::setw(6) << std::setfill('0') << (frame_idx_ > 0 ? (frame_idx_ - 1) : 0) << ".png";

        if (cv::imwrite(oss.str(), last_bgr_)) {
            std::cerr << "[INFO] Saved " << oss.str() << "\n";
        } else {
            std::cerr << "[ERROR] Failed to save " << oss.str() << "\n";
        }
    }

    void handle_key(int key) {
        if (key == 27 || key == 'q' || key == 'Q') {
            throw std::runtime_error("quit");
        } else if (key == ' ') {
            paused_ = !paused_;
            step_once_ = false;
        } else if (key == 'n' || key == 'N') {
            if (paused_)
                step_once_ = true;
            else {
                paused_ = true;
                step_once_ = true;
            }
        } else if (key == 'r' || key == 'R') {
            restart();
        } else if (key == 's' || key == 'S') {
            save_last_frame_png();
        }
    }

  private:
    ViewerConfig cfg_;

    std::ifstream file_;
    bool file_opened_ = false;

    size_t frame_bytes_ = 0;
    int64_t total_frames_ = 0;

    std::vector<uint8_t> buf_;

    bool paused_ = false;
    bool step_once_ = false;

    int64_t frame_idx_ = 0;
    int64_t shown_ = 0;

    int last_key_delay_ms_ = 1;

    cv::Mat last_bgr_;
    PostPreviewCallback post_preview_cb_;
};

YuvViewer::YuvViewer(ViewerConfig cfg) : impl_(std::make_unique<Impl>(std::move(cfg))) {}

YuvViewer::~YuvViewer() = default;

YuvViewer::YuvViewer(YuvViewer&&) noexcept = default;
YuvViewer& YuvViewer::operator=(YuvViewer&&) noexcept = default;

int YuvViewer::run() {
    return impl_->run();
}

int64_t YuvViewer::total_frames() const {
    return impl_->total_frames();
}

void YuvViewer::set_post_preview_callback(PostPreviewCallback cb) {
    impl_->set_post_preview_callback(std::move(cb));
}

} // namespace yuvv
