#include "yuvv.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace yuv {

static std::string to_lower(std::string s) {
    for (auto& c : s)
        c = (char)std::tolower((unsigned char)c);
    return s;
}

bool YuvViewer::parse_format_str(const std::string& s, YuvFormat& out) {
    const std::string t = to_lower(s);
    if (t == "i420" || t == "yuv420p") {
        out = YuvFormat::I420;
        return true;
    }
    if (t == "nv12") {
        out = YuvFormat::NV12;
        return true;
    }
    if (t == "nv21") {
        out = YuvFormat::NV21;
        return true;
    }
    if (t == "yuy2" || t == "yuyv") {
        out = YuvFormat::YUY2;
        return true;
    }
    if (t == "uyvy") {
        out = YuvFormat::UYVY;
        return true;
    }
    return false;
}

size_t YuvViewer::frame_size_bytes(int w, int h, YuvFormat fmt) {
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

int YuvViewer::cvt_code(YuvFormat fmt) {
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

void PrintUsage(const char* argv0) {
    std::cerr << "Usage:\n"
              << "  " << argv0
              << " --file <path.yuv> --w <width> --h <height> --fmt <i420|nv12|nv21|yuy2|uyvy> [options]\n\n"
              << "Options:\n"
              << "  --fps <num>        Playback FPS (default 30)\n"
              << "  --loop             Loop playback\n"
              << "  --start <N>        Start from frame N (default 0)\n"
              << "  --count <N>        Show only N frames (default all)\n"
              << "  --no-overlay       Disable overlay text\n\n"
              << "Controls:\n"
              << "  SPACE  pause/resume\n"
              << "  n      next frame (when paused)\n"
              << "  r      restart\n"
              << "  s      save current frame (PNG)\n"
              << "  q/ESC  quit\n";
}

bool ParseArgs(int argc, char** argv, ViewerConfig& cfg) {
    auto need = [&](int& i, const char* name) -> const char* {
        if (i + 1 >= argc) {
            std::cerr << "Missing value after " << name << "\n";
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--file") {
            const char* v = need(i, "--file");
            if (!v) return false;
            cfg.file = v;
        } else if (k == "--w") {
            const char* v = need(i, "--w");
            if (!v) return false;
            cfg.w = std::stoi(v);
        } else if (k == "--h") {
            const char* v = need(i, "--h");
            if (!v) return false;
            cfg.h = std::stoi(v);
        } else if (k == "--fmt") {
            const char* v = need(i, "--fmt");
            if (!v) return false;
            if (!YuvViewer::parse_format_str(v, cfg.fmt)) {
                std::cerr << "Unknown format: " << v << "\n";
                return false;
            }
        } else if (k == "--fps") {
            const char* v = need(i, "--fps");
            if (!v) return false;
            cfg.fps = std::stod(v);
            if (cfg.fps <= 0) cfg.fps = 30.0;
        } else if (k == "--loop") {
            cfg.loop = true;
        } else if (k == "--start") {
            const char* v = need(i, "--start");
            if (!v) return false;
            cfg.start_frame = std::stoll(v);
            if (cfg.start_frame < 0) cfg.start_frame = 0;
        } else if (k == "--count") {
            const char* v = need(i, "--count");
            if (!v) return false;
            cfg.max_frames = std::stoll(v);
        } else if (k == "--no-overlay") {
            cfg.overlay_info = false;
        } else if (k == "--help" || k == "-h") {
            return false;
        } else {
            std::cerr << "Unknown argument: " << k << "\n";
            return false;
        }
    }

    if (cfg.file.empty() || cfg.w <= 0 || cfg.h <= 0) return false;
    return true;
}

YuvViewer::YuvViewer(ViewerConfig cfg) : cfg_(std::move(cfg)) {}

bool YuvViewer::open_file() {
    if (file_) return true;

    frame_bytes_ = frame_size_bytes(cfg_.w, cfg_.h, cfg_.fmt);
    if (frame_bytes_ == 0) {
        std::cerr << "[ERROR] Invalid frame size params.\n";
        return false;
    }
    if (cvt_code(cfg_.fmt) < 0) {
        std::cerr << "[ERROR] Unsupported YUV format for cvtColor.\n";
        return false;
    }

    auto* f = new std::ifstream(cfg_.file, std::ios::binary);
    if (!(*f)) {
        delete f;
        std::cerr << "[ERROR] Cannot open file: " << cfg_.file << "\n";
        return false;
    }

    // Compute total frames (floor)
    f->seekg(0, std::ios::end);
    const std::streamoff file_size = f->tellg();
    if (file_size <= 0) {
        delete f;
        std::cerr << "[ERROR] Empty file.\n";
        return false;
    }
    total_frames_ = (int64_t)(file_size / (std::streamoff)frame_bytes_);
    f->seekg(0, std::ios::beg);

    if (total_frames_ <= 0) {
        delete f;
        std::cerr << "[ERROR] File too small for one frame.\n";
        return false;
    }
    if (cfg_.start_frame >= total_frames_) {
        delete f;
        std::cerr << "[ERROR] start_frame " << cfg_.start_frame << " >= total_frames " << total_frames_ << "\n";
        return false;
    }

    buf_.assign(frame_bytes_, 0);
    file_ = f;

    frame_idx_ = cfg_.start_frame;
    shown_ = 0;
    paused_ = false;
    step_once_ = false;

    last_key_delay_ms_ = std::max(1, (int)(1000.0 / cfg_.fps + 0.5));
    return true;
}

bool YuvViewer::seek_to_frame(int64_t frame_idx) {
    const std::streamoff off = (std::streamoff)frame_idx * (std::streamoff)frame_bytes_;
    file_->clear();
    file_->seekg(off, std::ios::beg);
    return (bool)(*file_);
}

bool YuvViewer::read_frame_bgr(int64_t frame_idx, cv::Mat& out_bgr) {
    if (!seek_to_frame(frame_idx)) return false;

    file_->read(reinterpret_cast<char*>(buf_.data()), (std::streamsize)buf_.size());
    if (!(*file_) || file_->gcount() != (std::streamsize)buf_.size()) return false;

    cv::Mat yuv;
    if (cfg_.fmt == YuvFormat::I420 || cfg_.fmt == YuvFormat::NV12 || cfg_.fmt == YuvFormat::NV21) {
        yuv = cv::Mat(cfg_.h * 3 / 2, cfg_.w, CV_8UC1, buf_.data());
    } else {
        yuv = cv::Mat(cfg_.h, cfg_.w, CV_8UC2, buf_.data());
    }

    cv::cvtColor(yuv, out_bgr, cvt_code(cfg_.fmt));
    return true;
}

void YuvViewer::restart() {
    frame_idx_ = cfg_.start_frame;
    shown_ = 0;
    paused_ = false;
    step_once_ = false;
}

void YuvViewer::save_last_frame_png() {
    if (last_bgr_.empty()) {
        std::cerr << "[WARN] No frame to save yet.\n";
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

void YuvViewer::handle_key(int key) {
    if (key == 27 || key == 'q' || key == 'Q') {
        throw std::runtime_error("quit");
    } else if (key == ' ') {
        paused_ = !paused_;
        step_once_ = false;
    } else if (key == 'n' || key == 'N') {
        // step exactly one frame and stay paused
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

int YuvViewer::run() {
    if (!open_file()) return 2;

    cv::namedWindow(cfg_.window_name, cv::WINDOW_NORMAL);

    try {
        while (true) {
            // Stop conditions
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

                // preview BEFORE any user processing
                cv::Mat vis = bgr; // no clone needed unless you draw overlays
                if (cfg_.overlay_info) {
                    vis = bgr.clone();
                    std::string text =
                        "frame " + std::to_string(frame_idx_) + " / " + std::to_string(total_frames_ - 1);
                    cv::putText(vis, text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255),
                                2, cv::LINE_AA);
                }

                cv::imshow(cfg_.window_name, vis);
                last_bgr_ = bgr; // keep original BGR for saving / later use

                if (post_preview_cb_) {
                    post_preview_cb_(bgr, frame_idx_);
                }

                frame_idx_++;
                shown_++;

                if (step_once_) {
                    step_once_ = false;
                    paused_ = true; // stay paused after one step
                }
            }

            const int delay = paused_ ? 0 : last_key_delay_ms_;
            const int key = cv::waitKey(delay);
            if (key != -1) handle_key(key);
        }
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()) != "quit") throw;
    }

    // cleanup
    if (file_) {
        delete file_;
        file_ = nullptr;
    }
    return 0;
}

} // namespace yuv