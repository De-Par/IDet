#include "cli.h"

#include <cctype>
#include <iostream>
#include <string>

/**
 * @file cli.cpp
 * @brief CLI parsing implementation (see cli.h for API contract)
 */

// ASCII-only normalization for CLI inputs (format names)
static std::string to_lower(std::string s) {
    for (auto& c : s)
        c = (char)std::tolower((unsigned char)c);
    return s;
}

// Accept a few common aliases (e.g., yuv420p for I420)
static bool parse_format_str(const std::string& s, yuvv::YuvFormat& out) {
    const std::string t = to_lower(s);
    if (t == "i420" || t == "yuv420p") {
        out = yuvv::YuvFormat::I420;
        return true;
    }
    if (t == "nv12") {
        out = yuvv::YuvFormat::NV12;
        return true;
    }
    if (t == "nv21") {
        out = yuvv::YuvFormat::NV21;
        return true;
    }
    if (t == "yuy2" || t == "yuyv") {
        out = yuvv::YuvFormat::YUY2;
        return true;
    }
    if (t == "uyvy") {
        out = yuvv::YuvFormat::UYVY;
        return true;
    }
    return false;
}

void print_usage(const char* argv0) {
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

bool parse_args(int argc, char** argv, yuvv::ViewerConfig& cfg) {
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
            if (!parse_format_str(v, cfg.fmt)) {
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
