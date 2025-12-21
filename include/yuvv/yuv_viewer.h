#pragma once

#if defined(__APPLE__)
#include <opencv2/opencv.hpp>
#else
#include <opencv4/opencv2/opencv.hpp>
#endif

#include "export.h"

#include <cstdint>
#include <functional>
#include <string>

namespace yuv {

enum class YuvFormat {
    I420, // YUV420p planar: Y + U + V
    NV12, // Y + interleaved UV
    NV21, // Y + interleaved VU
    YUY2, // packed 4:2:2 (YUYV)
    UYVY  // packed 4:2:2 (UYVY)
};

struct YUVV_API ViewerConfig {
    std::string file;
    int w = 0;
    int h = 0;
    YuvFormat fmt = YuvFormat::NV12;

    double fps = 30.0;
    bool loop = false;

    int64_t start_frame = 0;
    int64_t max_frames = -1; // -1 => all
    std::string window_name = "YUV Viewer";
    bool overlay_info = true;
};

YUVV_API void PrintUsage(const char* argv0);
YUVV_API bool ParseArgs(int argc, char** argv, ViewerConfig& cfg);

// Optional hook: called after preview is shown (so “before processing” preview is guaranteed).
using PostPreviewCallback = std::function<void(const cv::Mat& bgr, int64_t frame_idx)>;

class YUVV_API YuvViewer final {
  public:
    explicit YuvViewer(ViewerConfig cfg);

    // Run interactive viewer loop. Returns exit code.
    int run();

    void set_post_preview_callback(PostPreviewCallback cb) {
        post_preview_cb_ = std::move(cb);
    }

    int64_t total_frames() const {
        return total_frames_;
    }

    static bool parse_format_str(const std::string& s, YuvFormat& out);

  private:
    static size_t frame_size_bytes(int w, int h, YuvFormat fmt);
    static int cvt_code(YuvFormat fmt);

    bool open_file();
    bool seek_to_frame(int64_t frame_idx);
    bool read_frame_bgr(int64_t frame_idx, cv::Mat& out_bgr);

    void handle_key(int key);
    void restart();
    void save_last_frame_png();

  private:
    ViewerConfig cfg_;

    std::ifstream* file_ = nullptr; // managed manually to keep header light; allocated in cpp
    size_t frame_bytes_ = 0;
    int64_t total_frames_ = 0;

    std::vector<uint8_t> buf_;

    bool paused_ = false;
    bool step_once_ = false;

    int64_t frame_idx_ = 0;
    int64_t shown_ = 0;

    cv::Mat last_bgr_;
    int last_key_delay_ms_ = 1;

    PostPreviewCallback post_preview_cb_;
};

} // namespace yuv