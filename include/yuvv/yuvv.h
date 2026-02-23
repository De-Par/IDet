#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#if defined(YUVV_BUILD_STATIC)
    #define YUVV_API
#else
    #if defined(_WIN32) || defined(__CYGWIN__)
        #if defined(YUVV_BUILD_SHARED)
            #define YUVV_API __declspec(dllexport)
        #elif defined(YUVV_USE_SHARED)
            #define YUVV_API __declspec(dllimport)
        #else
            #define YUVV_API
        #endif
    #else
        #if defined(YUVV_BUILD_SHARED) && (defined(__GNUC__) || defined(__clang__))
            #define YUVV_API __attribute__((visibility("default")))
        #else
            #define YUVV_API
        #endif
    #endif
#endif

namespace yuvv {

enum class YuvFormat : uint8_t {
    I420 = 0,
    NV12 = 1,
    NV21 = 2,
    YUY2 = 3,
    UYVY = 4,
};

struct ViewerConfig {
    std::string file;
    int w = 0;
    int h = 0;
    YuvFormat fmt = YuvFormat::I420;
    double fps = 30.0;
    bool loop = true;
    int64_t start_frame = 0;
    int64_t max_frames = -1;
    std::string window_name = "YUV Viewer";
    bool overlay_info = true;
};

struct BgrFrameView {
    int w = 0;
    int h = 0;
    int channels = 3;
    int stride_bytes = 0;
    const uint8_t* data = nullptr;
};

using PostPreviewCallback = std::function<void(const BgrFrameView& frame, int64_t frame_idx)>;

class YUVV_API YuvViewer final {
  public:
    explicit YuvViewer(ViewerConfig cfg);
    ~YuvViewer();

    YuvViewer(YuvViewer&&) noexcept;
    YuvViewer& operator=(YuvViewer&&) noexcept;

    YuvViewer(const YuvViewer&) = delete;
    YuvViewer& operator=(const YuvViewer&) = delete;

    int run();
    void set_post_preview_callback(PostPreviewCallback cb);
    int64_t total_frames() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace yuvv
