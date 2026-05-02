/**
 * @file yuvv.h
 * @brief Public API for the lightweight YUV video viewer helper.
 *
 * The YUVV API is intentionally small: configure a raw YUV stream, optionally install a
 * post-preview callback that receives BGR frames, and run the viewer loop. It is used by
 * examples and tooling around IDet, but is kept as a separate library surface with its own
 * export macro.
 *
 * @ingroup yuvv_api
 */

/**
 * @defgroup yuvv_api YUVV Public API
 * @brief Raw YUV viewing utilities.
 * @{
 */

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

/**
 * @brief Supported raw YUV pixel layouts.
 *
 * Values describe how bytes are laid out in the source file. Dimensions passed through
 * @ref yuvv::ViewerConfig must be compatible with the selected format; planar 4:2:0 formats
 * require even width and height.
 */
enum class YuvFormat : uint8_t {
    /** Planar YUV 4:2:0: Y plane, then U plane, then V plane. */
    I420 = 0,
    /** Semi-planar YUV 4:2:0: Y plane followed by interleaved UV. */
    NV12 = 1,
    /** Semi-planar YUV 4:2:0: Y plane followed by interleaved VU. */
    NV21 = 2,
    /** Packed YUV 4:2:2: Y0 U0 Y1 V0 byte order. */
    YUY2 = 3,
    /** Packed YUV 4:2:2: U0 Y0 V0 Y1 byte order. */
    UYVY = 4,
};

/**
 * @brief Runtime configuration for @ref YuvViewer.
 */
struct ViewerConfig {
    /** @brief Path to the raw YUV file. */
    std::string file;
    /** @brief Frame width in pixels. */
    int w = 0;
    /** @brief Frame height in pixels. */
    int h = 0;
    /** @brief Raw frame format. */
    YuvFormat fmt = YuvFormat::I420;
    /** @brief Playback frame rate. Values <= 0 are implementation-defined. */
    double fps = 30.0;
    /** @brief Restart from the first frame when the end of file is reached. */
    bool loop = true;
    /** @brief Zero-based frame index to start from. */
    int64_t start_frame = 0;
    /** @brief Maximum number of frames to preview; negative means no explicit limit. */
    int64_t max_frames = -1;
    /** @brief Native window title used by the preview backend. */
    std::string window_name = "YUV Viewer";
    /** @brief Draw frame/format information on top of the preview. */
    bool overlay_info = true;
};

/**
 * @brief Non-owning BGR frame view passed to callbacks.
 *
 * The memory is owned by the viewer implementation and is valid only for the duration of
 * the callback call.
 */
struct BgrFrameView {
    /** @brief Frame width in pixels. */
    int w = 0;
    /** @brief Frame height in pixels. */
    int h = 0;
    /** @brief Number of interleaved channels. The viewer currently emits BGR with 3 channels. */
    int channels = 3;
    /** @brief Row stride in bytes. */
    int stride_bytes = 0;
    /** @brief Pointer to the first byte of the first BGR row. */
    const uint8_t* data = nullptr;
};

/**
 * @brief Callback invoked after a frame has been converted to BGR for preview.
 *
 * @param frame Non-owning BGR view valid only during the callback.
 * @param frame_idx Zero-based source frame index.
 */
using PostPreviewCallback = std::function<void(const BgrFrameView& frame, int64_t frame_idx)>;

/**
 * @brief Raw YUV file viewer with optional BGR callback hook.
 *
 * @thread_safety
 * The viewer object is move-only and not internally synchronized for concurrent member calls.
 * Drive it from one thread, or provide external synchronization.
 */
class YUVV_API YuvViewer final {
  public:
    /**
     * @brief Construct a viewer from a configuration snapshot.
     * @param cfg Viewer configuration.
     */
    explicit YuvViewer(ViewerConfig cfg);

    /** @brief Stop preview and release backend resources. */
    ~YuvViewer();

    /** @brief Move-construct a viewer, transferring backend ownership. */
    YuvViewer(YuvViewer&&) noexcept;

    /** @brief Move-assign a viewer, replacing the current backend. */
    YuvViewer& operator=(YuvViewer&&) noexcept;

    /** @brief Copy construction is disabled because the backend owns native resources. */
    YuvViewer(const YuvViewer&) = delete;

    /** @brief Copy assignment is disabled. */
    YuvViewer& operator=(const YuvViewer&) = delete;

    /**
     * @brief Run the viewer loop.
     * @return Process exit-style code: 0 on success, non-zero on failure.
     */
    int run();

    /**
     * @brief Install or replace the callback invoked after BGR conversion.
     * @param cb Callback object. Passing an empty callback disables the hook.
     */
    void set_post_preview_callback(PostPreviewCallback cb);

    /**
     * @brief Return the number of complete frames available in the configured file.
     * @return Total frame count, or 0 before successful initialization / for invalid input.
     */
    int64_t total_frames() const;

  private:
    /** @brief Hidden implementation that owns OpenCV/UI resources. */
    class Impl;

    /** @brief Owned implementation pointer. */
    std::unique_ptr<Impl> impl_;
};

} // namespace yuvv

/** @} */ // end of group yuvv_api
