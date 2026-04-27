/**
 * @file letterbox.h
 * @ingroup idet_internal
 * @brief Aspect-preserving resize with constant padding for fixed-shape network inputs.
 *
 * @details
 * Most CNN detectors (DBNet, SCRFD, YOLO families) are trained on inputs that preserve the
 * aspect ratio of the source image, with any unused area filled with a constant pad value
 * (typically 0 for DBNet/SCRFD and 114 for YOLO). Feeding the model a directly-stretched
 * image breaks aspect ratio and slightly biases the regression heads, producing detections
 * that are shifted and either over-extended or under-covering relative to the target.
 *
 * This header provides a single helper, @ref letterbox_bgr, that:
 * - rescales the image so it fits inside the target rectangle while preserving aspect ratio,
 * - pads the remainder with a caller-supplied constant value,
 * - returns a @ref LetterboxInfo struct so the postprocessing path can undo the transform.
 *
 * Mapping back to original image coordinates:
 *   x_orig = (x_net - pad_x) / scale
 *   y_orig = (y_net - pad_y) / scale
 *
 * @note This is an internal header and is not part of the stable public API.
 */

#pragma once

#include "internal/opencv_headers.h" // IWYU pragma: keep

#include <algorithm>
#include <cmath>

namespace idet::internal {

/**
 * @brief Geometry information describing an aspect-preserving resize-and-pad operation.
 *
 * @details
 * Forward transform applied to the source image:
 *   1) scale by @ref scale (uniform on both axes),
 *   2) translate by (@ref pad_x, @ref pad_y) so the resized content lands in the top-left
 *      of the target buffer (we use top-left padding, NOT centered).
 *
 * The outer rectangle (resized + padding) has dimensions (@ref dst_w, @ref dst_h).
 *
 * @par Choice of top-left vs centered padding
 * Top-left padding is the convention used by PaddleOCR-style DBNet and most YOLO ONNX
 * exports. It also makes the inverse transform a single subtract+divide. SCRFD inference
 * code in this library uses the same convention to keep one shared helper.
 */
struct LetterboxInfo {
    /** @brief Uniform scale applied to the source (both axes). */
    float scale = 1.0f;

    /** @brief Horizontal pad placed before the resized content (in target pixels). */
    int pad_x = 0;

    /** @brief Vertical pad placed before the resized content (in target pixels). */
    int pad_y = 0;

    /** @brief Width of the destination (network input) image. */
    int dst_w = 0;

    /** @brief Height of the destination (network input) image. */
    int dst_h = 0;

    /** @brief Width of the resized content inside the destination buffer. */
    int resized_w = 0;

    /** @brief Height of the resized content inside the destination buffer. */
    int resized_h = 0;
};

/**
 * @brief Aspect-preserving resize with constant padding ("letterbox").
 *
 * @details
 * Computes the largest uniform @c scale such that:
 *   resized_w = round(src.cols * scale) <= dst_w
 *   resized_h = round(src.rows * scale) <= dst_h
 *
 * The image is then resized via @c cv::INTER_LINEAR and copied into the top-left corner of a
 * @c (dst_h, dst_w) buffer of type @c CV_8UC3. The remaining area is filled with the
 * caller-supplied @p pad value on each channel (BGR).
 *
 * @param src       Source image (must be @c CV_8UC3, non-empty).
 * @param dst       Destination image. Resized to @c (dst_h, dst_w) @c CV_8UC3 if needed.
 * @param dst_w     Target width in pixels (must be > 0).
 * @param dst_h     Target height in pixels (must be > 0).
 * @param pad_value Pixel intensity for padded area, applied to all 3 channels.
 *
 * @return Geometry of the letterbox transform (@ref LetterboxInfo).
 *
 * @note
 * If @p src is empty or has the wrong type, the function falls back to producing a blank
 * @c (dst_h, dst_w) image filled with @p pad_value and a @ref LetterboxInfo with @c scale=1
 * and @c pad_x=pad_y=0. This keeps callers simple while still allowing them to detect a
 * degenerate input via @c src.empty() before calling.
 */
inline LetterboxInfo letterbox_bgr(const cv::Mat& src, cv::Mat& dst, int dst_w, int dst_h,
                                   std::uint8_t pad_value) noexcept {
    LetterboxInfo info{};
    info.dst_w = dst_w;
    info.dst_h = dst_h;

    if (dst_w <= 0 || dst_h <= 0) {
        dst.release();
        return info;
    }

    const cv::Scalar fill(static_cast<double>(pad_value), static_cast<double>(pad_value),
                          static_cast<double>(pad_value));

    if (src.empty() || src.type() != CV_8UC3 || src.cols <= 0 || src.rows <= 0) {
        dst = cv::Mat(dst_h, dst_w, CV_8UC3, fill);
        return info;
    }

    const float sx = static_cast<float>(dst_w) / static_cast<float>(src.cols);
    const float sy = static_cast<float>(dst_h) / static_cast<float>(src.rows);
    const float scale = std::min(sx, sy);

    // Round to nearest pixel; clamp to [1, dst_*] to defend against degenerate scales.
    int rw = static_cast<int>(std::lround(src.cols * scale));
    int rh = static_cast<int>(std::lround(src.rows * scale));
    rw = std::max(1, std::min(rw, dst_w));
    rh = std::max(1, std::min(rh, dst_h));

    info.scale = scale;
    info.pad_x = 0; // top-left padding by convention (see header doc)
    info.pad_y = 0;
    info.resized_w = rw;
    info.resized_h = rh;

    // Allocate destination once and either resize directly into the top-left ROI or fill +
    // resize. We always (re)initialize the full buffer with the pad value so the right/bottom
    // strips are deterministic without needing copyMakeBorder.
    dst = cv::Mat(dst_h, dst_w, CV_8UC3, fill);

    cv::Mat roi = dst(cv::Rect(0, 0, rw, rh));
    if (rw == src.cols && rh == src.rows) {
        src.copyTo(roi);
    } else {
        cv::resize(src, roi, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
    }
    return info;
}

} // namespace idet::internal
