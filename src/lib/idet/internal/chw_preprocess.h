/**
 * @file chw_preprocess.h
 * @ingroup idet_internal
 * @brief Preprocessing helpers: convert BGR U8 OpenCV images into CHW float32 tensors.
 *
 * This header provides small, header-only utilities for preparing neural network inputs in a
 * common tensor layout:
 * - input: OpenCV @c cv::Mat in BGR format with type @c CV_8UC3
 * - output: contiguous @c float buffer in CHW layout (channels-first)
 *
 * Normalization:
 * - @p mean and @p inv_std must be specified in B, G, R order.
 * - Each channel is normalized as: @c (value - mean[c]) * inv_std[c].
 *
 * Performance notes:
 * - These routines are intentionally minimal and avoid extra abstractions.
 * - Only limited validation is performed; callers must respect documented preconditions.
 *
 * @note
 * This is an internal header and is not part of the stable public API.
 */

#pragma once

#include "internal/opencv_headers.h" // IWYU pragma: keep

#include <cstddef>
#include <cstdint>

namespace idet::internal {

/**
 * @brief Converts a BGR @c CV_8UC3 image into a CHW float32 tensor (same spatial size).
 *
 * The output layout is channels-first:
 * - @c dst_chw[0 * H*W ... 1 * H*W) contains the B plane
 * - @c dst_chw[1 * H*W ... 2 * H*W) contains the G plane
 * - @c dst_chw[2 * H*W ... 3 * H*W) contains the R plane
 *
 * Normalization is applied per channel:
 * @c out = (in - mean[c]) * inv_std[c], where @p mean / @p inv_std are provided in B,G,R order.
 *
 * Preconditions:
 * - @p bgr must be a non-empty @c cv::Mat with type @c CV_8UC3 (8-bit, 3 channels).
 * - @p dst_chw must point to a writable buffer of at least @c 3 * H * W floats.
 * - @p mean and @p inv_std must point to arrays of 3 floats (B,G,R order).
 *
 * @param bgr Input image (must be @c CV_8UC3).
 * @param dst_chw Output buffer of size @c 3 * H * W floats (CHW).
 * @param mean Per-channel mean in B,G,R order.
 * @param inv_std Per-channel inverse standard deviation in B,G,R order.
 *
 * @warning
 * Passing an image with a different type (not @c CV_8UC3) violates the preconditions and results
 * in incorrect interpretation of memory.
 *
 * @note
 * The function is @c noexcept and performs no allocations. It is safe to call concurrently as long
 * as the input image and output buffer are not concurrently mutated by other threads.
 */
inline void bgr_u8_to_chw_f32_same_size(const cv::Mat& bgr, float* dst_chw, const float mean[3],
                                        const float inv_std[3]) noexcept {
    const int H = bgr.rows;
    const int W = bgr.cols;

    const std::size_t plane = (std::size_t)H * (std::size_t)W;
    float* B = dst_chw + 0 * plane;
    float* G = dst_chw + 1 * plane;
    float* R = dst_chw + 2 * plane;

    for (int y = 0; y < H; ++y) {
        const std::uint8_t* p = bgr.ptr<std::uint8_t>(y);
        for (int x = 0; x < W; ++x) {
            const std::size_t idx = (std::size_t)y * (std::size_t)W + (std::size_t)x;
            B[idx] = (float(p[0]) - mean[0]) * inv_std[0];
            G[idx] = (float(p[1]) - mean[1]) * inv_std[1];
            R[idx] = (float(p[2]) - mean[2]) * inv_std[2];
            p += 3;
        }
    }
}

/**
 * @brief Resizes (if needed) and converts a BGR @c CV_8UC3 image into a CHW float32 tensor.
 *
 * If the input already matches the requested output size, this function performs a direct
 * conversion via @ref bgr_u8_to_chw_f32_same_size without allocating temporaries.
 *
 * Otherwise, it uses @c cv::resize (linear interpolation) into a temporary @c cv::Mat and then
 * converts the resized image.
 *
 * Preconditions:
 * - Same as @ref bgr_u8_to_chw_f32_same_size for the input image type and buffer sizes.
 * - @p dst_w and @p dst_h must be positive.
 * - @p dst_chw must point to a writable buffer of at least @c 3 * dst_h * dst_w floats.
 *
 * @param bgr Input image (must be @c CV_8UC3).
 * @param dst_w Destination width in pixels.
 * @param dst_h Destination height in pixels.
 * @param dst_chw Output buffer of size @c 3 * dst_h * dst_w floats (CHW).
 * @param mean Per-channel mean in B,G,R order.
 * @param inv_std Per-channel inverse standard deviation in B,G,R order.
 *
 * @throws cv::Exception
 * OpenCV may throw if resizing fails (e.g., invalid sizes or internal errors).
 *
 * @note
 * This function may allocate a temporary buffer only when resizing is required.
 */
inline void bgr_u8_to_chw_f32_resize(const cv::Mat& bgr, int dst_w, int dst_h, float* dst_chw, const float mean[3],
                                     const float inv_std[3]) {
    if (bgr.cols == dst_w && bgr.rows == dst_h) {
        bgr_u8_to_chw_f32_same_size(bgr, dst_chw, mean, inv_std);
        return;
    }
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);
    bgr_u8_to_chw_f32_same_size(resized, dst_chw, mean, inv_std);
}

} // namespace idet::internal
