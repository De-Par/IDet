/**
 * @file letterbox_info.h
 * @ingroup idet_internal
 * @brief OpenCV-free geometry metadata for aspect-preserving letterbox transforms.
 */

#pragma once

namespace idet::internal {

/**
 * @brief Geometry information describing an aspect-preserving resize-and-pad operation.
 *
 * @details
 * Forward transform applied to the source image:
 *   1) scale by @ref scale (uniform on both axes),
 *   2) translate by (@ref pad_x, @ref pad_y) so the resized content lands in the top-left
 *      of the target buffer.
 *
 * To map network-input coordinates back to original image coordinates:
 *   x_orig = (x_net - pad_x) / scale
 *   y_orig = (y_net - pad_y) / scale
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

} // namespace idet::internal
