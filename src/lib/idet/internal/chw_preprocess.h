/**
 * @file chw_preprocess.h
 * @ingroup idet_internal
 * @brief Preprocessing helpers: convert BGR U8 image views into CHW float32 tensors.
 *
 * This header provides small, header-only utilities for preparing neural network inputs in a
 * common tensor layout:
 * - input: OpenCV-free @ref BgrImageView in BGR_U8 layout
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

#include "internal/bgr_image.h"
#include "internal/letterbox_info.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(_OPENMP)
    #include <omp.h>
#endif

namespace idet::internal {

namespace detail {

inline void fill_chw_constant_(float* dst_chw, int w, int h, std::uint8_t value, const float mean[3],
                               const float inv_std[3]) noexcept {
    if (!dst_chw || w <= 0 || h <= 0) return;

    const std::size_t plane = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    for (int c = 0; c < 3; ++c) {
        const float v = (static_cast<float>(value) - mean[c]) * inv_std[c];
        std::fill(dst_chw + static_cast<std::size_t>(c) * plane, dst_chw + static_cast<std::size_t>(c + 1) * plane, v);
    }
}

} // namespace detail

/**
 * @brief Converts a BGR_U8 image view into a CHW float32 tensor (same spatial size).
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
 * - @p bgr must be a valid BGR_U8 view.
 * - @p dst_chw must point to a writable buffer of at least @c 3 * H * W floats.
 * - @p mean and @p inv_std must point to arrays of 3 floats (B,G,R order).
 *
 * @param bgr Input BGR_U8 image view.
 * @param dst_chw Output buffer of size @c 3 * H * W floats (CHW).
 * @param mean Per-channel mean in B,G,R order.
 * @param inv_std Per-channel inverse standard deviation in B,G,R order.
 *
 * @warning
 * Passing an invalid view violates the preconditions and results in no output guarantees.
 *
 * @note
 * The function is @c noexcept and performs no allocations. It is safe to call concurrently as long
 * as the input image and output buffer are not concurrently mutated by other threads.
 */
inline void bgr_u8_to_chw_f32_same_size(const BgrImageView& bgr, float* dst_chw, const float mean[3],
                                        const float inv_std[3]) noexcept {
    const int H = bgr.height;
    const int W = bgr.width;

    const std::size_t plane = static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
    float* const B = dst_chw + 0 * plane;
    float* const G = dst_chw + 1 * plane;
    float* const R = dst_chw + 2 * plane;

    const float mB = mean[0], mG = mean[1], mR = mean[2];
    const float sB = inv_std[0], sG = inv_std[1], sR = inv_std[2];

    // Hot path: runs for every inference and every tile in tiled mode.
    //
    // - Rows are independent, so we parallelize across @c y when the image is large enough to
    //   amortize fork/join cost. @c OMP_MAX_ACTIVE_LEVELS=1 (set by platform/omp_config.cpp)
    //   prevents oversubscription when called from inside infer_tiled()'s parallel region.
    // - Per-row pointers are precomputed to avoid per-pixel index multiplications and to give
    //   the compiler an easier time auto-vectorizing the inner loop.
    constexpr int kParallelMinPixels = 64 * 64;
    bool parallel = (H * W) >= kParallelMinPixels;
#if defined(_OPENMP)
    // Avoid spawning a nested OpenMP region when preprocessing is called from tiled inference.
    // The outer tile loop already owns the CPU budget; keeping the inner loop serial prevents
    // oversubscription and removes repeated fork/join overhead in tile-heavy pipelines.
    parallel = parallel && !omp_in_parallel();
#endif

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (parallel)
#endif
    for (int y = 0; y < H; ++y) {
        const std::uint8_t* p = bgr.row(y);
        const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(W);
        float* br = B + row;
        float* gr = G + row;
        float* rr = R + row;
        for (int x = 0; x < W; ++x) {
            const int x3 = 3 * x;
            br[x] = (float(p[x3 + 0]) - mB) * sB;
            gr[x] = (float(p[x3 + 1]) - mG) * sG;
            rr[x] = (float(p[x3 + 2]) - mR) * sR;
        }
    }
    (void)parallel; // silence unused-var when _OPENMP is not defined
}

/**
 * @brief Aspect-preserving letterbox resize and BGR_U8 to CHW float32 conversion.
 *
 * @details
 * The destination tensor is first filled with the normalized pad value, then the source image is
 * bilinearly resized into the top-left letterbox content rectangle. This avoids allocating an
 * intermediate resized BGR image in the inference hot path.
 *
 * The resize mapping follows the common half-pixel convention used by bilinear image resizers.
 *
 * Preconditions:
 * - @p dst_w and @p dst_h must be positive.
 * - @p dst_chw must point to a writable buffer of at least @c 3 * dst_h * dst_w floats.
 *
 * @param bgr Input BGR_U8 image view.
 * @param dst_w Destination width in pixels.
 * @param dst_h Destination height in pixels.
 * @param pad_value Pixel intensity for padded area, applied to all 3 channels.
 * @param dst_chw Output buffer of size @c 3 * dst_h * dst_w floats (CHW).
 * @param mean Per-channel mean in B,G,R order.
 * @param inv_std Per-channel inverse standard deviation in B,G,R order.
 * @return Letterbox geometry used to map decoded detections back to source-image coordinates.
 */
inline LetterboxInfo bgr_u8_to_chw_f32_letterbox(const BgrImageView& bgr, int dst_w, int dst_h, std::uint8_t pad_value,
                                                 float* dst_chw, const float mean[3], const float inv_std[3]) noexcept {
    LetterboxInfo info{};
    info.dst_w = dst_w;
    info.dst_h = dst_h;

    if (dst_w <= 0 || dst_h <= 0 || !dst_chw) return info;

    if (!bgr.is_valid()) {
        detail::fill_chw_constant_(dst_chw, dst_w, dst_h, pad_value, mean, inv_std);
        return info;
    }

    const float sx = static_cast<float>(dst_w) / static_cast<float>(bgr.width);
    const float sy = static_cast<float>(dst_h) / static_cast<float>(bgr.height);
    const float scale = std::min(sx, sy);

    int rw = static_cast<int>(std::lround(static_cast<float>(bgr.width) * scale));
    int rh = static_cast<int>(std::lround(static_cast<float>(bgr.height) * scale));
    rw = std::max(1, std::min(rw, dst_w));
    rh = std::max(1, std::min(rh, dst_h));

    info.scale = scale;
    info.pad_x = 0;
    info.pad_y = 0;
    info.resized_w = rw;
    info.resized_h = rh;

    if (rw == bgr.width && rh == bgr.height && dst_w == bgr.width && dst_h == bgr.height) {
        bgr_u8_to_chw_f32_same_size(bgr, dst_chw, mean, inv_std);
        return info;
    }

    detail::fill_chw_constant_(dst_chw, dst_w, dst_h, pad_value, mean, inv_std);

    const std::size_t plane = static_cast<std::size_t>(dst_w) * static_cast<std::size_t>(dst_h);
    float* const B = dst_chw + 0 * plane;
    float* const G = dst_chw + 1 * plane;
    float* const R = dst_chw + 2 * plane;

    if (rw == bgr.width && rh == bgr.height) {
        for (int y = 0; y < rh; ++y) {
            const std::uint8_t* p = bgr.row(y);
            const std::size_t row = static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w);
            float* br = B + row;
            float* gr = G + row;
            float* rr = R + row;
            for (int x = 0; x < rw; ++x) {
                const int x3 = 3 * x;
                br[x] = (static_cast<float>(p[x3 + 0]) - mean[0]) * inv_std[0];
                gr[x] = (static_cast<float>(p[x3 + 1]) - mean[1]) * inv_std[1];
                rr[x] = (static_cast<float>(p[x3 + 2]) - mean[2]) * inv_std[2];
            }
        }
        return info;
    }

    const float x_scale = static_cast<float>(bgr.width) / static_cast<float>(rw);
    const float y_scale = static_cast<float>(bgr.height) / static_cast<float>(rh);

    constexpr int kParallelMinPixels = 64 * 64;
    bool parallel = (rw * rh) >= kParallelMinPixels;
#if defined(_OPENMP)
    parallel = parallel && !omp_in_parallel();
#endif

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (parallel)
#endif
    for (int y = 0; y < rh; ++y) {
        const float fy = (static_cast<float>(y) + 0.5f) * y_scale - 0.5f;
        const int y0 = std::max(0, std::min(static_cast<int>(std::floor(fy)), bgr.height - 1));
        const int y1 = std::min(y0 + 1, bgr.height - 1);
        const float wy = std::min(1.0f, std::max(0.0f, fy - static_cast<float>(y0)));

        const std::uint8_t* row0 = bgr.row(y0);
        const std::uint8_t* row1 = bgr.row(y1);
        const std::size_t out_row = static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w);
        float* br = B + out_row;
        float* gr = G + out_row;
        float* rr = R + out_row;

        for (int x = 0; x < rw; ++x) {
            const float fx = (static_cast<float>(x) + 0.5f) * x_scale - 0.5f;
            const int x0 = std::max(0, std::min(static_cast<int>(std::floor(fx)), bgr.width - 1));
            const int x1 = std::min(x0 + 1, bgr.width - 1);
            const float wx = std::min(1.0f, std::max(0.0f, fx - static_cast<float>(x0)));

            const std::uint8_t* p00 = row0 + static_cast<std::ptrdiff_t>(x0) * 3;
            const std::uint8_t* p01 = row0 + static_cast<std::ptrdiff_t>(x1) * 3;
            const std::uint8_t* p10 = row1 + static_cast<std::ptrdiff_t>(x0) * 3;
            const std::uint8_t* p11 = row1 + static_cast<std::ptrdiff_t>(x1) * 3;

            const float w00 = (1.0f - wx) * (1.0f - wy);
            const float w01 = wx * (1.0f - wy);
            const float w10 = (1.0f - wx) * wy;
            const float w11 = wx * wy;

            const float b = w00 * static_cast<float>(p00[0]) + w01 * static_cast<float>(p01[0]) +
                            w10 * static_cast<float>(p10[0]) + w11 * static_cast<float>(p11[0]);
            const float g = w00 * static_cast<float>(p00[1]) + w01 * static_cast<float>(p01[1]) +
                            w10 * static_cast<float>(p10[1]) + w11 * static_cast<float>(p11[1]);
            const float r = w00 * static_cast<float>(p00[2]) + w01 * static_cast<float>(p01[2]) +
                            w10 * static_cast<float>(p10[2]) + w11 * static_cast<float>(p11[2]);

            br[x] = (b - mean[0]) * inv_std[0];
            gr[x] = (g - mean[1]) * inv_std[1];
            rr[x] = (r - mean[2]) * inv_std[2];
        }
    }
    (void)parallel;
    return info;
}

} // namespace idet::internal
