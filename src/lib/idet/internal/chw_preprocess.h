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

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define IDET_CHW_PREPROCESS_HAS_NEON 1
#endif

#if defined(__SSSE3__) && defined(__SSE4_1__)
    #include <immintrin.h>
    #define IDET_CHW_PREPROCESS_HAS_SSE41 1
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

inline void bgr_row_to_chw_scalar_(const std::uint8_t* src, int width, int x0, float* dst_b, float* dst_g, float* dst_r,
                                   float mean_b, float mean_g, float mean_r, float inv_b, float inv_g,
                                   float inv_r) noexcept {
    for (int x = x0; x < width; ++x) {
        const int x3 = 3 * x;
        dst_b[x] = (static_cast<float>(src[x3 + 0]) - mean_b) * inv_b;
        dst_g[x] = (static_cast<float>(src[x3 + 1]) - mean_g) * inv_g;
        dst_r[x] = (static_cast<float>(src[x3 + 2]) - mean_r) * inv_r;
    }
}

#if defined(IDET_CHW_PREPROCESS_HAS_NEON)
inline void store_u8x8_to_f32_neon_(uint8x8_t v, float* dst, float32x4_t mean, float32x4_t inv_std) noexcept {
    const uint16x8_t u16 = vmovl_u8(v);
    const float32x4_t lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16)));
    const float32x4_t hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16)));

    vst1q_f32(dst + 0, vmulq_f32(vsubq_f32(lo, mean), inv_std));
    vst1q_f32(dst + 4, vmulq_f32(vsubq_f32(hi, mean), inv_std));
}
#endif

#if defined(IDET_CHW_PREPROCESS_HAS_SSE41)
inline void store_u8x4_to_f32_sse41_(__m128i v, float* dst, __m128 mean, __m128 inv_std) noexcept {
    const __m128 f = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(v));
    _mm_storeu_ps(dst, _mm_mul_ps(_mm_sub_ps(f, mean), inv_std));
}
#endif

inline int bgr_row_to_chw_simd_(const std::uint8_t* src, int width, float* dst_b, float* dst_g, float* dst_r,
                                float mean_b, float mean_g, float mean_r, float inv_b, float inv_g,
                                float inv_r) noexcept {
#if defined(IDET_CHW_PREPROCESS_HAS_NEON)
    const float32x4_t mb = vdupq_n_f32(mean_b);
    const float32x4_t mg = vdupq_n_f32(mean_g);
    const float32x4_t mr = vdupq_n_f32(mean_r);
    const float32x4_t sb = vdupq_n_f32(inv_b);
    const float32x4_t sg = vdupq_n_f32(inv_g);
    const float32x4_t sr = vdupq_n_f32(inv_r);

    int x = 0;
    for (; x + 8 <= width; x += 8) {
        const uint8x8x3_t pix = vld3_u8(src + static_cast<std::ptrdiff_t>(x) * 3);
        store_u8x8_to_f32_neon_(pix.val[0], dst_b + x, mb, sb);
        store_u8x8_to_f32_neon_(pix.val[1], dst_g + x, mg, sg);
        store_u8x8_to_f32_neon_(pix.val[2], dst_r + x, mr, sr);
    }
    return x;
#elif defined(IDET_CHW_PREPROCESS_HAS_SSE41)
    const __m128 mb = _mm_set1_ps(mean_b);
    const __m128 mg = _mm_set1_ps(mean_g);
    const __m128 mr = _mm_set1_ps(mean_r);
    const __m128 sb = _mm_set1_ps(inv_b);
    const __m128 sg = _mm_set1_ps(inv_g);
    const __m128 sr = _mm_set1_ps(inv_r);

    constexpr char z = static_cast<char>(-128);
    const __m128i bmask = _mm_setr_epi8(0, 3, 6, 9, z, z, z, z, z, z, z, z, z, z, z, z);
    const __m128i gmask = _mm_setr_epi8(1, 4, 7, 10, z, z, z, z, z, z, z, z, z, z, z, z);
    const __m128i rmask = _mm_setr_epi8(2, 5, 8, 11, z, z, z, z, z, z, z, z, z, z, z, z);

    int x = 0;
    // The 16-byte load touches 4 extra bytes after the 4 converted pixels, so keep two
    // source pixels of headroom and let the scalar tail handle the row end.
    for (; x + 6 <= width; x += 4) {
        const __m128i pix = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + static_cast<std::ptrdiff_t>(x) * 3));
        store_u8x4_to_f32_sse41_(_mm_shuffle_epi8(pix, bmask), dst_b + x, mb, sb);
        store_u8x4_to_f32_sse41_(_mm_shuffle_epi8(pix, gmask), dst_g + x, mg, sg);
        store_u8x4_to_f32_sse41_(_mm_shuffle_epi8(pix, rmask), dst_r + x, mr, sr);
    }
    return x;
#else
    (void)src;
    (void)width;
    (void)dst_b;
    (void)dst_g;
    (void)dst_r;
    (void)mean_b;
    (void)mean_g;
    (void)mean_r;
    (void)inv_b;
    (void)inv_g;
    (void)inv_r;
    return 0;
#endif
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
        const int x0 = detail::bgr_row_to_chw_simd_(p, W, br, gr, rr, mB, mG, mR, sB, sG, sR);
        detail::bgr_row_to_chw_scalar_(p, W, x0, br, gr, rr, mB, mG, mR, sB, sG, sR);
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
            const int x0 = detail::bgr_row_to_chw_simd_(p, rw, br, gr, rr, mean[0], mean[1], mean[2], inv_std[0],
                                                        inv_std[1], inv_std[2]);
            detail::bgr_row_to_chw_scalar_(p, rw, x0, br, gr, rr, mean[0], mean[1], mean[2], inv_std[0], inv_std[1],
                                           inv_std[2]);
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
