/**
 * @file image.cpp
 * @brief Implementation of image loading and deep-copy utilities for IDet.
 *
 * @details
 * This translation unit implements:
 * - @ref idet::Image::copy_from        : deep copy into an owning buffer.
 * - @ref idet::load_image              : image decoding from disk (stb_image) and format conversion.
 * - @ref idet::load_image_or_throw     : throwing convenience wrapper.
 *
 * Decoding is performed via stb_image (@c stbi_load). The implementation:
 * - requests 3 or 4 channels from stb_image depending on @ref idet::PixelFormat,
 * - optionally flips the decoded image vertically,
 * - optionally swaps R/B channels to produce BGR/BGRA formats,
 * - adopts the stb_image buffer into an @ref idet::Image using a custom deleter.
 *
 * @note
 * This file defines @c STB_IMAGE_IMPLEMENTATION, meaning it provides stb_image's
 * implementation. Ensure it is compiled into exactly one translation unit.
 */

#include "image.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__has_include) && __has_include(<stb_image.h>)
    #define STB_IMAGE_IMPLEMENTATION
    #include <stb_image.h>
#elif defined(__has_include) && __has_include(<stb/stb_image.h>)
    #define STB_IMAGE_IMPLEMENTATION
    #include <stb/stb_image.h>
#else
    #error "[ERROR] 'stb_image.h' header not found"
#endif

namespace idet {

namespace {

/**
 * @brief Returns whether the requested output format expects BGR channel order.
 *
 * @param fmt Desired output format.
 * @return True for @ref idet::PixelFormat::BGR_U8 and @ref idet::PixelFormat::BGRA_U8.
 */
[[nodiscard]] constexpr bool wants_bgr(PixelFormat fmt) noexcept {
    return fmt == PixelFormat::BGR_U8 || fmt == PixelFormat::BGRA_U8;
}

/**
 * @brief Returns desired channel count to request from stb_image for a given output format.
 *
 * @details
 * stb_image can be asked to return an explicit channel count:
 * - 4 for RGBA/BGRA outputs
 * - 3 for RGB/BGR outputs
 *
 * @param fmt Desired output format.
 * @return 3 or 4 depending on @p fmt.
 */
[[nodiscard]] constexpr int desired_channels(PixelFormat fmt) noexcept {
    const int ch = get_channels(fmt);
    return (ch == 4) ? 4 : 3;
}

/**
 * @brief Multiplies two size_t values with overflow detection.
 *
 * @details
 * Uses @c __builtin_mul_overflow when available; otherwise falls back to a portable check.
 *
 * @param a First multiplicand.
 * @param b Second multiplicand.
 * @param out Output pointer receiving @c a*b when no overflow occurs.
 * @return True if overflow occurred, false otherwise.
 */
[[nodiscard]] bool mul_overflow_size(std::size_t a, std::size_t b, std::size_t* out) noexcept {
#if defined(__has_builtin)
    #if __has_builtin(__builtin_mul_overflow)
    return __builtin_mul_overflow(a, b, out);
    #endif
#endif
    if (a == 0 || b == 0) {
        *out = 0;
        return false;
    }
    if (a > (static_cast<std::size_t>(-1) / b)) return true;
    *out = a * b;
    return false;
}

/**
 * @brief Swaps R and B channels in-place for an interleaved U8 image buffer.
 *
 * @details
 * Assumes each pixel has at least 3 channels and performs:
 * - swap(px[0], px[2]) for each pixel, leaving other channels intact.
 *
 * @param data Pointer to the first byte of the image buffer.
 * @param w Image width in pixels.
 * @param h Image height in pixels.
 * @param ch Channel count per pixel (must be >= 3).
 */
void swap_rb_in_place(std::uint8_t* data, int w, int h, int ch) noexcept {
    const std::size_t row_bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(ch);
    for (int y = 0; y < h; ++y) {
        std::uint8_t* row = data + static_cast<std::size_t>(y) * row_bytes;
        for (int x = 0; x < w; ++x) {
            std::uint8_t* px = row + static_cast<std::size_t>(x) * static_cast<std::size_t>(ch);
            std::swap(px[0], px[2]);
        }
    }
}

/**
 * @brief Flips an interleaved U8 image buffer vertically (in-place).
 *
 * @details
 * Swaps rows from top to bottom using a temporary row buffer of size @c w*ch.
 * This function allocates a temporary row buffer.
 *
 * @param data Pointer to the first byte of the image buffer.
 * @param w Image width in pixels.
 * @param h Image height in pixels.
 * @param ch Channel count per pixel.
 *
 * @throws std::bad_alloc if the temporary row allocation fails.
 */
void flip_y_in_place(std::uint8_t* data, int w, int h, int ch) {
    const std::size_t row_bytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(ch);
    std::vector<std::uint8_t> tmp(row_bytes);

    for (int y = 0; y < h / 2; ++y) {
        std::uint8_t* top = data + static_cast<std::size_t>(y) * row_bytes;
        std::uint8_t* bot = data + static_cast<std::size_t>(h - 1 - y) * row_bytes;

        std::memcpy(tmp.data(), top, row_bytes);
        std::memcpy(top, bot, row_bytes);
        std::memcpy(bot, tmp.data(), row_bytes);
    }
}

} // namespace

/**
 * @brief Deep-copies pixel data into an Image-owned buffer.
 *
 * @details
 * - Validates the input as a proper @ref idet::ImageView.
 * - Computes the destination stride as @c w*channels.
 * - Allocates a contiguous buffer sized @c dst_stride*h.
 * - Copies rows from the source buffer honoring @p src_stride_bytes.
 *
 * Ownership model:
 * The returned @ref idet::Image adopts the allocated @c std::vector<std::uint8_t> via
 * a shared owner token, ensuring pixel memory remains valid for the lifetime of
 * the Image instance (and any copies).
 *
 * Supported formats:
 * - 3-channel and 4-channel packed U8 formats only (e.g., RGB/BGR/RGBA/BGRA variants).
 *
 * @param fmt Pixel format.
 * @param w Width in pixels.
 * @param h Height in pixels.
 * @param src Pointer to the source pixel buffer.
 * @param src_stride_bytes Source row stride in bytes.
 * @return `Result<Image>` containing an owning Image on success, or an error status on failure.
 */
Result<Image> Image::copy_from(PixelFormat fmt, int w, int h, const std::uint8_t* src,
                               std::size_t src_stride_bytes) noexcept {
    ImageView in;
    in.data = src;
    in.width = w;
    in.height = h;
    in.stride_bytes = src_stride_bytes;
    in.format = fmt;

    if (!in.is_valid()) {
        return Result<Image>::Err(Status::Invalid("Image::copy_from: invalid input"));
    }

    const int ch = get_channels(fmt);
    if (ch != 3 && ch != 4) {
        return Result<Image>::Err(Status::Unsupported("Image::copy_from: unsupported PixelFormat"));
    }

    std::size_t dst_stride = 0;
    if (mul_overflow_size(static_cast<std::size_t>(w), static_cast<std::size_t>(ch), &dst_stride)) {
        return Result<Image>::Err(Status::Invalid("Image::copy_from: size overflow (stride)"));
    }

    std::size_t total = 0;
    if (mul_overflow_size(dst_stride, static_cast<std::size_t>(h), &total)) {
        return Result<Image>::Err(Status::Invalid("Image::copy_from: size overflow (total)"));
    }

    try {
        auto buf = std::make_shared<std::vector<std::uint8_t>>(total);

        for (int y = 0; y < h; ++y) {
            const auto* row_src = src + static_cast<std::size_t>(y) * src_stride_bytes;
            auto* row_dst = buf->data() + static_cast<std::size_t>(y) * dst_stride;
            std::memcpy(row_dst, row_src, dst_stride);
        }

        Image out = Image::wrap(fmt, w, h, buf->data(), dst_stride, std::static_pointer_cast<void>(buf));
        if (!out) {
            return Result<Image>::Err(Status::Internal("Image::copy_from: output invalid"));
        }
        return Result<Image>::Ok(std::move(out));

    } catch (const std::bad_alloc&) {
        return Result<Image>::Err(Status::OutOfMemory("Image::copy_from: allocation failed"));
    } catch (...) {
        return Result<Image>::Err(Status::Internal("Image::copy_from: unknown exception"));
    }
}

/**
 * @brief Loads an image from disk and returns it as an @ref idet::Image.
 *
 * @details
 * Decoding is performed by stb_image. The function:
 * - validates the requested @p output_format (must be 3 or 4 channels),
 * - decodes the image into an stb-owned buffer (U8 interleaved),
 * - optionally flips the image vertically,
 * - optionally swaps R/B channels to output BGR/BGRA,
 * - adopts the stb buffer into an @ref idet::Image with a custom deleter.
 *
 * The returned image is tightly packed:
 * @code
 * stride_bytes = width * requested_channels
 * @endcode
 *
 * @param path Filesystem path to the input image file.
 * @param output_format Desired output pixel format.
 * @param flip_y If true, performs a vertical flip (top-bottom) after decode.
 * @return `Result<Image>` with an owning image on success, or an error status on failure.
 */
IDET_API Result<Image> load_image(const std::string& path, PixelFormat output_format, bool flip_y) noexcept {
    const int out_ch = get_channels(output_format);
    if (out_ch != 3 && out_ch != 4) {
        return Result<Image>::Err(Status::Unsupported("load_image: unsupported output PixelFormat"));
    }

    const int req_ch = desired_channels(output_format);

    int w = 0, h = 0, n = 0;
    stbi_uc* px = stbi_load(path.c_str(), &w, &h, &n, req_ch);
    if (!px) {
        return Result<Image>::Err(Status::DecodeError(std::string("stbi_load failed: ") + stbi_failure_reason()));
    }

    const std::size_t stride = static_cast<std::size_t>(w) * static_cast<std::size_t>(req_ch);

    try {
        if (flip_y) {
            flip_y_in_place(reinterpret_cast<std::uint8_t*>(px), w, h, req_ch);
        }
    } catch (const std::bad_alloc&) {
        stbi_image_free(px);
        return Result<Image>::Err(Status::OutOfMemory("load_image: flip_y temp allocation failed"));
    } catch (...) {
        stbi_image_free(px);
        return Result<Image>::Err(Status::Internal("load_image: flip_y failed"));
    }

    if (wants_bgr(output_format)) {
        swap_rb_in_place(reinterpret_cast<std::uint8_t*>(px), w, h, req_ch);
    }

    Image img = Image::adopt(output_format, w, h, reinterpret_cast<std::uint8_t*>(px), stride,
                             [](void* p) { stbi_image_free(p); });

    if (!img) {
        // If this happens, treat it as a logic/internal error; also avoid leaking 'px'.
        stbi_image_free(px);
        return Result<Image>::Err(Status::Internal("load_image: invalid Image after adopt"));
    }

    return Result<Image>::Ok(std::move(img));
}

/**
 * @brief Loads an image from disk or throws on failure.
 *
 * @details
 * Calls @ref idet::load_image and throws @c std::runtime_error if loading fails.
 *
 * @param path Filesystem path to the input image file.
 * @param output_format Desired output pixel format.
 * @param flip_y If true, performs a vertical flip (top-bottom) after decode.
 * @return Owning @ref idet::Image on success.
 *
 * @throws std::runtime_error if @ref idet::load_image fails.
 */
IDET_API Image load_image_or_throw(const std::string& path, PixelFormat output_format, bool flip_y) {
    auto r = load_image(path, output_format, flip_y);
    if (!r.ok()) {
        throw std::runtime_error(r.status().message);
    }
    return std::move(r.value());
}

} // namespace idet
