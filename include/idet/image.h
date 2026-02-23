/**
 * @file image.h
 * @brief Lightweight image container and non-owning view utilities for IDet.
 *
 * This header provides:
 * - @ref idet::PixelFormat — a small enum describing packed 8-bit interleaved pixel layouts.
 * - @ref idet::ImageView   — a non-owning view over image memory (pointer + geometry + stride).
 * - @ref idet::Image       — a small value-type wrapper that may optionally share ownership of the
 *   backing memory via a lifetime token.
 *
 * The design supports multiple lifetime models:
 *  1) Non-owning view (caller-managed lifetime).
 *  2) View + external shared owner (shared lifetime).
 *  3) Adopt a raw pointer with a custom deleter (RAII via shared_ptr<void>).
 *  4) Deep copy into an Image-managed buffer.
 *
 * Terminology:
 * - `stride_bytes` is the number of bytes between the start of two consecutive rows in memory.
 * - For tightly packed interleaved 8-bit images, it is typically `width * channels`.
 *
 * @warning
 * @ref idet::ImageView::min_row_bytes() assumes 1 byte per channel (U8). If additional bit-depths
 * (e.g., U16/F16/F32) are introduced later, the API should define bytes-per-channel explicitly.
 */

/**
 * @defgroup idet_image Image
 * @brief Image views, ownership wrappers, and image loading helpers.
 * @{
 */

#pragma once

#include "export.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace idet {

/**
 * @ingroup idet_image
 * @brief Supported packed pixel formats (interleaved channels, 8-bit per channel).
 *
 * All formats are assumed to be tightly interleaved per pixel (e.g., RGBRGB...).
 * Planar layouts are not represented by this enum.
 *
 * The underlying storage type is @c std::uint8_t for compactness and ABI stability.
 */
enum class PixelFormat : std::uint8_t {
    /** Packed RGB, 8-bit per channel, 3 channels per pixel. */
    RGB_U8 = 0,
    /** Packed BGR, 8-bit per channel, 3 channels per pixel. */
    BGR_U8 = 1,
    /** Packed RGBA, 8-bit per channel, 4 channels per pixel. */
    RGBA_U8 = 2,
    /** Packed BGRA, 8-bit per channel, 4 channels per pixel. */
    BGRA_U8 = 3,
};

/**
 * @ingroup idet_image
 * @brief Returns the number of interleaved channels for a given @ref PixelFormat.
 *
 * @param f Pixel format.
 * @return Number of channels (3 or 4). Returns 0 for unknown values.
 *
 * @note
 * This function is @c constexpr and can be used in compile-time contexts.
 */
[[nodiscard]] constexpr int get_channels(PixelFormat f) noexcept {
    switch (f) {
    case PixelFormat::RGB_U8:
    case PixelFormat::BGR_U8:
        return 3;
    case PixelFormat::RGBA_U8:
    case PixelFormat::BGRA_U8:
        return 4;
    default:
        return 0;
    }
}

/**
 * @ingroup idet_image
 * @brief A non-owning view over packed 8-bit image memory.
 *
 * @details
 * @ref ImageView does not manage memory. It only describes how to interpret a memory region:
 * - @ref data points to the first byte of the first row.
 * - @ref width / @ref height define image dimensions in pixels.
 * - @ref stride_bytes is the number of bytes between consecutive row starts.
 * - @ref format defines channel order and channel count.
 *
 * Validity rules (see @ref is_valid()):
 * - @ref data is not null
 * - @ref width > 0 and @ref height > 0
 * - @ref stride_bytes >= @ref min_row_bytes() for U8 formats
 *
 * @note
 * This view is read-only because @ref data is a pointer to const bytes. If a mutable view is
 * needed, introduce a separate type to avoid accidental writes to shared buffers.
 *
 * @note
 * This type is cheap to copy and is intended to be passed by value when convenient.
 */
struct ImageView final {
    /**
     * @brief Pointer to the first byte of the first row.
     *
     * @warning
     * Lifetime is not managed by @ref ImageView. Ensure the memory remains valid for the entire
     * duration of any use of this view.
     */
    const std::uint8_t* data = nullptr;

    /** @brief Image width in pixels. Must be > 0 for a non-empty view. */
    int width = 0;

    /** @brief Image height in pixels. Must be > 0 for a non-empty view. */
    int height = 0;

    /**
     * @brief Row stride in bytes (distance between the start of adjacent rows).
     *
     * For tightly packed U8 images this is commonly `width * channels`, but may be larger due
     * to alignment/padding.
     */
    std::size_t stride_bytes = 0;

    /** @brief Pixel format describing channel order and channel count. */
    PixelFormat format = PixelFormat::RGB_U8;

    /**
     * @brief Checks whether the view is empty (no data or non-positive dimensions).
     * @return True if @ref data is null or @ref width <= 0 or @ref height <= 0.
     */
    [[nodiscard]] constexpr bool empty() const noexcept {
        return data == nullptr || width <= 0 || height <= 0;
    }

    /**
     * @brief Returns the number of interleaved channels for @ref format.
     * @return Channel count (3/4), or 0 for unknown formats.
     */
    [[nodiscard]] constexpr int channels() const noexcept {
        return get_channels(format);
    }

    /**
     * @brief Returns the minimum number of bytes required to store one row.
     *
     * Computed as `width * channels` for currently supported 8-bit packed formats.
     *
     * @return Minimum bytes per row, or 0 if width/channels are invalid.
     */
    [[nodiscard]] constexpr std::size_t min_row_bytes() const noexcept {
        const int ch = channels();
        if (ch <= 0 || width <= 0) return 0;
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(ch);
    }

    /**
     * @brief Validates the view invariants (pointer, dimensions, stride).
     * @return True if the view is non-empty and `stride_bytes >= min_row_bytes()`.
     */
    [[nodiscard]] constexpr bool is_valid() const noexcept {
        if (empty()) return false;
        const std::size_t min_row = min_row_bytes();
        return (min_row > 0) && (stride_bytes >= min_row);
    }

    /**
     * @brief Checks whether the image rows are tightly packed (no padding).
     *
     * A tightly packed view satisfies `stride_bytes == min_row_bytes()`.
     *
     * @return True if valid and tightly packed, otherwise false.
     */
    [[nodiscard]] constexpr bool tightly_packed() const noexcept {
        if (!is_valid()) return false;
        return stride_bytes == min_row_bytes();
    }
};

/**
 * @ingroup idet_image
 * @brief Image wrapper that may optionally share ownership of the underlying memory.
 *
 * @details
 * @ref Image is a small value-type composed of:
 * - an @ref ImageView describing the pixel memory, and
 * - an optional @c std::shared_ptr<void> owner token for lifetime management.
 *
 * Ownership models:
 * - **Non-owning**: created via @ref view; @ref owner() is empty.
 * - **Externally owned**: created via @ref wrap; @ref owner() keeps external memory alive.
 * - **Adopted**: created via @ref adopt; the shared owner runs a custom deleter.
 * - **Deep copy**: created via @ref copy_from; the returned Image manages its own buffer.
 *
 * @note
 * Copying @ref Image is cheap (copies the view and shared_ptr token).
 *
 * @thread_safety
 * This type is safe to copy and read concurrently. It does not synchronize access to the
 * underlying pixel memory. If multiple threads may modify the backing buffer, the program
 * must provide external synchronization.
 */
class Image final {
  public:
    /** @brief Constructs an empty image (invalid view, no owner). */
    Image() noexcept = default;

    /**
     * @brief Creates a non-owning image view.
     *
     * The caller must guarantee that the memory referenced by @p v remains valid for the entire
     * time the returned @ref Image (and any of its copies) is used.
     *
     * @param v Image view descriptor.
     * @return Image that references @p v and does not own memory.
     *
     * @note
     * Use @ref ImageView::is_valid() (or `if (img)`) to validate before use.
     */
    [[nodiscard]] static Image view(ImageView v) noexcept {
        Image img;
        img.view_ = v;
        return img;
    }

    /**
     * @brief Wraps a view together with a shared lifetime owner token.
     *
     * The @p owner token keeps the referenced memory alive as long as the returned @ref Image
     * (and any copies of it) exist.
     *
     * @param v Image view descriptor.
     * @param owner Shared lifetime token associated with the pixel memory.
     * @return Image that references @p v and shares ownership via @p owner.
     */
    [[nodiscard]] static Image wrap(ImageView v, std::shared_ptr<void> owner) noexcept {
        Image img;
        img.view_ = v;
        img.owner_ = std::move(owner);
        return img;
    }

    /**
     * @brief Convenience overload to wrap raw parameters into an @ref ImageView and owner token.
     *
     * @param fmt Pixel format.
     * @param w Width in pixels.
     * @param h Height in pixels.
     * @param data Pointer to the first byte of the first row.
     * @param stride_bytes Row stride in bytes.
     * @param owner Shared lifetime token (may be empty for non-owning semantics).
     * @return Image referencing the provided memory and sharing ownership via @p owner.
     *
     * @warning
     * If @p owner is empty, this behaves like a non-owning view; the caller must ensure the
     * lifetime of @p data.
     */
    [[nodiscard]] static Image wrap(PixelFormat fmt, int w, int h, const std::uint8_t* data, std::size_t stride_bytes,
                                    std::shared_ptr<void> owner = {}) noexcept {
        ImageView v;
        v.data = data;
        v.width = w;
        v.height = h;
        v.stride_bytes = stride_bytes;
        v.format = fmt;
        return wrap(v, std::move(owner));
    }

    /**
     * @brief Adopts a raw pointer with a user-provided deleter (RAII via shared_ptr<void>).
     *
     * Creates a shared owner token that will call @p deleter once the last copy of the returned
     * @ref Image is destroyed.
     *
     * @tparam Deleter Deleter type. Must be CopyConstructible.
     * @param fmt Pixel format.
     * @param w Width in pixels.
     * @param h Height in pixels.
     * @param data Pointer to pixel memory to adopt.
     * @param stride_bytes Row stride in bytes.
     * @param deleter Callable invoked with the raw pointer (as @c void*) when ownership is released.
     * @return Image that references @p data and owns it via a shared deleter token.
     *
     * @note
     * The deleter must be CopyConstructible because @c std::shared_ptr stores a copy of it.
     *
     * @warning
     * The deleter is invoked with a @c void*. Ensure your deleter can accept @c void*, or provide
     * a wrapper that performs the appropriate cast.
     */
    template <class Deleter>
    [[nodiscard]] static Image adopt(PixelFormat fmt, int w, int h, std::uint8_t* data, std::size_t stride_bytes,
                                     Deleter deleter) {
        std::shared_ptr<void> owner(data, [deleter](void* p) mutable { deleter(p); });
        return wrap(fmt, w, h, data, stride_bytes, std::move(owner));
    }

    /**
     * @brief Deep-copies pixel data into an Image-managed buffer.
     *
     * Allocates an internal buffer and copies @p h rows from @p src into it, honoring the source
     * stride. The returned image is safe to use after the source buffer is freed.
     *
     * @param fmt Pixel format.
     * @param w Width in pixels.
     * @param h Height in pixels.
     * @param src Source pointer to the first byte of the first row.
     * @param src_stride_bytes Source row stride in bytes.
     * @return Result containing an owning Image on success, or an error status on failure.
     */
    [[nodiscard]] static Result<Image> copy_from(PixelFormat fmt, int w, int h, const std::uint8_t* src,
                                                 std::size_t src_stride_bytes) noexcept;

    /**
     * @brief Returns the underlying image view descriptor.
     * @return Const reference to the stored @ref ImageView.
     */
    [[nodiscard]] const ImageView& view() const noexcept {
        return view_;
    }

    /**
     * @brief Returns the shared owner token (may be empty).
     *
     * If non-empty, the owner token keeps the underlying pixel memory alive.
     *
     * @return Const reference to the shared owner token.
     */
    [[nodiscard]] const std::shared_ptr<void>& owner() const noexcept {
        return owner_;
    }

    /**
     * @brief Checks whether the image view is valid.
     *
     * Equivalent to `view().is_valid()`.
     *
     * @return True if the view is valid, otherwise false.
     */
    [[nodiscard]] explicit operator bool() const noexcept {
        return view_.is_valid();
    }

  private:
    /** @brief Stored view descriptor (may be invalid for default-constructed Image). */
    ImageView view_{};

    /**
     * @brief Optional lifetime token for pixel memory.
     *
     * - Empty for non-owning views.
     * - Non-empty for wrapped/adopted/copied images.
     */
    std::shared_ptr<void> owner_{};
};

/**
 * @ingroup idet_image
 * @brief Loads an image from disk into an @ref Image.
 *
 * Reads the file at @p path, decodes it, optionally flips vertically, and converts to
 * @p output_format.
 *
 * @param path Filesystem path to the image.
 * @param output_format Desired output pixel format.
 * @param flip_y If true, the output image is flipped vertically (top-bottom).
 * @return Result with an Image on success, or an error status on failure.
 *
 * @note
 * On success, the returned Image is expected to keep the decoded pixel buffer alive (owned
 * internally or via a lifetime token).
 */
[[nodiscard]] IDET_API idet::Result<Image> load_image(const std::string& path, PixelFormat output_format,
                                                      bool flip_y = false) noexcept;

/**
 * @ingroup idet_image
 * @brief Loads an image from disk and throws on failure.
 *
 * Convenience wrapper over @ref load_image that converts failures into an exception.
 *
 * @param path Filesystem path to the image.
 * @param output_format Desired output pixel format.
 * @param flip_y If true, the output image is flipped vertically (top-bottom).
 * @return Image on success.
 *
 * @throws
 * Throws an exception on load/decode/convert failure (exception type is implementation-defined).
 */
IDET_API Image load_image_or_throw(const std::string& path, PixelFormat output_format, bool flip_y = false);

} // namespace idet

/** @} */ // end of group idet_image
