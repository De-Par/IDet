/**
 * @file bgr_image.h
 * @ingroup idet_internal
 * @brief OpenCV-free BGR image view/holder used by the detector core.
 */

#pragma once

#include "image.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace idet::internal {

/**
 * @brief Non-owning BGR U8 image view.
 *
 * The view represents interleaved BGR pixels with 3 bytes per pixel and an arbitrary positive
 * row stride. It intentionally carries no dependency on OpenCV; adapters can wrap it into
 * backend-specific image objects at the edge of the implementation.
 */
struct BgrImageView {
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    std::ptrdiff_t stride_bytes = 0;

    [[nodiscard]] bool is_valid() const noexcept {
        if (data == nullptr || width <= 0 || height <= 0 || stride_bytes <= 0) return false;

        const auto max_offset = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
        const auto min_row = static_cast<std::size_t>(width) * 3U;
        if (min_row > max_offset) return false;

        const auto stride = static_cast<std::size_t>(stride_bytes);
        const auto rows_before_last = static_cast<std::size_t>(height - 1);
        if (rows_before_last != 0 && stride > max_offset / rows_before_last) return false;

        const auto last_row_offset = rows_before_last * stride;
        if (last_row_offset > max_offset - min_row) return false;

        return stride_bytes >= static_cast<std::ptrdiff_t>(min_row);
    }

    [[nodiscard]] const std::uint8_t* row(int y) const noexcept {
        return data + static_cast<std::ptrdiff_t>(y) * stride_bytes;
    }

    [[nodiscard]] BgrImageView roi(int x, int y, int w, int h) const noexcept {
        if (!is_valid() || x < 0 || y < 0 || w <= 0 || h <= 0) {
            return {};
        }
        if (x > width - w || y > height - h) {
            return {};
        }
        return {row(y) + static_cast<std::ptrdiff_t>(x) * 3, w, h, stride_bytes};
    }
};

/**
 * @brief Integer rectangle used by tiling without depending on cv::Rect.
 */
struct RectI {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;

    [[nodiscard]] bool empty() const noexcept {
        return width <= 0 || height <= 0;
    }

    friend bool operator==(const RectI& a, const RectI& b) noexcept {
        return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
    }

    friend bool operator!=(const RectI& a, const RectI& b) noexcept {
        return !(a == b);
    }
};

/**
 * @brief Owns or retains backing storage for a BGR image view.
 */
class BgrImage final {
  public:
    BgrImage() = default;

    [[nodiscard]] static idet::Result<BgrImage> from(idet::Image img) noexcept {
        const auto& v = img.view();
        if (!v.is_valid()) {
            return idet::Result<BgrImage>::Err(idet::Status::Invalid("BgrImage::from: invalid Image"));
        }

        BgrImage out;

        if (v.format == idet::PixelFormat::BGR_U8) {
            out.hold_ = std::move(img);
            const auto& vv = out.hold_.view();
            if (vv.stride_bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
                return idet::Result<BgrImage>::Err(idet::Status::Invalid("BgrImage::from: BGR stride too large"));
            }
            out.view_ = {vv.data, vv.width, vv.height, static_cast<std::ptrdiff_t>(vv.stride_bytes)};
            if (!out.view_.is_valid()) {
                return idet::Result<BgrImage>::Err(idet::Status::Invalid("BgrImage::from: invalid BGR view"));
            }
            return idet::Result<BgrImage>::Ok(std::move(out));
        }

        const int ch = v.channels();
        if ((v.format != idet::PixelFormat::RGB_U8 && v.format != idet::PixelFormat::RGBA_U8 &&
             v.format != idet::PixelFormat::BGRA_U8) ||
            (ch != 3 && ch != 4)) {
            return idet::Result<BgrImage>::Err(
                idet::Status::Unsupported("BgrImage::from: unsupported PixelFormat for BGR conversion"));
        }

        try {
            std::size_t row_bytes = 0;
            if (mul_overflow_(static_cast<std::size_t>(v.width), 3U, row_bytes)) {
                return idet::Result<BgrImage>::Err(idet::Status::Invalid("BgrImage::from: row size overflow"));
            }
            if (row_bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
                return idet::Result<BgrImage>::Err(idet::Status::Invalid("BgrImage::from: row size too large"));
            }

            std::size_t total_bytes = 0;
            if (mul_overflow_(row_bytes, static_cast<std::size_t>(v.height), total_bytes)) {
                return idet::Result<BgrImage>::Err(idet::Status::Invalid("BgrImage::from: image size overflow"));
            }
            if (total_bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
                return idet::Result<BgrImage>::Err(idet::Status::Invalid("BgrImage::from: image span too large"));
            }

            const auto max_offset = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
            const auto src_row_bytes = v.min_row_bytes();
            const auto rows_before_last = static_cast<std::size_t>(v.height - 1);
            if (rows_before_last != 0 && v.stride_bytes > max_offset / rows_before_last) {
                return idet::Result<BgrImage>::Err(idet::Status::Invalid("BgrImage::from: source stride overflow"));
            }
            const auto last_row_offset = rows_before_last * v.stride_bytes;
            if (src_row_bytes > max_offset || last_row_offset > max_offset - src_row_bytes) {
                return idet::Result<BgrImage>::Err(idet::Status::Invalid("BgrImage::from: source span too large"));
            }

            const auto src_stride = static_cast<std::ptrdiff_t>(v.stride_bytes);
            const auto src_channels = static_cast<std::ptrdiff_t>(ch);

            out.owned_.resize(total_bytes);
            for (int y = 0; y < v.height; ++y) {
                const std::uint8_t* src = v.data + static_cast<std::ptrdiff_t>(y) * src_stride;
                std::uint8_t* dst = out.owned_.data() + static_cast<std::size_t>(y) * row_bytes;
                for (int x = 0; x < v.width; ++x) {
                    const auto src_offset = static_cast<std::ptrdiff_t>(x) * src_channels;
                    const auto dst_offset = static_cast<std::ptrdiff_t>(x) * std::ptrdiff_t{3};
                    const std::uint8_t* p = src + src_offset;
                    std::uint8_t* q = dst + dst_offset;
                    switch (v.format) {
                    case idet::PixelFormat::RGB_U8:
                    case idet::PixelFormat::RGBA_U8:
                        q[0] = p[2];
                        q[1] = p[1];
                        q[2] = p[0];
                        break;
                    case idet::PixelFormat::BGRA_U8:
                        q[0] = p[0];
                        q[1] = p[1];
                        q[2] = p[2];
                        break;
                    default:
                        return idet::Result<BgrImage>::Err(
                            idet::Status::Unsupported("BgrImage::from: unsupported PixelFormat"));
                    }
                }
            }
        } catch (const std::bad_alloc&) {
            return idet::Result<BgrImage>::Err(idet::Status::OutOfMemory("BgrImage::from: bad_alloc"));
        } catch (const std::exception& e) {
            return idet::Result<BgrImage>::Err(
                idet::Status::Internal(std::string("BgrImage::from: exception: ") + e.what()));
        } catch (...) {
            return idet::Result<BgrImage>::Err(idet::Status::Internal("BgrImage::from: unknown exception"));
        }

        out.view_ = {out.owned_.data(), v.width, v.height, static_cast<std::ptrdiff_t>(v.width) * 3};
        return idet::Result<BgrImage>::Ok(std::move(out));
    }

    [[nodiscard]] const BgrImageView& view() const noexcept {
        return view_;
    }

  private:
    [[nodiscard]] static bool mul_overflow_(std::size_t a, std::size_t b, std::size_t& out) noexcept {
        if (a == 0 || b == 0) {
            out = 0;
            return false;
        }
        if (a > std::numeric_limits<std::size_t>::max() / b) return true;
        out = a * b;
        return false;
    }

    idet::Image hold_{};
    std::vector<std::uint8_t> owned_{};
    BgrImageView view_{};
};

} // namespace idet::internal
