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
        return data != nullptr && width > 0 && height > 0 && stride_bytes >= static_cast<std::ptrdiff_t>(width * 3);
    }

    [[nodiscard]] const std::uint8_t* row(int y) const noexcept {
        return data + static_cast<std::ptrdiff_t>(y) * stride_bytes;
    }

    [[nodiscard]] BgrImageView roi(int x, int y, int w, int h) const noexcept {
        if (!is_valid() || x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > width || y + h > height) {
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
            out.view_ = {vv.data, vv.width, vv.height, static_cast<std::ptrdiff_t>(vv.stride_bytes)};
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
            out.owned_.resize(static_cast<std::size_t>(v.width) * static_cast<std::size_t>(v.height) * 3U);
            for (int y = 0; y < v.height; ++y) {
                const std::uint8_t* src = v.data + static_cast<std::ptrdiff_t>(y) * v.stride_bytes;
                std::uint8_t* dst =
                    out.owned_.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(v.width) * 3U;
                for (int x = 0; x < v.width; ++x) {
                    const std::uint8_t* p = src + static_cast<std::ptrdiff_t>(x) * ch;
                    std::uint8_t* q = dst + static_cast<std::ptrdiff_t>(x) * 3;
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

        out.view_ = {out.owned_.data(), v.width, v.height, static_cast<std::ptrdiff_t>(v.width * 3)};
        return idet::Result<BgrImage>::Ok(std::move(out));
    }

    [[nodiscard]] const BgrImageView& view() const noexcept {
        return view_;
    }

  private:
    idet::Image hold_{};
    std::vector<std::uint8_t> owned_{};
    BgrImageView view_{};
};

} // namespace idet::internal
