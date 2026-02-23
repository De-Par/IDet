/**
 * @file cv_bgr.h
 * @ingroup idet_internal
 * @brief Utilities for converting @ref idet::Image into a BGR @c cv::Mat for OpenCV-based pipelines.
 *
 * Many computer vision pipelines (including common OpenCV preprocessing) operate on BGR images.
 * This header provides @ref idet::internal::BgrMat, a small wrapper that:
 * - exposes a @c cv::Mat in BGR format (@c CV_8UC3),
 * - preserves input image lifetime when the returned @c cv::Mat is a non-owning view into the
 *   original memory,
 * - performs color conversion via @c cv::cvtColor when needed.
 *
 * Typical usage:
 * @code
 * auto bgr = idet::internal::BgrMat::from(image);
 * if (!bgr.ok()) return idet::Result<std::vector<idet::Quad>>::Err(bgr.status());
 * const cv::Mat& mat = bgr.value().mat();
 * // Use `mat` as a BGR CV_8UC3 image for OpenCV preprocessing / inference input preparation.
 * @endcode
 *
 * Lifetime safety:
 * - If the returned @c cv::Mat is a view into the input image buffer, the input @ref idet::Image is
 *   stored inside @ref idet::internal::BgrMat to extend the lifetime of the backing memory.
 * - If conversion is required, the resulting @c cv::Mat owns its own buffer and the wrapper does
 *   not need to retain the input image.
 *
 * @note
 * This is an internal header and is not part of the stable public API.
 */

#pragma once

#include "image.h"
#include "internal/opencv_headers.h" // IWYU pragma: keep
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace idet::internal {

/**
 * @brief A small holder that provides a BGR @c cv::Mat and manages backing memory lifetime.
 *
 * @details
 * @ref BgrMat is returned by value via @ref from.
 *
 * Conversion behavior:
 * - If input format is already @ref idet::PixelFormat::BGR_U8, a @c cv::Mat view is created over
 *   the existing buffer (no copy). In this case, the input @ref idet::Image is stored inside the
 *   returned object to keep the backing memory alive.
 * - If input format is RGB/RGBA/BGRA U8, @c cv::cvtColor is used to produce a new BGR matrix.
 *
 * Output:
 * - @ref mat() always returns a @c cv::Mat with type @c CV_8UC3 representing BGR pixels on success.
 *
 * Constness note:
 * - The input image data is exposed as a @c cv::Mat by using a @c const_cast to satisfy OpenCV's API
 *   (which uses non-const pointers). Callers must treat the returned matrix as read-only when it
 *   aliases a const input buffer.
 *
 * @thread_safety
 * Instances are safe to read concurrently if not mutated. @c cv::Mat reference counting is
 * thread-safe for independent instances, but do not concurrently modify the same underlying pixel
 * memory without external synchronization.
 */
class BgrMat final {
  public:
    /** @brief Constructs an empty wrapper with an empty @c cv::Mat. */
    BgrMat() = default;

    /**
     * @brief Creates a BGR @c cv::Mat view/copy from an @ref idet::Image.
     *
     * @details
     * Validates the input view, then:
     * - fast path for @ref idet::PixelFormat::BGR_U8: create a @c cv::Mat view over the input buffer
     *   and store the image in @ref hold_ to extend its lifetime,
     * - otherwise: create a @c cv::Mat view for the source and convert via @c cv::cvtColor.
     *
     * @param img Input image. Passed by value to allow transferring/retaining ownership if needed.
     * @return @c Result<BgrMat> containing a valid wrapper on success, or an error status on failure.
     *
     * @retval Status::InvalidArgument if the input image view is invalid.
     * @retval Status::Unsupported if the pixel format is not supported for BGR conversion.
     * @retval Status::Internal if OpenCV conversion fails or an unexpected exception occurs.
     *
     * @note
     * When the fast path is used (already BGR), no pixel data is copied.
     * When conversion is required, a new @c cv::Mat buffer is allocated by OpenCV.
     *
     * @note
     * This function is declared @c noexcept and therefore catches exceptions thrown by OpenCV and
     * converts them into @ref Status::Internal.
     */
    [[nodiscard]] static idet::Result<BgrMat> from(idet::Image img) noexcept {
        const auto& v = img.view();
        if (!v.is_valid()) {
            return idet::Result<BgrMat>::Err(idet::Status::Invalid("BgrMat::from: invalid Image"));
        }

        BgrMat out;

        // Fast path: already BGR. Return a view and keep Image alive.
        if (v.format == idet::PixelFormat::BGR_U8) {
            out.hold_ = std::move(img);
            const auto& vv = out.hold_.view();

            out.mat_ = cv::Mat(vv.height, vv.width, CV_8UC3, const_cast<std::uint8_t*>(vv.data),
                               static_cast<std::size_t>(vv.stride_bytes));
            return idet::Result<BgrMat>::Ok(std::move(out));
        }

        const int ch = v.channels();
        if (ch != 3 && ch != 4) {
            return idet::Result<BgrMat>::Err(idet::Status::Unsupported("BgrMat::from: unsupported PixelFormat"));
        }

        const int src_type = (ch == 4) ? CV_8UC4 : CV_8UC3;

        // src is a view into the input Image memory (do not mutate through it).
        cv::Mat src(v.height, v.width, src_type, const_cast<std::uint8_t*>(v.data),
                    static_cast<std::size_t>(v.stride_bytes));

        const int code = cvt_code_to_bgr_(v.format);
        if (code < 0) {
            return idet::Result<BgrMat>::Err(
                idet::Status::Unsupported("BgrMat::from: unsupported PixelFormat for BGR conversion"));
        }

        try {
            cv::cvtColor(src, out.mat_, code);
        } catch (const cv::Exception& e) {
            return idet::Result<BgrMat>::Err(
                idet::Status::Internal(std::string("BgrMat::from: cvtColor failed: ") + e.what()));
        } catch (const std::exception& e) {
            return idet::Result<BgrMat>::Err(
                idet::Status::Internal(std::string("BgrMat::from: exception: ") + e.what()));
        } catch (...) {
            return idet::Result<BgrMat>::Err(idet::Status::Internal("BgrMat::from: unknown exception"));
        }

        return idet::Result<BgrMat>::Ok(std::move(out));
    }

    /**
     * @brief Returns the resulting BGR matrix.
     * @return Const reference to the stored @c cv::Mat (type is @c CV_8UC3 on success).
     */
    [[nodiscard]] const cv::Mat& mat() const noexcept {
        return mat_;
    }

  private:
    /**
     * @brief Returns OpenCV color conversion code to produce BGR from a given pixel format.
     *
     * @param f Input pixel format.
     * @return OpenCV conversion code (e.g., @c cv::COLOR_RGB2BGR) or -1 if unsupported.
     *
     * @note
     * @ref idet::PixelFormat::BGR_U8 is handled by the fast path and is not returned here.
     */
    [[nodiscard]] static int cvt_code_to_bgr_(idet::PixelFormat f) noexcept {
        using PF = idet::PixelFormat;
        switch (f) {
        case PF::RGB_U8:
            return cv::COLOR_RGB2BGR;
        case PF::RGBA_U8:
            return cv::COLOR_RGBA2BGR;
        case PF::BGRA_U8:
            return cv::COLOR_BGRA2BGR;
        default:
            // PF::BGR_U8 handled by fast path; others unsupported.
            return -1;
        }
    }

    /**
     * @brief Keeps backing memory alive when @ref mat_ is a view into the input image.
     *
     * This is non-empty only for the fast path (input already BGR) where @ref mat_ references the
     * input buffer. When conversion is performed via OpenCV, @ref mat_ owns its memory and
     * @ref hold_ typically remains empty.
     */
    idet::Image hold_{}; // keeps backing memory alive if mat_ is a view

    /** @brief Resulting BGR matrix. */
    cv::Mat mat_{};
};

} // namespace idet::internal
