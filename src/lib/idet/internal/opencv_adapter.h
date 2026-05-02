/**
 * @file opencv_adapter.h
 * @ingroup idet_internal
 * @brief OpenCV adapters for core image/geometry types.
 */

#pragma once

#include "idet.h"
#include "internal/bgr_image.h"
#include "internal/opencv_headers.h" // IWYU pragma: keep

#include <array>

namespace idet::internal::opencv_adapter {

/**
 * @brief Wrap a non-owning BGR view as a non-owning @c cv::Mat header.
 *
 * @param bgr Source BGR view. Must remain valid while the returned matrix is used.
 * @return OpenCV matrix header, or an empty matrix for an invalid view.
 */
[[nodiscard]] inline cv::Mat wrap_bgr_view(const BgrImageView& bgr) noexcept {
    if (!bgr.is_valid()) return {};
    return cv::Mat(bgr.height, bgr.width, CV_8UC3, const_cast<std::uint8_t*>(bgr.data),
                   static_cast<std::size_t>(bgr.stride_bytes));
}

/**
 * @brief Convert an IDet point to an OpenCV point.
 */
[[nodiscard]] inline cv::Point2f to_cv(idet::Point2f p) noexcept {
    return {p.x, p.y};
}

/**
 * @brief Convert an OpenCV point to an IDet point.
 */
[[nodiscard]] inline idet::Point2f from_cv(const cv::Point2f& p) noexcept {
    return {p.x, p.y};
}

/**
 * @brief Convert an IDet quadrilateral to OpenCV point storage.
 */
[[nodiscard]] inline std::array<cv::Point2f, 4> to_cv_quad(const std::array<idet::Point2f, 4>& q) noexcept {
    return {to_cv(q[0]), to_cv(q[1]), to_cv(q[2]), to_cv(q[3])};
}

/**
 * @brief Convert an OpenCV quadrilateral to IDet point storage.
 */
[[nodiscard]] inline std::array<idet::Point2f, 4> from_cv_quad(const std::array<cv::Point2f, 4>& q) noexcept {
    return {from_cv(q[0]), from_cv(q[1]), from_cv(q[2]), from_cv(q[3])};
}

} // namespace idet::internal::opencv_adapter
