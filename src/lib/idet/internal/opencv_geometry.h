/**
 * @file opencv_geometry.h
 * @ingroup idet_internal
 * @brief OpenCV-backed geometry helpers isolated from core geometry declarations.
 */

#pragma once

#include "internal/opencv_headers.h" // IWYU pragma: keep

#include <vector>

namespace idet::internal::opencv_geometry {

/**
 * @brief Scratch buffers reused by contour scoring to avoid per-call allocations.
 */
struct ContourScoreScratch {
    /** @brief Reusable full-size mask used for mean-score extraction. */
    cv::Mat mask_full;
    /** @brief Reusable contour wrapper passed to OpenCV drawing routines. */
    std::vector<std::vector<cv::Point>> cnt{1};
};

/**
 * @brief Compute mean probability inside a contour.
 *
 * @param prob Single-channel probability map.
 * @param contour Contour in probability-map coordinates.
 * @return Mean score inside the contour, or 0 for invalid input.
 */
float contour_score(const cv::Mat& prob, const std::vector<cv::Point>& contour);

/**
 * @brief Compute mean probability inside a contour using caller-owned scratch buffers.
 *
 * @param prob Single-channel probability map.
 * @param contour Contour in probability-map coordinates.
 * @param scratch Reusable OpenCV buffers.
 * @return Mean score inside the contour, or 0 for invalid input.
 */
float contour_score(const cv::Mat& prob, const std::vector<cv::Point>& contour, ContourScoreScratch& scratch);

} // namespace idet::internal::opencv_geometry
