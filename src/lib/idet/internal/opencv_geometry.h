/**
 * @file opencv_geometry.h
 * @ingroup idet_internal
 * @brief OpenCV-backed geometry helpers isolated from core geometry declarations.
 */

#pragma once

#include "internal/opencv_headers.h" // IWYU pragma: keep

#include <vector>

namespace idet::internal::opencv_geometry {

struct ContourScoreScratch {
    cv::Mat mask_full;
    std::vector<std::vector<cv::Point>> cnt{1};
};

float contour_score(const cv::Mat& prob, const std::vector<cv::Point>& contour);
float contour_score(const cv::Mat& prob, const std::vector<cv::Point>& contour, ContourScoreScratch& scratch);

} // namespace idet::internal::opencv_geometry
