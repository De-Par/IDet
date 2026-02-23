/**
 * @file geometry.h
 * @ingroup idet_algo
 * @brief Geometry helpers for quadrilateral detections: ordering, scoring, IoU and aspect-fit.
 *
 * @details
 * This header defines the common geometric primitives used across detectors and post-processing:
 * - canonical quadrilateral ordering (TL,TR,BR,BL),
 * - contour scoring over a probability map (DBNet-style),
 * - quad IoU (exact convex polygon IoU or a fast AABB approximation),
 * - aspect-ratio preserving fit-to-square with stride alignment (e.g. 32).
 */

#pragma once

#include "internal/opencv_headers.h" // IWYU pragma: keep

#include <array>
#include <utility>
#include <vector>

namespace idet::algo {

/**
 * @brief Generic detection primitive used across engines/algorithms.
 *
 * @details
 * `pts` represent a quadrilateral in image coordinates (float pixels).
 * Convention expected by several algorithms:
 *  - points are ordered (top-left, top-right, bottom-right, bottom-left)
 *  - polygon is convex (required by quad_iou() exact mode)
 *
 * Engines are responsible for producing consistently ordered quads (or calling @ref order_quad).
 */
struct Detection {
    /**
     * @brief Quadrilateral corner points in image coordinates.
     *
     * @details
     * Points are in `cv::Point2f` (float pixel coordinates). The semantic ordering is
     * not guaranteed unless explicitly normalized by @ref order_quad.
     */
    std::array<cv::Point2f, 4> pts;

    /**
     * @brief Detection confidence score.
     *
     * @details
     * The interpretation is model-specific:
     * - DBNet: usually a textness/box confidence.
     * - SCRFD: usually a face classification score.
     */
    float score = 0.0f;
};

/**
 * @brief Canonicalize quadrilateral point order.
 *
 * @details
 * Reorders points to a stable TL,TR,BR,BL layout.
 * Contains fallback logic for degenerate cases when sum/diff heuristics collide.
 *
 * @param quad Array of 4 points in arbitrary order (modified in-place).
 */
void order_quad(cv::Point2f quad[4]) noexcept;

/**
 * @brief Compute mean probability inside a contour.
 *
 * @details
 * Computes an average value of @p prob inside the polygon represented by @p contour.
 * This is typically used for DBNet-style scoring of connected components on a probmap.
 *
 * @param prob Single-channel probability map (CV_32F).
 * @param contour Contour points in prob coordinates.
 * @return Mean value of prob inside contour; returns 0 if contour/bbox invalid.
 *
 * @note Uses thread_local buffers for mask/temporary contour storage for performance.
 */
float contour_score(const cv::Mat& prob, const std::vector<cv::Point>& contour);

/**
 * @brief IoU of two quadrilaterals.
 *
 * @details
 * Exact mode uses @c cv::intersectConvexConvex and requires:
 *  - both quads are convex,
 *  - point order describes the polygon boundary (CW/CCW).
 *
 * If @p use_fast_iou is true, falls back to AABB IoU approximation via @ref aabb_iou.
 *
 * @param A First quad (ideally ordered and convex).
 * @param B Second quad (ideally ordered and convex).
 * @param use_fast_iou If true, compute AABB IoU instead of polygon IoU.
 * @return IoU value in range [0, 1] (returns 0 if union is 0).
 */
float quad_iou(const std::array<cv::Point2f, 4>& A, const std::array<cv::Point2f, 4>& B, bool use_fast_iou = false);

/**
 * @brief Computes IoU using axis-aligned bounding boxes (AABB) derived from quads.
 *
 * @details
 * This is a cheaper approximation of @ref quad_iou:
 * - each quad is reduced to its min/max X/Y extents (AABB),
 * - IoU is computed for those AABBs.
 *
 * Useful as:
 * - a fast reject test before expensive polygon IoU,
 * - a simplified NMS metric for near-axis-aligned boxes.
 *
 * @param A First quad.
 * @param B Second quad.
 * @return AABB IoU value in range [0, 1] (implementation should return 0 if union is 0).
 */
float aabb_iou(const std::array<cv::Point2f, 4>& A, const std::array<cv::Point2f, 4>& B);

/**
 * @brief Computes a size that fits an image into a square side while preserving aspect ratio,
 *        and aligns dimensions to multiples of 32.
 *
 * @details
 * Many CNN backbones require spatial dimensions divisible by a given stride (commonly 32).
 * This helper computes a target (width, height) such that:
 * - the longer side is clamped/fit to @p side,
 * - aspect ratio is preserved,
 * - both output dimensions are aligned to 32 (implementation-dependent rounding policy).
 *
 * @param iw Input width in pixels.
 * @param ih Input height in pixels.
 * @param side Target side length for the longer edge (max dimension).
 * @return A pair `{out_w, out_h}` aligned to 32.
 *
 * @pre @p iw > 0 and @p ih > 0.
 * @pre @p side > 0.
 */
std::pair<int, int> aspect_fit32(const int iw, const int ih, const int side);

} // namespace idet::algo
