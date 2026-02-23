/**
 * @file nms.h
 * @ingroup idet_algo
 * @brief Non-maximum suppression (NMS) for quadrilateral detections.
 *
 * @details
 * Implements score-sorted greedy NMS on detections using @ref quad_iou().
 * Uses an optional uniform grid acceleration (AABB-based) to reduce IoU checks.
 *
 * The IoU backend can be chosen at runtime:
 * - exact polygon IoU (default),
 * - fast AABB IoU approximation (set @p use_fast_iou = true).
 */

#pragma once

#include "algo/geometry.h"

#include <vector>

namespace idet::algo {

/**
 * @brief Axis-aligned bounding box (AABB) in float image coordinates.
 *
 * @details
 * Represents a rectangle aligned to the image axes:
 * - minx/miny: lower bounds
 * - maxx/maxy: upper bounds
 *
 * Used for fast overlap checks and as an approximation to polygon IoU.
 */
struct AABB {
    float minx, miny, maxx, maxy;
};

/**
 * @brief Greedy NMS for quad detections.
 *
 * @param dets Input detections (quad + score).
 * @param iou_thr IoU threshold.
 * @param use_fast_iou If true, uses AABB IoU approximation inside @ref quad_iou.
 *
 * @return Filtered detections in descending score order.
 *
 * @par Special cases
 *  - iou_thr <= 0 : returns all detections sorted by score (no suppression)
 *  - iou_thr >= 1 : returns the single best detection (max score)
 *
 * @note Uses AABB overlap as a cheap rejection before computing quad IoU.
 */
std::vector<algo::Detection> nms_poly(const std::vector<algo::Detection>& dets, float iou_thr,
                                      bool use_fast_iou = false);

} // namespace idet::algo
