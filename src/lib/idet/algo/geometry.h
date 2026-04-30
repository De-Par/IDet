/**
 * @file geometry.h
 * @ingroup idet_algo
 * @brief Geometry helpers for quadrilateral detections: ordering, scoring, IoU and aspect-fit.
 *
 * @details
 * This header defines the common geometric primitives used across detectors and post-processing:
 * - canonical quadrilateral ordering (TL,TR,BR,BL),
 * - quad IoU (exact convex polygon IoU or a fast AABB approximation),
 * - aspect-ratio preserving fit-to-square with stride alignment (e.g. 32).
 */

#pragma once

#include "idet.h"

#include <array>
#include <memory>
#include <utility>

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
     * Points are in float pixel coordinates. The semantic ordering is not guaranteed unless
     * explicitly normalized by @ref order_quad.
     */
    idet::Quad pts;

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
void order_quad(idet::Point2f quad[4]) noexcept;

/**
 * @brief Reusable scratch buffers for @ref quad_iou (exact polygon path).
 *
 * @details
 * Holds the per-call vectors used by the exact convex-IoU path so that NMS can call
 * @ref quad_iou thousands of times without paying for fresh heap allocations each time.
 *
 * Thread-safety:
 * - Not thread-safe. Use one instance per thread, or guard externally.
 */
struct QuadIouScratch {
    struct Impl;

    QuadIouScratch();
    ~QuadIouScratch();

    QuadIouScratch(QuadIouScratch&&) noexcept;
    QuadIouScratch& operator=(QuadIouScratch&&) noexcept;

    QuadIouScratch(const QuadIouScratch&) = delete;
    QuadIouScratch& operator=(const QuadIouScratch&) = delete;

    std::unique_ptr<Impl> impl;
};

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
 *
 * @note This overload allocates fresh buffers on each call. Prefer the
 * @ref quad_iou(const idet::Quad&, const idet::Quad&, bool, QuadIouScratch&)
 * overload from inner loops (e.g. NMS).
 */
float quad_iou(const idet::Quad& A, const idet::Quad& B, bool use_fast_iou = false);

/**
 * @brief IoU of two quadrilaterals using caller-provided scratch buffers.
 *
 * @details
 * Equivalent in behavior to the buffer-less overload, but reuses heap allocations from
 * @p scratch across calls. Result is bit-identical.
 *
 * @param A First quad (ideally ordered and convex).
 * @param B Second quad (ideally ordered and convex).
 * @param use_fast_iou If true, compute AABB IoU (scratch buffers are unused in this mode).
 * @param scratch Reusable scratch buffers.
 * @return IoU value in range [0, 1] (returns 0 if union is 0).
 */
float quad_iou(const idet::Quad& A, const idet::Quad& B, bool use_fast_iou, QuadIouScratch& scratch);

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
float aabb_iou(const idet::Quad& A, const idet::Quad& B);

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
