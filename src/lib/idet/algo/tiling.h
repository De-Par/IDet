/**
 * @file tiling.h
 * @ingroup idet_algo
 * @brief Image tiling utilities and generic tiled inference wrapper for @ref idet::engine::IEngine.
 *
 * @details
 * This module provides:
 *  - A helper to split an image into a regular grid of (optionally overlapping) tiles.
 *  - A generic tiled inference wrapper that runs engine inference per-tile and merges detections
 *    back into the full-image coordinate space.
 *
 * Coordinate conventions:
 *  - Tiles are expressed as @c cv::Rect in full-image pixel coordinates.
 *  - Detections returned by the engine for a tile are assumed to be in tile-local coordinates
 *    and are shifted by (tile.x, tile.y) when merging.
 *
 * Threading:
 *  - The tiling loop may use OpenMP if available (implementation-defined).
 *  - If bound inference is used in parallel, each concurrently processed tile must use a distinct
 *    bound context (see @ref idet::engine::IEngine::setup_binding).
 *
 * @note This header declares utilities only. The implementation is expected to be best-effort
 *       w.r.t. threading knobs (OpenMP thread count) and must not throw across API boundaries.
 */

#pragma once

#include "algo/geometry.h"
#include "engine/engine.h"
#include "idet.h"
#include "internal/opencv_headers.h" // IWYU pragma: keep
#include "status.h"

#include <vector>

namespace idet::algo {

/**
 * @brief Build overlapping tiles for an image.
 *
 * @details
 * Builds a regular grid of @p grid.rows x @p grid.cols tiles that cover the full image.
 * If @p overlap_rel is greater than 0, adjacent tiles overlap by a fraction of the nominal
 * tile size (best-effort; exact rounding is implementation-defined).
 *
 * Typical behavior (implementation-defined but recommended):
 *  - Clamp overlap to a safe range, e.g. [0 .. 0.9].
 *  - Ensure tiles are clipped to the image bounds.
 *  - Ensure every pixel is covered by at least one tile.
 *
 * @param img_w Image width in pixels (must be > 0).
 * @param img_h Image height in pixels (must be > 0).
 * @param grid Grid specification (rows x cols), must be valid (@c rows>0 && cols>0).
 * @param overlap_rel Relative overlap in [0..0.9] applied to the nominal tile size.
 * @return Vector of tile rectangles in full-image coordinates (size is typically rows*cols).
 */
std::vector<cv::Rect> make_tiles(int img_w, int img_h, const GridSpec& grid, float overlap_rel);

/**
 * @brief Run inference per-tile and merge detections into full-image coordinates.
 *
 * @details
 * High-level algorithm:
 *  1) Build tiles via @ref make_tiles.
 *  2) For each tile, run inference on ROI view (no deep copy) and get detections in tile-local space.
 *  3) Shift detections by tile origin (x += rc.x, y += rc.y) and append to the merged list.
 *
 * Merging:
 *  - This function only concatenates detections from tiles.
 *  - It does NOT perform cross-tile suppression; call @ref nms_poly on the merged output if needed.
 *
 * Unbound vs bound inference:
 *  - If @p bound is false, calls @ref idet::engine::IEngine::infer_unbound for every tile.
 *    This path is typically safe for parallel execution since tensors are per-call.
 *  - If @p bound is true, calls @ref idet::engine::IEngine::infer_bound and requires successful
 *    @ref idet::engine::IEngine::setup_binding beforehand.
 *
 * Bound contexts:
 *  - If @p parallel_bound is false, all tiles use @p ctx_idx.
 *  - If @p parallel_bound is true, tiles are distributed across contexts using a stable rule
 *    (documented contract: @c (tile_index % eng.bound_contexts())).
 *    Correctness/performance requires that the number of concurrently processed tiles does not
 *    exceed the number of prepared contexts; otherwise contexts may be reused concurrently
 *    (which is unsafe for bound inference).
 *
 * OpenMP:
 *  - @p tile_omp_threads is a best-effort request for the tiling loop parallelism.
 *    If OpenMP is unavailable or disabled, this parameter may be ignored.
 *
 * @param eng Engine instance (DBNet/SCRFD/...).
 * @param img_bgr Input image (must be CV_8UC3 BGR).
 * @param bound If true, uses bound inference; otherwise uses unbound inference.
 * @param ctx_idx Context index used when @p bound && !@p parallel_bound (must be valid).
 * @param parallel_bound If true, distribute tiles across bound contexts.
 * @param grid Grid spec (rows x cols).
 * @param overlap_rel Relative overlap between tiles in [0..0.9] (best-effort).
 * @param tile_omp_threads Desired OpenMP threads for the tiling loop (best-effort).
 *
 * @return Result with concatenated detections (full-image coordinates) or an error status.
 *
 * @warning Bound-mode parallelism requires @ref idet::engine::IEngine::binding_ready to be true
 *          and enough bound contexts for the intended concurrency.
 */
Result<std::vector<algo::Detection>> infer_tiled(engine::IEngine& eng, const cv::Mat& img_bgr, bool bound, int ctx_idx,
                                                 bool parallel_bound, const GridSpec& grid, float overlap_rel,
                                                 int tile_omp_threads) noexcept;

} // namespace idet::algo
