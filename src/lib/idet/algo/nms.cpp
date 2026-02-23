/**
 * @file nms.cpp
 * @ingroup idet_algo
 * @brief Implementation of polygon NMS for quadrilateral detections.
 *
 * @details
 * Implements score-sorted greedy NMS for @ref idet::algo::Detection using @ref idet::algo::quad_iou.
 *
 * Performance notes:
 *  - Uses AABB overlap as a cheap reject test before computing polygon IoU.
 *  - Optionally enables a uniform grid acceleration structure to reduce candidate comparisons.
 *    The grid is disabled automatically if the number of grid cells exceeds a safety limit.
 *
 * Output:
 *  - Returns detections in descending score order (processing order after sorting).
 */

#include "algo/nms.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace idet::algo {

/**
 * @brief Computes an axis-aligned bounding box (AABB) of a quadrilateral detection.
 *
 * @details
 * The AABB is derived by taking min/max of X and Y over the 4 quad vertices.
 * Used as a fast overlap pre-check in NMS to avoid expensive polygon IoU calls.
 *
 * @param d Detection with 4 quad points.
 * @return AABB covering the quad.
 */
static inline algo::AABB aabb_of(const algo::Detection& d) noexcept {
    float minx = d.pts[0].x, miny = d.pts[0].y, maxx = d.pts[0].x, maxy = d.pts[0].y;
    for (int k = 1; k < 4; ++k) {
        minx = std::min(minx, d.pts[k].x);
        miny = std::min(miny, d.pts[k].y);
        maxx = std::max(maxx, d.pts[k].x);
        maxy = std::max(maxy, d.pts[k].y);
    }
    return {minx, miny, maxx, maxy};
}

/**
 * @brief Checks whether two AABBs overlap (non-empty intersection test).
 *
 * @details
 * Returns false if the boxes are separated along X or Y axes.
 *
 * @param a First AABB.
 * @param b Second AABB.
 * @return True if AABBs overlap, false otherwise.
 */
static inline bool aabb_overlap(const algo::AABB& a, const algo::AABB& b) noexcept {
    return !(a.maxx < b.minx || b.maxx < a.minx || a.maxy < b.miny || b.maxy < a.miny);
}

/**
 * @brief Performs Non-Maximum Suppression (NMS) on quadrilateral detections using polygon IoU or AABB IoU.
 *
 * @details
 * High-level behavior:
 * - Input detections are sorted by decreasing @ref Detection::score.
 * - Iterate in that order, keep the current detection if not suppressed.
 * - Suppress any lower-ranked detections whose IoU with the kept detection is >= @p iou_thr_in.
 *
 * Performance optimizations:
 * - Uses AABB overlap as a cheap pre-filter before calling @ref quad_iou.
 * - Uses a uniform grid acceleration for candidate enumeration (CSR layout).
 *   This avoids allocating millions of small vectors.
 * - Uses a stamp + @c seen array to avoid processing the same candidate multiple times
 *   when a box spans multiple grid cells.
 *
 * Special cases:
 * - If @p iou_thr_in <= 0: no suppression is performed; output is all detections sorted by score.
 * - If @p iou_thr_in >= 1: only the single best-scoring detection is returned.
 *
 * @param dets Input detections.
 * @param iou_thr_in IoU threshold.
 * @param use_fast_iou If true, internally uses AABB IoU approximation via @ref quad_iou(..., true).
 * @return A filtered subset of @p dets after NMS, in descending score order.
 */
std::vector<algo::Detection> nms_poly(const std::vector<algo::Detection>& dets, float iou_thr_in, bool use_fast_iou) {
    const int N = (int)dets.size();
    if (N == 0) return {};

    float iou_thr = iou_thr_in;

    // Threshold <= 0: disable suppression, just return detections sorted by score.
    if (iou_thr <= 0.0f) {
        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return dets[a].score > dets[b].score; });

        std::vector<Detection> out;
        out.reserve((std::size_t)N);
        for (int i : order)
            out.push_back(dets[i]);
        return out;
    }

    // Threshold >= 1: only keep the best element (since IoU is in [0,1]).
    if (iou_thr >= 1.0f) {
        int best = 0;
        for (int i = 1; i < N; ++i)
            if (dets[i].score > dets[best].score) best = i;
        return {dets[best]};
    }

    // Sort detections by descending score; order[] is the processing permutation.
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return dets[a].score > dets[b].score; });

    // rank[idx] gives the position in the sorted order, used to enforce "only suppress lower-ranked".
    std::vector<int> rank(N, 0);
    for (int p = 0; p < N; ++p)
        rank[order[p]] = p;

    // Precompute AABBs and stats for grid sizing.
    std::vector<algo::AABB> boxes((std::size_t)N);

    float minx = std::numeric_limits<float>::infinity();
    float miny = std::numeric_limits<float>::infinity();
    float maxx = -std::numeric_limits<float>::infinity();
    float maxy = -std::numeric_limits<float>::infinity();

    float mean_w = 0.f;
    float mean_h = 0.f;

    for (int i = 0; i < N; ++i) {
        boxes[(std::size_t)i] = aabb_of(dets[(std::size_t)i]);

        minx = std::min(minx, boxes[(std::size_t)i].minx);
        miny = std::min(miny, boxes[(std::size_t)i].miny);
        maxx = std::max(maxx, boxes[(std::size_t)i].maxx);
        maxy = std::max(maxy, boxes[(std::size_t)i].maxy);

        mean_w += std::max(1.f, boxes[(std::size_t)i].maxx - boxes[(std::size_t)i].minx);
        mean_h += std::max(1.f, boxes[(std::size_t)i].maxy - boxes[(std::size_t)i].miny);
    }
    mean_w /= (float)N;
    mean_h /= (float)N;

    // Shift grid origin to (minx, miny) to handle potential negative coords safely.
    const float ox = std::isfinite(minx) ? minx : 0.f;
    const float oy = std::isfinite(miny) ? miny : 0.f;

    const float span_x = std::max(1.f, maxx - ox);
    const float span_y = std::max(1.f, maxy - oy);

    /**
     * Choose a uniform grid cell size based on average box size.
     * - Start from 0.5*(mean_w + mean_h)
     * - Clamp to [48, 256]
     * - Snap to {64, 128, 256}
     */
    float cell_f = 0.5f * (mean_w + mean_h);
    cell_f = std::clamp(cell_f, 48.0f, 256.0f);
    int cell = (int)std::lround(cell_f);

    if (cell < 64)
        cell = 64;
    else if (cell < 128)
        cell = 128;
    else
        cell = 256;

    const int nx = std::max(1, (int)std::floor(span_x / (float)cell) + 1);
    const int ny = std::max(1, (int)std::floor(span_y / (float)cell) + 1);

    const std::size_t grid_cells = (std::size_t)nx * (std::size_t)ny;

    // Safety: CSR arrays are cheap-ish, but still cap in extreme cases.
    // (You can tune this threshold; CSR is far lighter than vector-of-vectors.)
    const bool use_grid = (grid_cells <= 2'000'000ULL);

    // CSR grid storage:
    // offsets[c]..offsets[c+1] is a list of detection indices whose AABB overlaps cell c.
    std::vector<std::uint32_t> offsets;
    std::vector<std::uint32_t> cursor;
    std::vector<int> items;

    auto cell_id = [&](int x, int y) noexcept -> std::size_t {
        return (std::size_t)y * (std::size_t)nx + (std::size_t)x;
    };

    if (use_grid) {
        std::vector<std::uint32_t> counts(grid_cells, 0);

        // Pass 1: count insertions per cell.
        for (int i = 0; i < N; ++i) {
            const algo::AABB& a = boxes[(std::size_t)i];

            const int x0 = std::clamp((int)std::floor((a.minx - ox) / (float)cell), 0, nx - 1);
            const int x1 = std::clamp((int)std::floor((a.maxx - ox) / (float)cell), 0, nx - 1);
            const int y0 = std::clamp((int)std::floor((a.miny - oy) / (float)cell), 0, ny - 1);
            const int y1 = std::clamp((int)std::floor((a.maxy - oy) / (float)cell), 0, ny - 1);

            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    const std::size_t id = cell_id(x, y);
                    counts[id] += 1;
                }
            }
        }

        // Prefix sum -> offsets
        offsets.resize(grid_cells + 1);
        offsets[0] = 0;
        for (std::size_t c = 0; c < grid_cells; ++c) {
            offsets[c + 1] = offsets[c] + counts[c];
        }

        // Allocate flat items and make a cursor copy.
        items.resize((std::size_t)offsets.back());
        cursor = offsets; // cursor will be incremented while filling

        // Pass 2: fill items.
        for (int i = 0; i < N; ++i) {
            const algo::AABB& a = boxes[(std::size_t)i];

            const int x0 = std::clamp((int)std::floor((a.minx - ox) / (float)cell), 0, nx - 1);
            const int x1 = std::clamp((int)std::floor((a.maxx - ox) / (float)cell), 0, nx - 1);
            const int y0 = std::clamp((int)std::floor((a.miny - oy) / (float)cell), 0, ny - 1);
            const int y1 = std::clamp((int)std::floor((a.maxy - oy) / (float)cell), 0, ny - 1);

            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    const std::size_t id = cell_id(x, y);
                    const std::uint32_t pos = cursor[id]++;
                    items[(std::size_t)pos] = i;
                }
            }
        }
    }

    std::vector<std::uint8_t> suppressed((std::size_t)N, 0);
    std::vector<algo::Detection> keep;
    keep.reserve((std::size_t)N);

    // "Seen" marker array to avoid duplicates when scanning multiple cells.
    std::vector<int> seen(N, -1);
    int stamp = 0;

    for (int p = 0; p < N; ++p) {
        const int i = order[p];
        if (suppressed[(std::size_t)i]) continue;

        keep.push_back(dets[(std::size_t)i]);
        const algo::AABB& ai = boxes[(std::size_t)i];

        ++stamp;

        auto process_j = [&](int j) {
            if (j == i) return;
            if (suppressed[(std::size_t)j]) return;

            // Only suppress strictly lower-ranked detections.
            if (rank[j] <= rank[i]) return;

            // Cheap reject via AABB overlap.
            if (!aabb_overlap(ai, boxes[(std::size_t)j])) return;

            // Accurate overlap test via quad IoU (or fast AABB IoU).
            const float iou = quad_iou(dets[(std::size_t)i].pts, dets[(std::size_t)j].pts, use_fast_iou);
            if (iou >= iou_thr) suppressed[(std::size_t)j] = 1;
        };

        if (!use_grid) {
            // Fallback: scan remaining detections in score order.
            for (int q = p + 1; q < N; ++q) {
                const int j = order[q];
                if (suppressed[(std::size_t)j]) continue;
                if (!aabb_overlap(ai, boxes[(std::size_t)j])) continue;
                process_j(j);
            }
            continue;
        }

        // Grid-based candidate enumeration: only scan cells overlapped by current AABB.
        const int x0 = std::clamp((int)std::floor((ai.minx - ox) / (float)cell), 0, nx - 1);
        const int x1 = std::clamp((int)std::floor((ai.maxx - ox) / (float)cell), 0, nx - 1);
        const int y0 = std::clamp((int)std::floor((ai.miny - oy) / (float)cell), 0, ny - 1);
        const int y1 = std::clamp((int)std::floor((ai.maxy - oy) / (float)cell), 0, ny - 1);

        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const std::size_t id = cell_id(x, y);
                const std::uint32_t beg = offsets[id];
                const std::uint32_t end = offsets[id + 1];
                for (std::uint32_t k = beg; k < end; ++k) {
                    const int j = items[(std::size_t)k];
                    if (seen[j] == stamp) continue;
                    seen[j] = stamp;
                    process_j(j);
                }
            }
        }
    }

    return keep;
}

} // namespace idet::algo
