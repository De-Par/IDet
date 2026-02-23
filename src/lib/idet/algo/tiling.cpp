/**
 * @file tiling.cpp
 * @ingroup idet_algo
 * @brief Implementation of image tiling and tiled inference wrapper for @ref idet::engine::IEngine.
 *
 * @details
 * This translation unit implements:
 *  - @ref idet::algo::make_tiles : regular grid tiling with optional overlap (clipped to image bounds)
 *  - @ref idet::algo::infer_tiled : per-tile inference (optionally OpenMP-parallel) with safe bound-mode policy
 *
 * Key design points:
 *  - Tiles are represented as @c cv::Rect in full-image coordinates.
 *  - Tile extraction uses ROI views (@c cv::Mat tile = img_bgr(rc)), i.e. no deep copy.
 *  - Detections produced by engines are assumed to be tile-local and are translated back by (rc.x, rc.y).
 *  - In bound mode, parallel execution is allowed only when there are enough independent contexts
 *    (see @ref idet::engine::IEngine::setup_binding). Contexts are selected as (tid % contexts).
 *  - Errors are captured best-effort: the first failing status is propagated.
 *
 * @note This module does not apply cross-tile NMS. The caller should run @ref idet::algo::nms_poly
 *       on the merged detections if needed.
 */

#include "algo/tiling.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

#if defined(_OPENMP)
    #include <omp.h>
#endif

namespace idet::algo {

namespace {

/**
 * @brief Clamp integer value to an inclusive range.
 *
 * @param v Input value.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @return Clamped value.
 */
static inline int clampi(int v, int lo, int hi) noexcept {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

/**
 * @brief Clamp float value to an inclusive range.
 *
 * @param v Input value.
 * @param lo Lower bound (inclusive).
 * @param hi Upper bound (inclusive).
 * @return Clamped value.
 */
static inline float clampf(float v, float lo, float hi) noexcept {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

/**
 * @brief Split a 1D length @p L into @p K contiguous segments.
 *
 * @details
 * Produces two arrays:
 *  - @p starts[i] = starting offset of segment i
 *  - @p lens[i]   = length of segment i
 *
 * Segments cover [0, L) without gaps and without overlap.
 * Remainder is distributed to the first segments (classic "balanced split"):
 *  - base = L / K, rem = L % K
 *  - len[i] = base + (i < rem ? 1 : 0)
 *
 * @param L Total length (>= 0).
 * @param K Number of segments (> 0 expected by caller).
 * @param starts Output vector of starts (resized to K).
 * @param lens Output vector of lengths (resized to K).
 */
static inline void split_1d(int L, int K, std::vector<int>& starts, std::vector<int>& lens) {
    starts.resize((size_t)K);
    lens.resize((size_t)K);

    const int base = (K > 0) ? (L / K) : 0;
    const int rem = (K > 0) ? (L % K) : 0;

    int s = 0;
    for (int i = 0; i < K; ++i) {
        const int len = base + ((i < rem) ? 1 : 0);
        starts[(size_t)i] = s;
        lens[(size_t)i] = len;
        s += len;
    }
}

/**
 * @brief Offset a detection quad by a constant integer translation.
 *
 * @details
 * Tiled inference produces detections in tile-local coordinates. This helper converts
 * them back to the global image coordinate space by adding the tile origin (dx, dy)
 * to each of the 4 quad vertices.
 *
 * @param d Detection to be modified in-place.
 * @param dx Translation along X axis (pixels).
 * @param dy Translation along Y axis (pixels).
 */
static inline void offset_detection(algo::Detection& d, int dx, int dy) noexcept {
    for (int k = 0; k < 4; ++k) {
        d.pts[k].x += float(dx);
        d.pts[k].y += float(dy);
    }
}

} // namespace

std::vector<cv::Rect> make_tiles(int img_w, int img_h, const GridSpec& grid, float overlap) {
    std::vector<cv::Rect> out;

    if (img_h <= 0 || img_w <= 0) return out;
    if (grid.cols <= 0 || grid.rows <= 0) return out;

    // Safety clamp: too high overlap can explode tile sizes and reduce efficiency.
    overlap = clampf(overlap, 0.0f, 0.95f);

    std::vector<int> xs, ws, ys, hs;
    split_1d(img_w, grid.cols, xs, ws); // cols -> X
    split_1d(img_h, grid.rows, ys, hs); // rows -> Y

    out.reserve((size_t)grid.cols * (size_t)grid.rows);

    for (int ry = 0; ry < grid.rows; ++ry) {
        for (int cx = 0; cx < grid.cols; ++cx) {
            const int x0 = xs[(size_t)cx];
            const int w0 = ws[(size_t)cx];
            const int y0 = ys[(size_t)ry];
            const int h0 = hs[(size_t)ry];

            // Expand each tile by overlap fraction on each side (best-effort rounding).
            const int ex = (int)std::lround((double)w0 * (double)overlap);
            const int ey = (int)std::lround((double)h0 * (double)overlap);

            // Clip to image bounds.
            const int x1 = clampi(x0 - ex, 0, img_w);
            const int y1 = clampi(y0 - ey, 0, img_h);
            const int x2 = clampi(x0 + w0 + ex, 0, img_w);
            const int y2 = clampi(y0 + h0 + ey, 0, img_h);

            const int ww = std::max(0, x2 - x1);
            const int hh = std::max(0, y2 - y1);

            if (ww > 0 && hh > 0) out.emplace_back(x1, y1, ww, hh);
        }
    }

    return out;
}

Result<std::vector<algo::Detection>> infer_tiled(engine::IEngine& eng, const cv::Mat& img_bgr, bool bound, int ctx_idx,
                                                 bool parallel_bound, const GridSpec& grid, float overlap_rel,
                                                 int tile_omp_threads) noexcept {
    if (img_bgr.empty() || img_bgr.type() != CV_8UC3) {
        return Result<std::vector<algo::Detection>>::Err(Status::Invalid("infer_tiled: expected CV_8UC3 BGR"));
    }

    const int img_h = img_bgr.rows;
    const int img_w = img_bgr.cols;

    const auto rects = make_tiles(img_w, img_h, grid, overlap_rel);

    const int num_tiles = (int)rects.size();
    if (num_tiles == 0) return Result<std::vector<algo::Detection>>::Ok({});

    /**
     * @details
     * Determine tiling loop parallelism (best-effort):
     * - If OpenMP is available, prefer user-provided tile_omp_threads (if > 0),
     *   otherwise use omp_get_max_threads().
     * - If OpenMP is not available, fall back to 1.
     */
    int n_threads = 1;
#if defined(_OPENMP)
    n_threads = (tile_omp_threads > 0) ? tile_omp_threads : omp_get_max_threads();
    n_threads = std::max(1, n_threads);
#endif

    /**
     * @details
     * Bound inference safety rules:
     * - Bound mode requires eng.binding_ready().
     * - If parallel_bound is false => force single-thread execution and validate ctx_idx.
     * - If parallel_bound is true  => use at most 'contexts' threads and map tid -> ctx via tid%contexts.
     */
    const int contexts = eng.bound_contexts();
    if (bound) {
        if (!eng.binding_ready()) {
            return Result<std::vector<algo::Detection>>::Err(Status::Invalid("infer_tiled(bound): binding not ready"));
        }
        if (!parallel_bound) {
            // Safe mode: single thread + single explicitly requested context.
            n_threads = 1;
            if (ctx_idx < 0 || ctx_idx >= contexts) {
                return Result<std::vector<algo::Detection>>::Err(
                    Status::Invalid("infer_tiled(bound): ctx out of range"));
            }
        } else {
            // Parallel bound tiling: distribute tiles across independent contexts.
            if (contexts <= 0) {
                return Result<std::vector<algo::Detection>>::Err(Status::Invalid("infer_tiled(bound): contexts <= 0"));
            }
            n_threads = std::min(n_threads, contexts);
        }
    }

    /**
     * @details
     * Per-thread output buffers (TLS).
     *
     * We accumulate detections per-thread to avoid contention in the hot loop.
     * After the parallel region finishes, we merge all TLS vectors into one output.
     *
     * Reserve heuristic: ~4 detections per tile on average (rough guess).
     */
    std::vector<std::vector<algo::Detection>> tls((std::size_t)n_threads);
    for (auto& v : tls)
        v.reserve((std::size_t)(num_tiles * 4 / std::max(1, n_threads) + 8));

    /**
     * @details
     * Error propagation in parallel region:
     * - failed: atomic flag indicating that some iteration has failed.
     * - fail_status: first captured Status (best-effort).
     *
     * Note: we use an OpenMP critical section to reduce races on fail_status assignment.
     */
    std::atomic<bool> failed{false};
    Status fail_status = Status::Ok();

#if defined(_OPENMP)
    #pragma omp parallel num_threads(n_threads)
#endif
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        auto& local = tls[(std::size_t)tid];

#if defined(_OPENMP)
    #pragma omp for schedule(static)
#endif
        for (int i = 0; i < num_tiles; ++i) {
            if (failed.load(std::memory_order_relaxed)) continue;

            const cv::Rect& rc = rects[(std::size_t)i];

            // Create a view into the source image (no copy): tile shares data with img_bgr.
            cv::Mat tile = img_bgr(rc);

            // Context selection (bound-only):
            // - safe mode: ctx_idx
            // - parallel mode: tid % contexts
            const int use_ctx = bound ? (parallel_bound ? (tid % contexts) : ctx_idx) : 0;

            auto r = bound ? eng.infer_bound(tile, use_ctx) : eng.infer_unbound(tile);

            if (!r.ok()) {
                failed.store(true, std::memory_order_relaxed);
#if defined(_OPENMP)
    #pragma omp critical
#endif
                {
                    // Capture the first error status (best-effort).
                    if (fail_status.ok()) fail_status = r.status();
                }
                continue;
            }

            /**
             * @details
             * Move out the returned detections with O(1) swap to avoid extra allocations.
             * The result vector holds tile-local coordinates; convert to global coords.
             */
            std::vector<algo::Detection> dets;
            dets.swap(r.value()); // O(1)

            for (auto& d : dets)
                offset_detection(d, rc.x, rc.y);

            local.insert(local.end(), std::make_move_iterator(dets.begin()), std::make_move_iterator(dets.end()));
        }
    }

    if (failed.load(std::memory_order_relaxed)) {
        return Result<std::vector<algo::Detection>>::Err(fail_status.ok() ? Status::Internal("infer_tiled: failed")
                                                                          : fail_status);
    }

    /**
     * @details
     * Merge TLS vectors into a single output vector (move elements).
     */
    std::vector<algo::Detection> all;
    std::size_t total = 0;
    for (const auto& v : tls)
        total += v.size();
    all.reserve(total);
    for (auto& v : tls)
        all.insert(all.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));

    return Result<std::vector<algo::Detection>>::Ok(std::move(all));
}

} // namespace idet::algo
