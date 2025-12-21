#include "nms.h"

#include "geometry.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

static inline AABB aabb_of(const Detection& d) noexcept {
    float minx = d.pts[0].x, miny = d.pts[0].y, maxx = d.pts[0].x, maxy = d.pts[0].y;
    for (int k = 1; k < 4; ++k) {
        minx = std::min(minx, d.pts[k].x);
        miny = std::min(miny, d.pts[k].y);
        maxx = std::max(maxx, d.pts[k].x);
        maxy = std::max(maxy, d.pts[k].y);
    }
    return {minx, miny, maxx, maxy};
}

static inline bool aabb_overlap(const AABB& a, const AABB& b) noexcept {
    return !(a.maxx < b.minx || b.maxx < a.minx || a.maxy < b.miny || b.maxy < a.miny);
}

// Greedy NMS with spatial grid (fast & correct).
std::vector<Detection> nms_poly(const std::vector<Detection>& dets, const float iou_thr_in) {
    const int N = (int)dets.size();
    if (N == 0) return {};

    float iou_thr = iou_thr_in;
    if (iou_thr <= 0.0f) {
        // keep all, but return sorted-by-score (common expectation)
        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return dets[a].score > dets[b].score; });
        std::vector<Detection> out;
        out.reserve(N);
        for (int i : order)
            out.push_back(dets[i]);
        return out;
    }
    if (iou_thr >= 1.0f) {
        // keep best only
        int best = 0;
        for (int i = 1; i < N; ++i)
            if (dets[i].score > dets[best].score) best = i;
        return {dets[best]};
    }

    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return dets[a].score > dets[b].score; });

    // rank in sorted order to enforce "suppress only lower-score"
    std::vector<int> rank(N, 0);
    for (int p = 0; p < N; ++p)
        rank[order[p]] = p;

    // AABBs + bounds for grid
    std::vector<AABB> boxes(N);
    float maxx = 0.f, maxy = 0.f;
    float mean_w = 0.f, mean_h = 0.f;
    for (int i = 0; i < N; ++i) {
        boxes[i] = aabb_of(dets[i]);
        maxx = std::max(maxx, boxes[i].maxx);
        maxy = std::max(maxy, boxes[i].maxy);
        mean_w += std::max(1.f, boxes[i].maxx - boxes[i].minx);
        mean_h += std::max(1.f, boxes[i].maxy - boxes[i].miny);
    }
    mean_w /= (float)N;
    mean_h /= (float)N;

    // cell size heuristic (adaptive, clamped)
    float cell_f = 0.5f * (mean_w + mean_h);
    cell_f = std::clamp(cell_f, 48.0f, 256.0f);
    int cell = (int)std::lround(cell_f);
    // snap to a power-of-two-ish bucket for faster divisions (optional)
    if (cell < 64)
        cell = 64;
    else if (cell < 128)
        cell = 128;
    else
        cell = 256;

    const int nx = std::max(1, (int)(maxx / cell) + 1);
    const int ny = std::max(1, (int)(maxy / cell) + 1);

    // If grid explodes (very large coords), fallback to brute force
    const long long grid_cells = 1LL * nx * ny;
    const bool use_grid = (grid_cells <= 2'000'000LL);

    std::vector<std::vector<int>> grid;
    if (use_grid) {
        grid.assign((size_t)grid_cells, {});
        grid.shrink_to_fit();
        grid.assign((size_t)grid_cells, {});
        // Fill grid
        for (int i = 0; i < N; ++i) {
            const AABB& a = boxes[i];
            const int x0 = std::clamp((int)(a.minx / cell), 0, nx - 1);
            const int x1 = std::clamp((int)(a.maxx / cell), 0, nx - 1);
            const int y0 = std::clamp((int)(a.miny / cell), 0, ny - 1);
            const int y1 = std::clamp((int)(a.maxy / cell), 0, ny - 1);
            for (int y = y0; y <= y1; ++y) {
                const int row = y * nx;
                for (int x = x0; x <= x1; ++x) {
                    grid[(size_t)(row + x)].push_back(i);
                }
            }
        }
    }

    std::vector<uint8_t> suppressed((size_t)N, 0);
    std::vector<Detection> keep;
    keep.reserve((size_t)N);

    // per-iteration duplicate filter for grid candidates
    std::vector<int> seen(N, -1);
    int stamp = 0;

    for (int p = 0; p < N; ++p) {
        const int i = order[p];
        if (suppressed[(size_t)i]) continue;

        keep.push_back(dets[i]);
        const AABB& ai = boxes[i];

        ++stamp;

        auto process_j = [&](int j) {
            if (j == i) return;
            if (suppressed[(size_t)j]) return;
            if (rank[j] <= rank[i]) return; // only lower score
            if (!aabb_overlap(ai, boxes[j])) return;

            const float iou = quad_iou(dets[i].pts, dets[j].pts);
            if (iou >= iou_thr) suppressed[(size_t)j] = 1;
        };

        if (!use_grid) {
            // brute force but with safe AABB overlap reject
            for (int q = p + 1; q < N; ++q) {
                const int j = order[q];
                if (suppressed[(size_t)j]) continue;
                if (!aabb_overlap(ai, boxes[j])) continue;
                process_j(j);
            }
            continue;
        }

        const int x0 = std::clamp((int)(ai.minx / cell), 0, nx - 1);
        const int x1 = std::clamp((int)(ai.maxx / cell), 0, nx - 1);
        const int y0 = std::clamp((int)(ai.miny / cell), 0, ny - 1);
        const int y1 = std::clamp((int)(ai.maxy / cell), 0, ny - 1);

        for (int y = y0; y <= y1; ++y) {
            const int row = y * nx;
            for (int x = x0; x <= x1; ++x) {
                const auto& cell_list = grid[(size_t)(row + x)];
                for (int j : cell_list) {
                    if (seen[j] == stamp) continue;
                    seen[j] = stamp;
                    process_j(j);
                }
            }
        }
    }

    return keep;
}