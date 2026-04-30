/**
 * @file geometry.cpp
 * @ingroup idet_algo
 * @brief Implementations for quad ordering, contour scoring, IoU, and aspect-fit helpers.
 *
 * @details
 * Implements:
 *  - order_quad(): robust canonical ordering TL,TR,BR,BL with fallbacks for degenerate input,
 *  - contour_score(): mean probability inside a contour using a masked ROI (optional caller-provided scratch buffers),
 *  - aabb_iou(): fast axis-aligned IoU approximation from quad extents,
 *  - quad_iou(): exact convex IoU via OpenCV (or AABB approximation when USE_FAST_IOU=1),
 *  - aspect_fit32(): aspect-ratio fit to a square side + 32-alignment.
 *
 * Notes:
 *  - Exact quad_iou() relies on convex hulls; for invalid/degenerate inputs returns 0.
 *  - Many routines include NaN/Inf guards to keep behavior deterministic in production.
 */

#include "algo/geometry.h"
#include "internal/opencv_adapter.h"
#include "internal/opencv_geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace idet::algo {

struct QuadIouScratch::Impl {
    std::vector<cv::Point2f> pts;
    std::vector<cv::Point2f> a;
    std::vector<cv::Point2f> b;
    std::vector<cv::Point2f> inter;
};

QuadIouScratch::QuadIouScratch() : impl(std::make_unique<Impl>()) {}
QuadIouScratch::~QuadIouScratch() = default;
QuadIouScratch::QuadIouScratch(QuadIouScratch&&) noexcept = default;
QuadIouScratch& QuadIouScratch::operator=(QuadIouScratch&&) noexcept = default;

void order_quad(idet::Point2f quad[4]) noexcept {
    // kEpsLex is an absolute pixel-space tolerance for image coordinates and is intentionally
    // not scaled: typical images keep coordinates in O(10^3) and lex tie-breaking only needs to
    // resolve sub-pixel ambiguities.
    constexpr float kEpsLex = 1e-4f;
    constexpr float kQuarter = 0.25f;

    // Relative tolerance used to derive scale-aware epsilons further down. The previous
    // implementation used a fixed kEpsAng = 1e-6f for both half-plane and cross-product
    // comparisons, which is meaningless for quadrilaterals with pixel-scale coordinates: for a
    // box at ~10^3 pixels the cross product magnitude is ~10^6, so 1e-6 was effectively zero,
    // and for tiny boxes the same threshold became enormous relative to the underlying scale.
    // We therefore derive eps_pos / eps_cross from the actual radius of the quad below.
    constexpr float kEpsRel = 1e-6f;

    auto absf = [](float x) noexcept { return std::fabs(x); };

    auto is_finite = [](const idet::Point2f& p) noexcept { return std::isfinite(p.x) && std::isfinite(p.y); };

    auto sub = [](const idet::Point2f& a, const idet::Point2f& b) noexcept -> idet::Point2f {
        return {a.x - b.x, a.y - b.y};
    };

    auto cross2 = [](const idet::Point2f& a, const idet::Point2f& b) noexcept -> float {
        return a.x * b.y - a.y * b.x;
    };

    auto sqr_len = [](const idet::Point2f& v) noexcept -> float { return v.x * v.x + v.y * v.y; };

    auto lex_yx_less = [&](const idet::Point2f& a, const idet::Point2f& b) noexcept {
        if (a.y < b.y - kEpsLex) return true;
        if (a.y > b.y + kEpsLex) return false;
        return a.x < b.x - kEpsLex;
    };

    // 1) NaN/Inf -> deterministic lex fallback
    for (int i = 0; i < 4; ++i) {
        if (!is_finite(quad[i])) {
            idet::Quad r = {quad[0], quad[1], quad[2], quad[3]};

            auto swap_lex = [&](int i0, int i1) noexcept {
                if (lex_yx_less(r[i1], r[i0])) std::swap(r[i0], r[i1]);
            };
            swap_lex(0, 1);
            swap_lex(2, 3);
            swap_lex(0, 2);
            swap_lex(1, 3);
            swap_lex(1, 2);

            const idet::Point2f tl = r[0];
            const idet::Point2f br = r[3];
            const idet::Point2f p1 = r[1];
            const idet::Point2f p2 = r[2];

            idet::Point2f tr = p1, bl = p2;
            // TR = more right; tie -> more top
            if (p2.x > p1.x + kEpsLex || (absf(p2.x - p1.x) <= kEpsLex && p2.y < p1.y - kEpsLex)) {
                tr = p2;
                bl = p1;
            }

            quad[0] = tl;
            quad[1] = tr;
            quad[2] = br;
            quad[3] = bl;
            return;
        }
    }

    // 2) centroid
    idet::Point2f c;
    c.x = (quad[0].x + quad[1].x + quad[2].x + quad[3].x) * kQuarter;
    c.y = (quad[0].y + quad[1].y + quad[2].y + quad[3].y) * kQuarter;

    // 3) compute the radius of the quad relative to its centroid so we can derive scale-aware
    //    epsilons. max_r2 is the largest squared distance from any vertex to the centroid.
    //    - eps_pos has the same dimension as a coordinate (units of pixels), so it scales with
    //      sqrt(max_r2); the additive 1.0 prevents underflow on degenerate inputs.
    //    - eps_cross has the dimension of an area (cross-product magnitude scales with r^2),
    //      so it scales with max_r2 directly. The additive 1.0 keeps a sensible floor for
    //      sub-pixel quads.
    float max_r2 = 0.f;
    for (int i = 0; i < 4; ++i) {
        const idet::Point2f v = sub(quad[i], c);
        max_r2 = std::max(max_r2, sqr_len(v));
    }
    const float eps_pos = kEpsRel * (std::sqrt(max_r2) + 1.f);
    const float eps_cross = kEpsRel * (max_r2 + 1.f);

    // 4) angle ordering without atan2: half-plane + cross. Uses scale-aware epsilons so the
    //    behavior is consistent for both small (~1 px) and large (~10^4 px) quads.
    auto angle_less = [&](const idet::Point2f& p, const idet::Point2f& q) noexcept {
        const idet::Point2f vp = sub(p, c);
        const idet::Point2f vq = sub(q, c);

        // upper half-plane first: (y < 0) or (y ~= 0 and x >= 0)
        const bool up_p = (vp.y < -eps_pos) || (absf(vp.y) <= eps_pos && vp.x >= 0.f);
        const bool up_q = (vq.y < -eps_pos) || (absf(vq.y) <= eps_pos && vq.x >= 0.f);
        if (up_p != up_q) return up_p > up_q;

        const float cr = cross2(vp, vq);
        if (absf(cr) > eps_cross) return cr > 0.f;

        // collinear: farther first (stable). |dp - dq| has the same dimension as cross product.
        const float dp = sqr_len(vp);
        const float dq = sqr_len(vq);
        if (absf(dp - dq) > eps_cross) return dp > dq;

        // full tie: deterministic (x then y)
        if (p.x < q.x - kEpsLex) return true;
        if (p.x > q.x + kEpsLex) return false;
        return p.y < q.y - kEpsLex;
    };

    idet::Quad r = {quad[0], quad[1], quad[2], quad[3]};

    // sorting network: 4 elems, 5 comps
    auto swap_if = [&](int i0, int i1) noexcept {
        if (angle_less(r[i1], r[i0])) std::swap(r[i0], r[i1]);
    };
    swap_if(0, 1);
    swap_if(2, 3);
    swap_if(0, 2);
    swap_if(1, 3);
    swap_if(1, 2);

    // 5) degeneracy check: area2 scaled by quad radius (same scale-aware epsilon as above).
    auto poly_area2 = [&](const idet::Quad& p) noexcept -> float {
        float a = 0.f;
        for (int i = 0; i < 4; ++i) {
            const int j = (i + 1) & 3;
            a += p[i].x * p[j].y - p[j].x * p[i].y;
        }
        return a;
    };

    const float a2 = poly_area2(r);
    const float deg_thr = eps_cross;

    if (absf(a2) <= deg_thr) {
        // fallback: lex sort + TL/BR + split remaining
        auto swap_lex = [&](int i0, int i1) noexcept {
            if (lex_yx_less(r[i1], r[i0])) std::swap(r[i0], r[i1]);
        };
        swap_lex(0, 1);
        swap_lex(2, 3);
        swap_lex(0, 2);
        swap_lex(1, 3);
        swap_lex(1, 2);

        const idet::Point2f tl = r[0];
        const idet::Point2f br = r[3];
        const idet::Point2f p1 = r[1];
        const idet::Point2f p2 = r[2];

        idet::Point2f tr = p1, bl = p2;
        if (p2.x > p1.x + kEpsLex || (absf(p2.x - p1.x) <= kEpsLex && p2.y < p1.y - kEpsLex)) {
            tr = p2;
            bl = p1;
        }

        quad[0] = tl;
        quad[1] = tr;
        quad[2] = br;
        quad[3] = bl;
        return;
    }

    // 5) rotate so first is TL (top-most then left-most)
    int i_tl = 0;
    for (int i = 1; i < 4; ++i) {
        if (lex_yx_less(r[i], r[i_tl])) i_tl = i;
    }

    idet::Quad t;
    t[0] = r[(i_tl + 0) & 3];
    t[1] = r[(i_tl + 1) & 3];
    t[2] = r[(i_tl + 2) & 3];
    t[3] = r[(i_tl + 3) & 3];

    // 6) disambiguate TR vs BL among neighbors (t[1], t[3])
    const bool t1_lower = (t[1].y > t[3].y + kEpsLex);
    const bool same_y = (absf(t[1].y - t[3].y) <= kEpsLex);
    const bool t1_left = (t[1].x < t[3].x - kEpsLex);
    if (t1_lower || (same_y && t1_left)) std::swap(t[1], t[3]);

    quad[0] = t[0];
    quad[1] = t[1];
    quad[2] = t[2];
    quad[3] = t[3];
}

} // namespace idet::algo

namespace idet::internal::opencv_geometry {

float contour_score(const cv::Mat& prob, const std::vector<cv::Point>& contour, ContourScoreScratch& scratch) {
    if (contour.empty()) return 0.f;

    cv::Rect bbox = cv::boundingRect(contour) & cv::Rect(0, 0, prob.cols, prob.rows);
    if (bbox.empty()) return 0.f;

    // Reuse a single full-size mask buffer across calls. We re-create only when the size or type
    // doesn't match (typically only on the first call per detector run). The reused buffer is
    // cleared lazily — only inside the bbox ROI for this contour — so previous draws outside the
    // current bbox are irrelevant.
    if (scratch.mask_full.rows != prob.rows || scratch.mask_full.cols != prob.cols ||
        scratch.mask_full.type() != CV_8U) {
        scratch.mask_full.create(prob.rows, prob.cols, CV_8U);
    }
    cv::Mat mask_roi = scratch.mask_full(bbox);
    mask_roi.setTo(cv::Scalar(0));

    if (scratch.cnt.empty()) scratch.cnt.resize(1);
    auto& cnt0 = scratch.cnt[0];
    cnt0.clear();
    cnt0.reserve(contour.size());

    for (const auto& p_orig : contour) {
        cv::Point p = p_orig;

        if (p.x < bbox.x)
            p.x = bbox.x;
        else if (p.x >= bbox.x + bbox.width)
            p.x = bbox.x + bbox.width - 1;

        if (p.y < bbox.y)
            p.y = bbox.y;
        else if (p.y >= bbox.y + bbox.height)
            p.y = bbox.y + bbox.height - 1;

        cnt0.push_back(p - bbox.tl());
    }

    cv::drawContours(mask_roi, scratch.cnt, 0, cv::Scalar(255), cv::FILLED);
    cv::Mat roi = prob(bbox);
    cv::Scalar m = cv::mean(roi, mask_roi);
    return static_cast<float>(m[0]);
}

float contour_score(const cv::Mat& prob, const std::vector<cv::Point>& contour) {
    // Backwards-compatible overload: allocate scratch on the stack/heap per call.
    ContourScoreScratch scratch;
    return contour_score(prob, contour, scratch);
}

} // namespace idet::internal::opencv_geometry

namespace idet::algo {

float aabb_iou(const idet::Quad& A, const idet::Quad& B) {
    auto is_finite = [](const idet::Point2f& p) noexcept { return std::isfinite(p.x) && std::isfinite(p.y); };

    for (int i = 0; i < 4; ++i) {
        if (!is_finite(A[i]) || !is_finite(B[i])) return 0.f;
    }

    auto minmax = [](const idet::Quad& q) {
        float minx = q[0].x, miny = q[0].y, maxx = q[0].x, maxy = q[0].y;
        for (int i = 1; i < 4; ++i) {
            minx = std::min(minx, q[i].x);
            miny = std::min(miny, q[i].y);
            maxx = std::max(maxx, q[i].x);
            maxy = std::max(maxy, q[i].y);
        }
        return std::array<float, 4>{minx, miny, maxx, maxy};
    };

    auto a = minmax(A), b = minmax(B);

    const float aw = std::max(0.f, a[2] - a[0]);
    const float ah = std::max(0.f, a[3] - a[1]);
    const float bw = std::max(0.f, b[2] - b[0]);
    const float bh = std::max(0.f, b[3] - b[1]);

    const float interW = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0]));
    const float interH = std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const float inter = interW * interH;

    const float areaA = aw * ah;
    const float areaB = bw * bh;

    const float denom = areaA + areaB - inter;
    if (!(denom > 1e-6f)) return 0.f;

    float iou = inter / denom;
    if (!std::isfinite(iou)) return 0.f;
    if (iou < 0.f) iou = 0.f;
    if (iou > 1.f) iou = 1.f;
    return iou;
}

float quad_iou(const idet::Quad& A, const idet::Quad& B, bool use_fast_iou, QuadIouScratch& scratch) {
    if (use_fast_iou) return aabb_iou(A, B);

    auto is_finite = [](const idet::Point2f& p) noexcept { return std::isfinite(p.x) && std::isfinite(p.y); };

    for (int i = 0; i < 4; ++i) {
        if (!is_finite(A[i]) || !is_finite(B[i])) return 0.f;
    }

    auto& cv_pts = scratch.impl->pts;
    auto& cv_a = scratch.impl->a;
    auto& cv_b = scratch.impl->b;
    auto& cv_inter = scratch.impl->inter;
    cv_pts.reserve(4);
    cv_a.reserve(4);
    cv_b.reserve(4);
    cv_inter.clear();
    cv_inter.reserve(8);

    auto make_hull = [&](const idet::Quad& q, std::vector<cv::Point2f>& hull) -> bool {
        cv_pts.clear();
        for (const auto& p : q)
            cv_pts.push_back(idet::internal::opencv_adapter::to_cv(p));
        hull.clear();
        hull.reserve(4);

        cv::convexHull(cv_pts, hull, /*clockwise=*/true, /*returnPoints=*/true);

        if (hull.size() < 3) return false;
        const double area = std::abs(cv::contourArea(hull));
        return area > 1e-9;
    };

    if (!make_hull(A, cv_a) || !make_hull(B, cv_b)) return 0.f;

    float inter_area = (float)cv::intersectConvexConvex(cv_a, cv_b, cv_inter, /*handleNested=*/true);
    if (!(inter_area > 0.f) || !std::isfinite(inter_area)) return 0.f;

    const float areaA = (float)std::abs(cv::contourArea(cv_a));
    const float areaB = (float)std::abs(cv::contourArea(cv_b));
    if (!(areaA > 0.f) || !(areaB > 0.f)) return 0.f;

    const float cap = std::min(areaA, areaB);
    if (inter_area > cap) inter_area = cap;

    const float uni = areaA + areaB - inter_area;
    if (!(uni > 1e-12f) || !std::isfinite(uni)) return 0.f;

    float iou = inter_area / uni;
    if (!std::isfinite(iou)) return 0.f;

    if (iou < 0.f) iou = 0.f;
    if (iou > 1.f) iou = 1.f;

    return iou;
}

float quad_iou(const idet::Quad& A, const idet::Quad& B, bool use_fast_iou) {
    // Backwards-compatible overload: allocate scratch on the stack per call.
    QuadIouScratch scratch;
    return quad_iou(A, B, use_fast_iou, scratch);
}

std::pair<int, int> aspect_fit32(const int iw, const int ih, const int side) {
    auto align_down32_safe = [](int v) {
        v = std::max(32, v);
        return v & ~31;
    };

    if (iw <= 0 || ih <= 0) return {32, 32};

    if (side <= 0) {
        return {align_down32_safe(iw), align_down32_safe(ih)};
    }

    const int m = std::max(iw, ih);
    const float s = (m > side ? float(side) / float(m) : 1.0f);

    int nw = std::max(1, (int)std::lround(iw * s));
    int nh = std::max(1, (int)std::lround(ih * s));

    nw = align_down32_safe(nw);
    nh = align_down32_safe(nh);
    return {nw, nh};
}

} // namespace idet::algo
