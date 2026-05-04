/**
 * @file geometry.cpp
 * @ingroup idet_algo
 * @brief Implementations for quad ordering, IoU, and aspect-fit helpers.
 *
 * @details
 * Implements:
 *  - order_quad(): robust canonical ordering TL,TR,BR,BL with fallbacks for degenerate input,
 *  - aabb_iou(): fast axis-aligned IoU approximation from quad extents,
 *  - quad_iou(): exact convex IoU via polygon clipping (or AABB approximation when USE_FAST_IOU=1),
 *  - aspect_fit32(): aspect-ratio fit to a square side + 32-alignment.
 *
 * Notes:
 *  - Exact quad_iou() relies on convex hulls; for invalid/degenerate inputs returns 0.
 *  - Many routines include NaN/Inf guards to keep behavior deterministic in production.
 */

#include "algo/geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace idet::algo {

struct QuadIouScratch::Impl {
    std::vector<idet::Point2f> a;
    std::vector<idet::Point2f> b;
    std::vector<idet::Point2f> tmp;
    std::vector<idet::Point2f> inter;
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
    for (std::size_t i = 0; i < 4; ++i) {
        if (!is_finite(quad[i])) {
            idet::Quad r = {quad[0], quad[1], quad[2], quad[3]};

            auto swap_lex = [&](std::size_t i0, std::size_t i1) noexcept {
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
    for (std::size_t i = 0; i < 4; ++i) {
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
    auto swap_if = [&](std::size_t i0, std::size_t i1) noexcept {
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
        for (std::size_t i = 0; i < 4; ++i) {
            const std::size_t j = (i + 1) & 3U;
            a += p[i].x * p[j].y - p[j].x * p[i].y;
        }
        return a;
    };

    const float a2 = poly_area2(r);
    const float deg_thr = eps_cross;

    if (absf(a2) <= deg_thr) {
        // fallback: lex sort + TL/BR + split remaining
        auto swap_lex = [&](std::size_t i0, std::size_t i1) noexcept {
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
    std::size_t i_tl = 0;
    for (std::size_t i = 1; i < 4; ++i) {
        if (lex_yx_less(r[i], r[i_tl])) i_tl = i;
    }

    idet::Quad t;
    t[0] = r[(i_tl + 0U) & 3U];
    t[1] = r[(i_tl + 1U) & 3U];
    t[2] = r[(i_tl + 2U) & 3U];
    t[3] = r[(i_tl + 3U) & 3U];

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

namespace {

static inline bool finite_pt_(const idet::Point2f& p) noexcept {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

static inline float cross_(const idet::Point2f& a, const idet::Point2f& b, const idet::Point2f& c) noexcept {
    const float abx = b.x - a.x;
    const float aby = b.y - a.y;
    const float acx = c.x - a.x;
    const float acy = c.y - a.y;
    return abx * acy - aby * acx;
}

static double polygon_area_signed_(const std::vector<idet::Point2f>& poly) noexcept {
    const std::size_t n = poly.size();
    if (n < 3) return 0.0;

    double a = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = poly[i];
        const auto& q = poly[(i + 1U) % n];
        a += static_cast<double>(p.x) * static_cast<double>(q.y) - static_cast<double>(q.x) * static_cast<double>(p.y);
    }
    return 0.5 * a;
}

static inline float polygon_area_(const std::vector<idet::Point2f>& poly) noexcept {
    const double a = std::fabs(polygon_area_signed_(poly));
    if (!std::isfinite(a) || a <= 0.0) return 0.0f;
    return static_cast<float>(a);
}

static bool make_convex_hull_(const idet::Quad& q, std::vector<idet::Point2f>& hull) {
    hull.clear();

    std::array<idet::Point2f, 4> pts{};
    std::size_t n = 0;
    for (const auto& p : q) {
        if (!finite_pt_(p)) return false;
        pts[n++] = p;
    }

    std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    });

    std::array<idet::Point2f, 4> uniq{};
    std::size_t m = 0;
    for (const auto& p : pts) {
        if (m == 0 || std::fabs(p.x - uniq[m - 1].x) > 1e-6f || std::fabs(p.y - uniq[m - 1].y) > 1e-6f) {
            uniq[m++] = p;
        }
    }
    if (m < 3) return false;

    std::array<idet::Point2f, 8> tmp{};
    std::size_t k = 0;
    for (std::size_t i = 0; i < m; ++i) {
        while (k >= 2 && cross_(tmp[k - 2], tmp[k - 1], uniq[i]) <= 1e-7f)
            --k;
        tmp[k++] = uniq[i];
    }
    const std::size_t lower = k;
    for (std::size_t ii = m - 1; ii > 0; --ii) {
        const auto& p = uniq[ii - 1];
        while (k > lower && cross_(tmp[k - 2], tmp[k - 1], p) <= 1e-7f)
            --k;
        tmp[k++] = p;
    }
    if (k > 1) --k; // remove duplicate first point
    if (k < 3) return false;

    hull.assign(tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(k));
    return polygon_area_(hull) > 1e-7f;
}

static inline bool inside_ccw_edge_(const idet::Point2f& a, const idet::Point2f& b, const idet::Point2f& p) noexcept {
    return cross_(a, b, p) >= -1e-5f;
}

static idet::Point2f line_intersection_(const idet::Point2f& s, const idet::Point2f& e, const idet::Point2f& a,
                                        const idet::Point2f& b) noexcept {
    const idet::Point2f r{e.x - s.x, e.y - s.y};
    const idet::Point2f edge{b.x - a.x, b.y - a.y};
    const float denom = r.x * edge.y - r.y * edge.x;
    if (std::fabs(denom) <= 1e-12f) return e;

    const idet::Point2f q{s.x - a.x, s.y - a.y};
    const float t = (edge.x * q.y - edge.y * q.x) / denom;
    return {s.x + t * r.x, s.y + t * r.y};
}

static void clip_convex_(const std::vector<idet::Point2f>& subject, const std::vector<idet::Point2f>& clip,
                         std::vector<idet::Point2f>& out, std::vector<idet::Point2f>& tmp) {
    out = subject;

    for (std::size_t ci = 0; ci < clip.size() && !out.empty(); ++ci) {
        const idet::Point2f a = clip[ci];
        const idet::Point2f b = clip[(ci + 1U) % clip.size()];

        tmp.clear();
        tmp.reserve(out.size() + clip.size());

        idet::Point2f s = out.back();
        bool s_inside = inside_ccw_edge_(a, b, s);

        for (const auto& e : out) {
            const bool e_inside = inside_ccw_edge_(a, b, e);

            if (e_inside) {
                if (!s_inside) tmp.push_back(line_intersection_(s, e, a, b));
                tmp.push_back(e);
            } else if (s_inside) {
                tmp.push_back(line_intersection_(s, e, a, b));
            }

            s = e;
            s_inside = e_inside;
        }

        out.swap(tmp);
    }
}

} // namespace

float aabb_iou(const idet::Quad& A, const idet::Quad& B) {
    auto is_finite = [](const idet::Point2f& p) noexcept { return std::isfinite(p.x) && std::isfinite(p.y); };

    for (std::size_t i = 0; i < A.size(); ++i) {
        if (!is_finite(A[i]) || !is_finite(B[i])) return 0.f;
    }

    auto minmax = [](const idet::Quad& q) {
        float minx = q[0].x, miny = q[0].y, maxx = q[0].x, maxy = q[0].y;
        for (std::size_t i = 1; i < q.size(); ++i) {
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

    auto& poly_a = scratch.impl->a;
    auto& poly_b = scratch.impl->b;
    auto& tmp = scratch.impl->tmp;
    auto& inter = scratch.impl->inter;

    if (!make_convex_hull_(A, poly_a) || !make_convex_hull_(B, poly_b)) return 0.f;

    const float areaA = polygon_area_(poly_a);
    const float areaB = polygon_area_(poly_b);
    if (!(areaA > 0.f) || !(areaB > 0.f)) return 0.f;

    clip_convex_(poly_a, poly_b, inter, tmp);

    float inter_area = polygon_area_(inter);
    if (!(inter_area > 0.f) || !std::isfinite(inter_area)) return 0.f;

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
