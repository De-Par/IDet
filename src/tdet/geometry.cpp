#include "geometry.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#if defined(__APPLE__)
#include <opencv2/imgproc.hpp>
#else
#include <opencv4/opencv2/imgproc.hpp>
#endif

// Robust quad ordering: [top-left, top-right, bottom-right, bottom-left]
void order_quad(cv::Point2f pts[4]) {
    // Try sum/diff method first (fast & robust for rectangles/quads)
    int i_tl = 0, i_tr = 0, i_br = 0, i_bl = 0;
    float minSum = pts[0].x + pts[0].y, maxSum = minSum;
    float minDiff = pts[0].x - pts[0].y, maxDiff = minDiff;

    for (int i = 1; i < 4; ++i) {
        const float sum = pts[i].x + pts[i].y;
        const float diff = pts[i].x - pts[i].y;
        if (sum < minSum) {
            minSum = sum;
            i_tl = i;
        }
        if (sum > maxSum) {
            maxSum = sum;
            i_br = i;
        }
        if (diff < minDiff) {
            minDiff = diff;
            i_tr = i;
        }
        if (diff > maxDiff) {
            maxDiff = diff;
            i_bl = i;
        }
    }

    const bool dup =
        (i_tl == i_tr) || (i_tl == i_br) || (i_tl == i_bl) || (i_tr == i_br) || (i_tr == i_bl) || (i_br == i_bl);

    if (!dup) {
        cv::Point2f out[4] = {pts[i_tl], pts[i_tr], pts[i_br], pts[i_bl]};
        for (int i = 0; i < 4; ++i)
            pts[i] = out[i];
        return;
    }

    // Fallback: angle sort around center + rotate to TL
    cv::Point2f c(0.f, 0.f);
    for (int i = 0; i < 4; ++i)
        c += pts[i];
    c *= 0.25f;

    std::array<cv::Point2f, 4> a = {pts[0], pts[1], pts[2], pts[3]};
    std::sort(a.begin(), a.end(), [&](const cv::Point2f& p1, const cv::Point2f& p2) {
        const float ang1 = std::atan2(p1.y - c.y, p1.x - c.x);
        const float ang2 = std::atan2(p2.y - c.y, p2.x - c.x);
        return ang1 < ang2;
    });

    int tl = 0;
    for (int i = 1; i < 4; ++i) {
        if (a[i].y < a[tl].y - 1e-4f || (std::abs(a[i].y - a[tl].y) <= 1e-4f && a[i].x < a[tl].x)) tl = i;
    }

    // rotate so that a[0] is TL
    std::array<cv::Point2f, 4> r;
    for (int i = 0; i < 4; ++i)
        r[i] = a[(tl + i) & 3];

    // decide TR/BL by which neighbor is more to the right from TL
    // we want [TL, TR, BR, BL]
    const cv::Point2f v1 = r[1] - r[0];
    const cv::Point2f v3 = r[3] - r[0];
    if (v1.x < v3.x) {
        // swap r[1] <-> r[3]
        std::swap(r[1], r[3]);
    }

    for (int i = 0; i < 4; ++i)
        pts[i] = r[i];
}

static inline float poly_area_vec(const std::vector<cv::Point2f>& p) {
    if (p.size() < 3) return 0.f;
    double a = 0.0;
    for (size_t i = 0, j = p.size() - 1; i < p.size(); j = i++)
        a += (double)p[j].x * p[i].y - (double)p[i].x * p[j].y;
    return static_cast<float>(std::abs(a) * 0.5);
}

// Reuse buffers to reduce allocations
float contour_score(const cv::Mat& prob, const std::vector<cv::Point>& contour) {
    if (contour.empty()) return 0.f;

    cv::Rect bbox = cv::boundingRect(contour) & cv::Rect(0, 0, prob.cols, prob.rows);
    if (bbox.empty()) return 0.f;

    thread_local cv::Mat mask;
    mask.create(bbox.size(), CV_8U);
    mask.setTo(0);

    thread_local std::vector<std::vector<cv::Point>> cnt(1);
    cnt[0].clear();
    cnt[0].reserve(contour.size());

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

        cnt[0].push_back(p - bbox.tl());
    }

    cv::drawContours(mask, cnt, 0, cv::Scalar(255), cv::FILLED);
    cv::Mat roi = prob(bbox);
    cv::Scalar m = cv::mean(roi, mask);
    return static_cast<float>(m[0]);
}

// Fast IoU approximation by AABB for speedup (enough for CPU NMS if enabled)
float aabb_iou(const std::array<cv::Point2f, 4>& A, const std::array<cv::Point2f, 4>& B) {
    auto minmax = [](const std::array<cv::Point2f, 4>& q) {
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
    const float interW = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0]));
    const float interH = std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const float inter = interW * interH;
    const float areaA = (a[2] - a[0]) * (a[3] - a[1]);
    const float areaB = (b[2] - b[0]) * (b[3] - b[1]);
    const float denom = areaA + areaB - inter;
    return (denom > 1e-6f) ? inter / denom : 0.f;
}

static inline float quad_area4(const std::array<cv::Point2f, 4>& q) {
    // assumes consistent winding (order_quad enforces)
    double a = 0.0;
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) & 3;
        a += (double)q[i].x * q[j].y - (double)q[j].x * q[i].y;
    }
    return static_cast<float>(std::abs(a) * 0.5);
}

// IoU with OpenCV intersectConvexConvex (optimized: no per-call allocations)
float quad_iou(const std::array<cv::Point2f, 4>& A, const std::array<cv::Point2f, 4>& B) {
#if USE_FAST_IOU
    return aabb_iou(A, B);
#else
    thread_local std::vector<cv::Point2f> a(4), b(4), inter;
    a[0] = A[0];
    a[1] = A[1];
    a[2] = A[2];
    a[3] = A[3];
    b[0] = B[0];
    b[1] = B[1];
    b[2] = B[2];
    b[3] = B[3];
    inter.clear();
    inter.reserve(8);

    const float inter_area = (float)cv::intersectConvexConvex(a, b, inter, true);
    if (inter_area <= 0.f) return 0.f;

    const float ua = quad_area4(A) + quad_area4(B) - inter_area;
    return ua > 1e-12f ? inter_area / ua : 0.f;
#endif
}

// aspect-fit to `side` and align to 32 (safe: never returns 0)
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