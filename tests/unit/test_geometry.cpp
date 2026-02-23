#if defined(__has_include) && __has_include(<gtest/gtest.h>)
    #include <gtest/gtest.h>
#elif defined(__has_include) && __has_include(<gtest.h>)
    #include <gtest.h>
#else
    #error "[ERROR] 'gtest.h' header not found"
#endif

#include "algo/geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <vector>

namespace {

static inline bool finite_pt(const cv::Point2f& p) noexcept {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

static inline int count_finite(const cv::Point2f q[4]) noexcept {
    int c = 0;
    for (int i = 0; i < 4; ++i)
        c += finite_pt(q[i]) ? 1 : 0;
    return c;
}

static inline int count_nan_any(const cv::Point2f q[4]) noexcept {
    int c = 0;
    for (int i = 0; i < 4; ++i) {
        if (std::isnan(q[i].x) || std::isnan(q[i].y)) ++c;
    }
    return c;
}

static inline int count_inf_any(const cv::Point2f q[4]) noexcept {
    int c = 0;
    for (int i = 0; i < 4; ++i) {
        if (std::isinf(q[i].x) || std::isinf(q[i].y)) ++c;
    }
    return c;
}

static inline bool lex_yx_less(const cv::Point2f& a, const cv::Point2f& b, float eps = 1e-4f) noexcept {
    // (y,x) lexicographic: top-most then left-most
    if (a.y < b.y - eps) return true;
    if (a.y > b.y + eps) return false;
    return a.x < b.x - eps;
}

static inline void expect_tl_is_lex_min(const cv::Point2f q[4]) {
    // TL must be lex-min among all points (weak invariant for finite inputs)
    for (int i = 1; i < 4; ++i) {
        EXPECT_FALSE(lex_yx_less(q[i], q[0])) << "q[0] is not lex-min (y,x)";
    }
}

static inline std::array<cv::Point2f, 4> make_rect(float x0, float y0, float x1, float y1) {
    // {TL, TR, BR, BL}
    return {{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}};
}

static inline void shuffle_quad(cv::Point2f q[4], std::uint32_t seed = 123) {
    std::array<int, 4> idx = {0, 1, 2, 3};
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    cv::Point2f tmp[4] = {q[idx[0]], q[idx[1]], q[idx[2]], q[idx[3]]};
    for (int i = 0; i < 4; ++i)
        q[i] = tmp[i];
}

static inline void expect_is_tl_tr_br_bl_rect(const cv::Point2f q[4]) {
    // TL is top-most then left-most among all 4
    for (int i = 1; i < 4; ++i) {
        ASSERT_FALSE(lex_yx_less(q[i], q[0])) << "q[0] is not TL";
    }
    // BR is bottom-most then right-most (not lex-less than any other)
    for (int i = 0; i < 4; ++i) {
        if (i == 2) continue;
        ASSERT_FALSE(lex_yx_less(q[2], q[i])) << "q[2] is not BR";
    }
}

static inline void copy_quad(cv::Point2f dst[4], const cv::Point2f src[4]) {
    for (int i = 0; i < 4; ++i)
        dst[i] = src[i];
}

static inline std::multiset<std::pair<float, float>> as_multiset(const cv::Point2f q[4]) {
    std::multiset<std::pair<float, float>> s;
    // compare as sets for exact float constants (safe for our crafted cases)
    for (int i = 0; i < 4; ++i)
        s.insert({q[i].x, q[i].y});
    return s;
}

static inline std::array<cv::Point2f, 4> to_array(const cv::Point2f q[4]) {
    return {q[0], q[1], q[2], q[3]};
}

// Generates a convex quad around a random center by picking 4 sorted angles and radii
// Returns points in arbitrary order (permuted) to exercise order_quad()
static std::array<cv::Point2f, 4> random_convex_quad(std::mt19937& rng) {
    constexpr float kPi = 3.14159265358979323846f;

    std::uniform_real_distribution<float> center_d(-200.f, 200.f);
    std::uniform_real_distribution<float> ang_d(0.f, 2.f * kPi);
    std::uniform_real_distribution<float> rad_d(20.f, 160.f);

    const cv::Point2f c(center_d(rng), center_d(rng));

    std::array<float, 4> a = {ang_d(rng), ang_d(rng), ang_d(rng), ang_d(rng)};
    std::sort(a.begin(), a.end());

    std::array<cv::Point2f, 4> q;
    for (int i = 0; i < 4; ++i) {
        const float r = rad_d(rng);
        q[i] = cv::Point2f(c.x + r * std::cos(a[i]), c.y + r * std::sin(a[i]));
    }

    // Permute to simulate arbitrary model output order
    std::array<int, 4> idx = {0, 1, 2, 3};
    std::shuffle(idx.begin(), idx.end(), rng);

    std::array<cv::Point2f, 4> out = {q[idx[0]], q[idx[1]], q[idx[2]], q[idx[3]]};
    return out;
}

static inline bool in_unit_interval_soft(float v) noexcept {
    // numerical tolerance: OpenCV intersection area computations can produce tiny eps deviations
    return std::isfinite(v) && v >= -1e-4f && v <= 1.0f + 1e-4f;
}

static inline bool to_strict_convex_quad_cw(const std::array<cv::Point2f, 4>& in, std::array<cv::Point2f, 4>& out) {
    std::vector<cv::Point2f> pts(in.begin(), in.end());
    std::vector<cv::Point2f> hull;
    cv::convexHull(pts, hull, /*clockwise=*/true, /*returnPoints=*/true);

    if (hull.size() != 4) return false;
    if (std::abs(cv::contourArea(hull)) < 1e-2) return false;

    for (int i = 0; i < 4; ++i)
        out[i] = hull[i];
    return true;
}

} // namespace

// ------------------------------- order_quad ----------------------------------

// Pipeline check: same quad, different permutations => after order_quad IoU must be 1
TEST(Geometry, OrderQuad_ThenQuadIou_SameShapeDifferentPermutation_IsOne) {
    // A non-axis-aligned convex quad
    cv::Point2f base[4] = {
        {30.f, 10.f},
        {80.f, 25.f},
        {70.f, 70.f},
        {20.f, 55.f},
    };

    cv::Point2f q1[4] = {base[0], base[1], base[2], base[3]};
    cv::Point2f q2[4] = {base[0], base[1], base[2], base[3]};

    shuffle_quad(q1, 111);
    shuffle_quad(q2, 777);

    idet::algo::order_quad(q1);
    idet::algo::order_quad(q2);

    const auto A = to_array(q1);
    const auto B = to_array(q2);

    const float iou = idet::algo::quad_iou(A, B);
    EXPECT_NEAR(iou, 1.0f, 1e-5f);
}

TEST(Geometry, OrderQuad_RectAxisAligned) {
    cv::Point2f q[4] = {
        {10.f, 20.f}, // tl
        {10.f, 80.f}, // bl
        {60.f, 80.f}, // br
        {60.f, 20.f}, // tr
    };

    std::swap(q[1], q[3]);
    idet::algo::order_quad(q);

    EXPECT_FLOAT_EQ(q[0].x, 10.f);
    EXPECT_FLOAT_EQ(q[0].y, 20.f);
    EXPECT_FLOAT_EQ(q[1].x, 60.f);
    EXPECT_FLOAT_EQ(q[1].y, 20.f);
    EXPECT_FLOAT_EQ(q[2].x, 60.f);
    EXPECT_FLOAT_EQ(q[2].y, 80.f);
    EXPECT_FLOAT_EQ(q[3].x, 10.f);
    EXPECT_FLOAT_EQ(q[3].y, 80.f);

    expect_is_tl_tr_br_bl_rect(q);
    expect_tl_is_lex_min(q);
}

TEST(Geometry, OrderQuad_Rect_ShuffledManyTimes_IsStable) {
    const auto r = make_rect(10.f, 20.f, 60.f, 80.f);
    for (std::uint32_t seed = 1; seed <= 80; ++seed) {
        cv::Point2f q[4] = {r[0], r[1], r[2], r[3]};
        shuffle_quad(q, seed);

        idet::algo::order_quad(q);

        EXPECT_FLOAT_EQ(q[0].x, 10.f);
        EXPECT_FLOAT_EQ(q[0].y, 20.f);
        EXPECT_FLOAT_EQ(q[1].x, 60.f);
        EXPECT_FLOAT_EQ(q[1].y, 20.f);
        EXPECT_FLOAT_EQ(q[2].x, 60.f);
        EXPECT_FLOAT_EQ(q[2].y, 80.f);
        EXPECT_FLOAT_EQ(q[3].x, 10.f);
        EXPECT_FLOAT_EQ(q[3].y, 80.f);
    }
}

TEST(Geometry, OrderQuad_Parallelogram_Rotated_AllFinite_TLIsLexMin) {
    cv::Point2f q[4] = {
        {30.f, 10.f},
        {80.f, 25.f},
        {70.f, 70.f},
        {20.f, 55.f},
    };
    shuffle_quad(q, 7);

    idet::algo::order_quad(q);

    ASSERT_EQ(count_finite(q), 4);
    expect_tl_is_lex_min(q);
}

TEST(Geometry, OrderQuad_NegativeCoords_Rect) {
    cv::Point2f q[4] = {
        {-10.f, -20.f}, // tl
        {-10.f, 80.f},  // bl
        {60.f, 80.f},   // br
        {60.f, -20.f},  // tr
    };
    shuffle_quad(q, 13);

    idet::algo::order_quad(q);

    EXPECT_FLOAT_EQ(q[0].x, -10.f);
    EXPECT_FLOAT_EQ(q[0].y, -20.f);
    EXPECT_FLOAT_EQ(q[1].x, 60.f);
    EXPECT_FLOAT_EQ(q[1].y, -20.f);
    EXPECT_FLOAT_EQ(q[2].x, 60.f);
    EXPECT_FLOAT_EQ(q[2].y, 80.f);
    EXPECT_FLOAT_EQ(q[3].x, -10.f);
    EXPECT_FLOAT_EQ(q[3].y, 80.f);
}

TEST(Geometry, OrderQuad_DuplicatePoints_DoesNotCrash_PermutationPreserved) {
    // Two points identical (degenerate)
    cv::Point2f q[4] = {
        {0.f, 0.f},
        {10.f, 0.f},
        {10.f, 10.f},
        {10.f, 10.f}, // duplicate
    };
    const auto in_set = as_multiset(q);
    shuffle_quad(q, 99);

    idet::algo::order_quad(q);

    // should not introduce NaN/Inf
    ASSERT_EQ(count_finite(q), 4);
    // should be a permutation (for exact constants)
    EXPECT_EQ(as_multiset(q), in_set);
    // TL should still be lex-min
    expect_tl_is_lex_min(q);
}

TEST(Geometry, OrderQuad_Idempotent_ForFiniteInput) {
    cv::Point2f q[4] = {
        {30.f, 10.f},
        {80.f, 25.f},
        {70.f, 70.f},
        {20.f, 55.f},
    };
    shuffle_quad(q, 5);

    idet::algo::order_quad(q);
    cv::Point2f once[4];
    copy_quad(once, q);

    idet::algo::order_quad(q);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(q[i].x, once[i].x);
        EXPECT_FLOAT_EQ(q[i].y, once[i].y);
    }
}

TEST(Geometry, OrderQuad_LargeMagnitudeCoords_StaysFinite) {
    cv::Point2f q[4] = {
        {1e8f, 1e8f},
        {1e8f + 1000.f, 1e8f + 10.f},
        {1e8f + 900.f, 1e8f + 2000.f},
        {1e8f - 50.f, 1e8f + 1500.f},
    };
    shuffle_quad(q, 1234);
    idet::algo::order_quad(q);
    ASSERT_EQ(count_finite(q), 4);
    expect_tl_is_lex_min(q);
}

TEST(Geometry, OrderQuad_Degenerate_Collinear_DoesNotCrash_AllFinite) {
    cv::Point2f q[4] = {
        {0.f, 0.f},
        {10.f, 0.0001f},
        {20.f, 0.0002f},
        {30.f, 0.0003f},
    };
    shuffle_quad(q, 42);

    idet::algo::order_quad(q);
    ASSERT_EQ(count_finite(q), 4);
}

TEST(Geometry, OrderQuad_WithNaN_DoesNotCreateExtraNonFinite) {
    cv::Point2f q[4] = {
        {10.f, 20.f},
        {std::numeric_limits<float>::quiet_NaN(), 30.f},
        {60.f, 80.f},
        {10.f, 80.f},
    };

    const int nan0 = count_nan_any(q);
    const int inf0 = count_inf_any(q);
    const int fin0 = count_finite(q);

    idet::algo::order_quad(q);

    EXPECT_EQ(count_nan_any(q), nan0);
    EXPECT_EQ(count_inf_any(q), inf0);
    EXPECT_EQ(count_finite(q), fin0);
}

TEST(Geometry, OrderQuad_WithInf_DoesNotCreateExtraNonFinite) {
    cv::Point2f q[4] = {
        {10.f, 20.f},
        {10.f, 80.f},
        {std::numeric_limits<float>::infinity(), 80.f},
        {60.f, 20.f},
    };

    const int nan0 = count_nan_any(q);
    const int inf0 = count_inf_any(q);
    const int fin0 = count_finite(q);

    idet::algo::order_quad(q);

    EXPECT_EQ(count_nan_any(q), nan0);
    EXPECT_EQ(count_inf_any(q), inf0);
    EXPECT_EQ(count_finite(q), fin0);
}

// ------------------------------- contour_score --------------------------------

TEST(Geometry, ContourScore_EmptyContour_IsZero) {
    cv::Mat prob(10, 10, CV_32F, cv::Scalar(0.5f));
    std::vector<cv::Point> contour;
    EXPECT_FLOAT_EQ(idet::algo::contour_score(prob, contour), 0.f);
}

TEST(Geometry, ContourScore_Rect_EqualsMeanUnderMask) {
    const int W = 8, H = 6;
    cv::Mat prob(H, W, CV_32F);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            prob.at<float>(y, x) = float(x + 10 * y);
        }
    }
    std::vector<cv::Point> contour = {{2, 1}, {5, 1}, {5, 4}, {2, 4}};

    cv::Mat mask(H, W, CV_8U, cv::Scalar(0));
    std::vector<std::vector<cv::Point>> cnt(1);
    cnt[0] = contour;
    cv::drawContours(mask, cnt, 0, cv::Scalar(255), cv::FILLED);

    const float ref = (float)cv::mean(prob, mask)[0];
    const float got = idet::algo::contour_score(prob, contour);

    EXPECT_NEAR(got, ref, 1e-6f);
}

TEST(Geometry, ContourScore_ContourOutsideBounds_DoesNotCrash_Finite) {
    cv::Mat prob(10, 10, CV_32F, cv::Scalar(0.2f));
    std::vector<cv::Point> contour = {{-100, -100}, {20, -100}, {20, 20}, {-100, 20}};
    const float got = idet::algo::contour_score(prob, contour);
    EXPECT_TRUE(std::isfinite(got));
    EXPECT_GE(got, 0.f);
}

// ------------------------------- aabb_iou ------------------------------------

TEST(Geometry, AabbIou_IdenticalIsOne) {
    const auto A = make_rect(0.f, 0.f, 10.f, 10.f);
    EXPECT_NEAR(idet::algo::aabb_iou(A, A), 1.0f, 1e-6f);
}

TEST(Geometry, AabbIou_DisjointIsZero) {
    const auto A = make_rect(0.f, 0.f, 10.f, 10.f);
    const auto B = make_rect(100.f, 100.f, 110.f, 110.f);
    EXPECT_NEAR(idet::algo::aabb_iou(A, B), 0.0f, 1e-6f);
}

TEST(Geometry, AabbIou_TouchingEdges_IsZero) {
    const auto A = make_rect(0.f, 0.f, 10.f, 10.f);
    const auto B = make_rect(10.f, 0.f, 20.f, 10.f);
    EXPECT_NEAR(idet::algo::aabb_iou(A, B), 0.0f, 1e-6f);
}

TEST(Geometry, AabbIou_ContainedBox_MatchesExpected) {
    const auto A = make_rect(0.f, 0.f, 10.f, 10.f);
    const auto B = make_rect(3.f, 3.f, 7.f, 7.f);
    EXPECT_NEAR(idet::algo::aabb_iou(A, B), 16.0f / 100.0f, 1e-6f);
}

TEST(Geometry, AabbIou_SymmetricProperty_Holds) {
    const auto A = make_rect(0.f, 0.f, 10.f, 10.f);
    const auto B = make_rect(5.f, 2.f, 12.f, 9.f);
    EXPECT_NEAR(idet::algo::aabb_iou(A, B), idet::algo::aabb_iou(B, A), 1e-6f);
}

TEST(Geometry, AabbIou_DegenerateZeroArea_ReturnsZero) {
    std::array<cv::Point2f, 4> A = {{{1.f, 1.f}, {1.f, 1.f}, {1.f, 1.f}, {1.f, 1.f}}};
    const auto B = make_rect(0.f, 0.f, 10.f, 10.f);
    EXPECT_NEAR(idet::algo::aabb_iou(A, B), 0.0f, 1e-6f);
}

// ------------------------------- quad_iou ------------------------------------

TEST(Geometry, QuadIou_UnorderedPoints_StillInRange) {
    std::array<cv::Point2f, 4> A = {{{0, 0}, {10, 10}, {0, 10}, {10, 0}}};
    std::array<cv::Point2f, 4> B = {{{0, 0}, {10, 0}, {10, 10}, {0, 10}}};

    const float iou = idet::algo::quad_iou(A, B);
    EXPECT_TRUE(std::isfinite(iou));
    EXPECT_GE(iou, 0.f);
    EXPECT_LE(iou, 1.f);
}

// Fuzz: random convex quads => IoU must be finite and within [0,1] (soft),
// and symmetric: iou(A,B) ~= iou(B,A)
TEST(Geometry, QuadIou_RandomConvex_FiniteInRange_AndSymmetric) {
    std::mt19937 rng(123456);

    int accepted = 0;
    const int target = 300;
    const int max_tries = 20000;

    for (int tries = 0; tries < max_tries && accepted < target; ++tries) {
        const auto qa0 = random_convex_quad(rng);
        const auto qb0 = random_convex_quad(rng);

        std::array<cv::Point2f, 4> A, B;
        if (!to_strict_convex_quad_cw(qa0, A)) continue;
        if (!to_strict_convex_quad_cw(qb0, B)) continue;

        const float ab = idet::algo::quad_iou(A, B);
        const float ba = idet::algo::quad_iou(B, A);

        EXPECT_TRUE(in_unit_interval_soft(ab)) << "ab=" << ab;
        EXPECT_TRUE(in_unit_interval_soft(ba)) << "ba=" << ba;
        EXPECT_NEAR(ab, ba, 1e-4f);

        ++accepted;
    }

    ASSERT_EQ(accepted, target) << "not enough strict convex quads (hull.size()==4)";
}

TEST(Geometry, QuadIou_IdenticalIsOne) {
    const auto A = make_rect(0.f, 0.f, 10.f, 10.f);
    EXPECT_NEAR(idet::algo::quad_iou(A, A), 1.0f, 1e-5f);
}

TEST(Geometry, QuadIou_PartialOverlap) {
    const auto A = make_rect(0.f, 0.f, 10.f, 10.f);
    const auto B = make_rect(5.f, 0.f, 15.f, 10.f);
    EXPECT_NEAR(idet::algo::quad_iou(A, B), 50.0f / 150.0f, 1e-3f);
}

TEST(Geometry, QuadIou_Disjoint_IsZero) {
    const auto A = make_rect(0.f, 0.f, 10.f, 10.f);
    const auto B = make_rect(20.f, 20.f, 30.f, 30.f);
    EXPECT_NEAR(idet::algo::quad_iou(A, B), 0.0f, 1e-6f);
}

TEST(Geometry, QuadIou_SymmetricProperty_Holds) {
    std::array<cv::Point2f, 4> A = {{{30.f, 10.f}, {80.f, 25.f}, {70.f, 70.f}, {20.f, 55.f}}};
    std::array<cv::Point2f, 4> B = {{{40.f, 15.f}, {90.f, 30.f}, {75.f, 75.f}, {25.f, 60.f}}};
    EXPECT_NEAR(idet::algo::quad_iou(A, B), idet::algo::quad_iou(B, A), 1e-5f);
}

TEST(Geometry, QuadIou_DegenerateZeroArea_ReturnsZero) {
    std::array<cv::Point2f, 4> A = {{{1.f, 1.f}, {1.f, 1.f}, {1.f, 1.f}, {1.f, 1.f}}};
    const auto B = make_rect(0.f, 0.f, 10.f, 10.f);
    EXPECT_NEAR(idet::algo::quad_iou(A, B), 0.0f, 1e-6f);
}

TEST(Geometry, QuadIou_TranslationInvariance) {
    const auto A0 = make_rect(0.f, 0.f, 10.f, 10.f);
    const auto B0 = make_rect(5.f, 0.f, 15.f, 10.f);
    const float base = idet::algo::quad_iou(A0, B0);

    const float dx = 123.4f;
    const float dy = -77.0f;

    std::array<cv::Point2f, 4> A1 = A0, B1 = B0;
    for (int i = 0; i < 4; ++i) {
        A1[i].x += dx;
        A1[i].y += dy;
        B1[i].x += dx;
        B1[i].y += dy;
    }

    EXPECT_NEAR(idet::algo::quad_iou(A1, B1), base, 1e-5f);
}

// ------------------------------ aspect_fit32 ---------------------------------

TEST(Geometry, AspectFit32_InvalidInput_Returns32) {
    auto r = idet::algo::aspect_fit32(0, 0, 960);
    EXPECT_EQ(r.first, 32);
    EXPECT_EQ(r.second, 32);
}

TEST(Geometry, AspectFit32_SideNonPositive_AlignsDownTo32) {
    auto r = idet::algo::aspect_fit32(100, 70, 0);
    EXPECT_EQ(r.first % 32, 0);
    EXPECT_EQ(r.second % 32, 0);
    EXPECT_GE(r.first, 32);
    EXPECT_GE(r.second, 32);
    EXPECT_LE(r.first, 100);
    EXPECT_LE(r.second, 70);
}

TEST(Geometry, AspectFit32_NoUpscale_WhenAlreadyBelowSide) {
    auto r = idet::algo::aspect_fit32(80, 60, 200);
    EXPECT_LE(r.first, 80);
    EXPECT_LE(r.second, 60);
    EXPECT_EQ(r.first % 32, 0);
    EXPECT_EQ(r.second % 32, 0);
}
