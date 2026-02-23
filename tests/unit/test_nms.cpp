#if defined(__has_include) && __has_include(<gtest/gtest.h>)
    #include <gtest/gtest.h>
#elif defined(__has_include) && __has_include(<gtest.h>)
    #include <gtest.h>
#else
    #error "[ERROR] 'gtest.h' header not found"
#endif

#include "algo/nms.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

static idet::algo::Detection rect(float x1, float y1, float x2, float y2, float score) {
    idet::algo::Detection d;
    d.score = score;
    d.pts[0] = {x1, y1};
    d.pts[1] = {x2, y1};
    d.pts[2] = {x2, y2};
    d.pts[3] = {x1, y2};
    return d;
}

static idet::algo::Detection diamond(float cx, float cy, float r, float score) {
    // convex quad rotated 45 degrees (a "diamond")
    idet::algo::Detection d;
    d.score = score;
    d.pts[0] = {cx, cy - r};
    d.pts[1] = {cx + r, cy};
    d.pts[2] = {cx, cy + r};
    d.pts[3] = {cx - r, cy};
    return d;
}

static bool is_sorted_desc(const std::vector<idet::algo::Detection>& v) {
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i - 1].score < v[i].score) return false;
    }
    return true;
}

static void expect_all_scores_present(const std::vector<idet::algo::Detection>& out, const std::vector<float>& scores) {
    // multiset-ish check for small test vectors
    std::vector<float> a, b;
    a.reserve(out.size());
    b = scores;
    for (auto& d : out)
        a.push_back(d.score);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        EXPECT_FLOAT_EQ(a[i], b[i]);
}

} // namespace

TEST(NMS, ThrLeZeroMeansSortedOnly) {
    std::vector<idet::algo::Detection> dets;
    dets.push_back(rect(0, 0, 10, 10, 0.2f));
    dets.push_back(rect(0, 0, 10, 10, 0.9f));
    dets.push_back(rect(0, 0, 10, 10, 0.5f));

    auto out = idet::algo::nms_poly(dets, 0.0f);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_TRUE(is_sorted_desc(out));
    EXPECT_FLOAT_EQ(out[0].score, 0.9f);
}

TEST(NMS, LargeThresholdBoundary_ExactlyOne_KeepsSingleBest) {
    std::vector<idet::algo::Detection> dets;
    dets.push_back(rect(0, 0, 10, 10, 0.2f));
    dets.push_back(rect(0, 0, 10, 10, 0.9f));
    dets.push_back(rect(0, 0, 10, 10, 0.5f));

    auto out = idet::algo::nms_poly(dets, 1.0f);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_FLOAT_EQ(out[0].score, 0.9f);
}

TEST(NMS, KeepsHighestAndSuppressesOverlap) {
    std::vector<idet::algo::Detection> dets;
    dets.push_back(rect(0, 0, 10, 10, 0.9f));
    dets.push_back(rect(1, 1, 9, 9, 0.8f));         // strong overlap
    dets.push_back(rect(100, 100, 110, 110, 0.7f)); // disjoint

    auto out = idet::algo::nms_poly(dets, 0.3f);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_TRUE(is_sorted_desc(out));
    EXPECT_FLOAT_EQ(out[0].score, 0.9f);
    EXPECT_FLOAT_EQ(out[1].score, 0.7f);
}

TEST(NMS, EmptyInput_ReturnsEmpty) {
    std::vector<idet::algo::Detection> dets;
    auto out = idet::algo::nms_poly(dets, 0.3f);
    EXPECT_TRUE(out.empty());
}

TEST(NMS, SingleBox_ReturnsSameBox) {
    std::vector<idet::algo::Detection> dets;
    dets.push_back(rect(0, 0, 10, 10, 0.42f));
    auto out = idet::algo::nms_poly(dets, 0.3f);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_FLOAT_EQ(out[0].score, 0.42f);
}

TEST(NMS, ThrNegative_TreatedAsSortedOnly) {
    // thr < 0 should behave like "no suppression": only sort
    std::vector<idet::algo::Detection> dets;
    dets.push_back(rect(0, 0, 10, 10, 0.3f));
    dets.push_back(rect(0, 0, 10, 10, 0.1f));
    dets.push_back(rect(0, 0, 10, 10, 0.8f));

    auto out = idet::algo::nms_poly(dets, -1.0f);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_TRUE(is_sorted_desc(out));
    expect_all_scores_present(out, {0.8f, 0.3f, 0.1f});
}

TEST(NMS, ThrAboveOne_TreatedAsKeepSingleBest) {
    std::vector<idet::algo::Detection> dets;
    dets.push_back(rect(0, 0, 10, 10, 0.3f));
    dets.push_back(rect(0, 0, 10, 10, 0.9f));
    dets.push_back(rect(0, 0, 10, 10, 0.7f));

    auto out = idet::algo::nms_poly(dets, 1.5f);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_FLOAT_EQ(out[0].score, 0.9f);
}

TEST(NMS, AlreadySortedInput_PreservesSortedOrderAfterNMS) {
    // if input already sorted, output should still be sorted and contain a prefix-ish set
    std::vector<idet::algo::Detection> dets;
    dets.push_back(rect(0, 0, 10, 10, 0.9f));
    dets.push_back(rect(1, 1, 9, 9, 0.8f));     // overlaps
    dets.push_back(rect(2, 2, 8, 8, 0.7f));     // overlaps even more
    dets.push_back(rect(50, 50, 60, 60, 0.6f)); // disjoint

    auto out = idet::algo::nms_poly(dets, 0.3f);
    ASSERT_GE(out.size(), 1u);
    EXPECT_TRUE(is_sorted_desc(out));
    EXPECT_FLOAT_EQ(out[0].score, 0.9f);
}

TEST(NMS, EqualScores_DeterministicSizeAndSorted) {
    // Two identical boxes with equal score: whichever kept, size must be 1 for thr>=0 and overlap high
    std::vector<idet::algo::Detection> dets;
    dets.push_back(rect(0, 0, 10, 10, 0.5f));
    dets.push_back(rect(0, 0, 10, 10, 0.5f));
    dets.push_back(rect(100, 100, 110, 110, 0.5f)); // disjoint

    auto out = idet::algo::nms_poly(dets, 0.3f);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_TRUE(is_sorted_desc(out));
    // all kept are 0.5
    EXPECT_FLOAT_EQ(out[0].score, 0.5f);
    EXPECT_FLOAT_EQ(out[1].score, 0.5f);
}

TEST(NMS, TouchingEdges_NoSuppressionWhenIoUZero) {
    // A and B touch at boundary -> intersection area 0 -> IoU 0 -> both should survive for thr>0
    std::vector<idet::algo::Detection> dets;
    dets.push_back(rect(0, 0, 10, 10, 0.9f));
    dets.push_back(rect(10, 0, 20, 10, 0.8f)); // touch at x=10
    auto out = idet::algo::nms_poly(dets, 0.1f);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_TRUE(is_sorted_desc(out));
    EXPECT_FLOAT_EQ(out[0].score, 0.9f);
    EXPECT_FLOAT_EQ(out[1].score, 0.8f);
}

TEST(NMS, ContainedBox_SuppressedForModerateThreshold) {
    // B inside A: IoU = area(B)/area(A) = 64/100 = 0.64, should be suppressed if thr <= 0.64
    std::vector<idet::algo::Detection> dets;
    dets.push_back(rect(0, 0, 10, 10, 0.9f)); // area 100
    dets.push_back(rect(1, 1, 9, 9, 0.8f));   // area 64, inside
    auto out = idet::algo::nms_poly(dets, 0.5f);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_FLOAT_EQ(out[0].score, 0.9f);
}

TEST(NMS, ContainedBox_NotSuppressedForHighThreshold) {
    // Same setup, but thr very high -> keep both
    std::vector<idet::algo::Detection> dets;
    dets.push_back(rect(0, 0, 10, 10, 0.9f));
    dets.push_back(rect(1, 1, 9, 9, 0.8f));
    auto out = idet::algo::nms_poly(dets, 0.99f);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_TRUE(is_sorted_desc(out));
    EXPECT_FLOAT_EQ(out[0].score, 0.9f);
    EXPECT_FLOAT_EQ(out[1].score, 0.8f);
}

TEST(NMS, RotatedQuads_DiamondOverlap_SuppressesLower) {
    // This checks polygon IoU path (not just AABB) assuming your nms_poly uses quad_iou
    // Two diamonds with strong overlap
    std::vector<idet::algo::Detection> dets;
    dets.push_back(diamond(0.f, 0.f, 10.f, 0.9f));
    dets.push_back(diamond(0.1f, 0.f, 10.f, 0.8f));   // shifted slightly => high overlap
    dets.push_back(diamond(100.f, 100.f, 5.f, 0.7f)); // disjoint

    auto out = idet::algo::nms_poly(dets, 0.3f);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_TRUE(is_sorted_desc(out));
    EXPECT_FLOAT_EQ(out[0].score, 0.9f);
    EXPECT_FLOAT_EQ(out[1].score, 0.7f);
}

TEST(NMS, DegenerateZeroAreaBoxes_DoNotCrash_StillSorted) {
    std::vector<idet::algo::Detection> dets;
    dets.push_back(rect(0, 0, 0, 10, 0.9f));  // zero width
    dets.push_back(rect(0, 0, 10, 0, 0.8f));  // zero height
    dets.push_back(rect(0, 0, 10, 10, 0.7f)); // normal
    dets.push_back(rect(0, 0, 10, 10, 0.6f)); // normal overlapping

    auto out = idet::algo::nms_poly(dets, 0.3f);
    ASSERT_FALSE(out.empty());
    EXPECT_TRUE(is_sorted_desc(out));

    bool has_07 = false;
    for (auto& d : out)
        has_07 |= (d.score == 0.7f);
    EXPECT_TRUE(has_07);
}

TEST(NMS, NaNScore_HandledDeterministically_NoCrash) {
    // If your NMS sorts by score, NaNs can break strict weak ordering
    // This test is mainly to catch crashes/UB; behavior may be defined by comparator
    std::vector<idet::algo::Detection> dets;
    dets.push_back(rect(0, 0, 10, 10, 0.9f));
    dets.push_back(rect(0, 0, 10, 10, std::numeric_limits<float>::quiet_NaN()));
    dets.push_back(rect(100, 100, 110, 110, 0.7f));

    auto out = idet::algo::nms_poly(dets, 0.3f);
    // Just ensure it returns something and doesn't crash
    EXPECT_FALSE(out.empty());
    // If NaNs are present, you may want a policy: treat as -inf or 0
    // If you implement that, tighten this test accordingly
}
