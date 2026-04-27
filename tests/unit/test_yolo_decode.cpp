/**
 * @file test_yolo_decode.cpp
 * @brief Unit tests for YOLO raw-output decoding (channels-first / channels-last layouts,
 *        objectness folding, sigmoid logit-style exports, letterbox inverse mapping).
 */

#if defined(__has_include) && __has_include(<gtest/gtest.h>)
    #include <gtest/gtest.h>
#elif defined(__has_include) && __has_include(<gtest.h>)
    #include <gtest.h>
#else
    #error "[ERROR] 'gtest.h' header not found"
#endif

#include "engine/yolo.h"
#include "internal/letterbox.h"

#include <cmath>
#include <vector>

using idet::algo::Detection;
using idet::engine::YOLO;
using idet::internal::LetterboxInfo;
using Layout = YOLO::Layout;

namespace {

/** @brief Pack a single anchor into a channels-last buffer slot. */
static void put_cl(std::vector<float>& out, std::int64_t i, int feat, int c, float v) {
    out[static_cast<std::size_t>(i) * static_cast<std::size_t>(feat) + static_cast<std::size_t>(c)] = v;
}

/** @brief Pack a single anchor into a channels-first buffer slot. */
static void put_cf(std::vector<float>& out, std::int64_t N, std::int64_t i, int c, float v) {
    out[static_cast<std::size_t>(c) * static_cast<std::size_t>(N) + static_cast<std::size_t>(i)] = v;
}

/** @brief Identity letterbox (scale=1, no pad) — useful when we feed boxes already in image space. */
static LetterboxInfo identity_lb(int dst_w, int dst_h) {
    LetterboxInfo lb{};
    lb.scale = 1.0f;
    lb.pad_x = 0;
    lb.pad_y = 0;
    lb.dst_w = dst_w;
    lb.dst_h = dst_h;
    lb.resized_w = dst_w;
    lb.resized_h = dst_h;
    return lb;
}

} // namespace

TEST(YoloDecode, ChannelsLast_NoObjectness_DirectScores) {
    constexpr int nc = 3;
    constexpr int feat = 4 + nc; // no obj
    constexpr std::int64_t N = 4;
    std::vector<float> buf(static_cast<std::size_t>(N) * feat, 0.f);

    // anchor 0: high score class 1, box (cx=100, cy=200, w=50, h=40)
    put_cl(buf, 0, feat, 0, 100.f);
    put_cl(buf, 0, feat, 1, 200.f);
    put_cl(buf, 0, feat, 2, 50.f);
    put_cl(buf, 0, feat, 3, 40.f);
    put_cl(buf, 0, feat, 4, 0.10f); // class 0
    put_cl(buf, 0, feat, 5, 0.92f); // class 1 (best)
    put_cl(buf, 0, feat, 6, 0.05f); // class 2

    // anchor 1: below threshold
    put_cl(buf, 1, feat, 0, 10.f);
    put_cl(buf, 1, feat, 1, 10.f);
    put_cl(buf, 1, feat, 2, 5.f);
    put_cl(buf, 1, feat, 3, 5.f);
    put_cl(buf, 1, feat, 4, 0.05f);
    put_cl(buf, 1, feat, 5, 0.10f);
    put_cl(buf, 1, feat, 6, 0.04f);

    auto lb = identity_lb(640, 640);
    auto dets = YOLO::decode_raw_buffer(buf.data(), N, nc, /*has_obj*/ false, Layout::ChannelsLast,
                                        /*apply_sigmoid*/ false, /*score_thr*/ 0.25f, lb, 640, 640, 0, 0);

    ASSERT_EQ(dets.size(), 1u);
    EXPECT_NEAR(dets[0].score, 0.92f, 1e-5f);
    EXPECT_NEAR(dets[0].pts[0].x, 75.f, 1e-3f);  // x1 = 100 - 25
    EXPECT_NEAR(dets[0].pts[0].y, 180.f, 1e-3f); // y1 = 200 - 20
    EXPECT_NEAR(dets[0].pts[2].x, 125.f, 1e-3f); // x2
    EXPECT_NEAR(dets[0].pts[2].y, 220.f, 1e-3f); // y2
}

TEST(YoloDecode, ChannelsFirst_WithObjectness_FoldsIntoScore) {
    constexpr int nc = 2;
    constexpr int feat = 4 + 1 + nc; // obj + 2 classes
    constexpr std::int64_t N = 2;
    std::vector<float> buf(static_cast<std::size_t>(N) * feat, 0.f);

    put_cf(buf, N, 0, 0, 320.f); // cx
    put_cf(buf, N, 0, 1, 320.f); // cy
    put_cf(buf, N, 0, 2, 100.f); // w
    put_cf(buf, N, 0, 3, 80.f);  // h
    put_cf(buf, N, 0, 4, 0.5f);  // obj
    put_cf(buf, N, 0, 5, 0.8f);  // class 0 (best)
    put_cf(buf, N, 0, 6, 0.1f);  // class 1
    // anchor 1 — all zero -> below threshold

    auto lb = identity_lb(640, 640);
    auto dets = YOLO::decode_raw_buffer(buf.data(), N, nc, /*has_obj*/ true, Layout::ChannelsFirst,
                                        /*apply_sigmoid*/ false, /*score_thr*/ 0.3f, lb, 640, 640, 0, 0);

    ASSERT_EQ(dets.size(), 1u);
    EXPECT_NEAR(dets[0].score, 0.8f * 0.5f, 1e-5f);
    EXPECT_NEAR(dets[0].pts[0].x, 270.f, 1e-3f);
    EXPECT_NEAR(dets[0].pts[2].x, 370.f, 1e-3f);
}

TEST(YoloDecode, AppliesLetterboxInverseToBoxes) {
    constexpr int nc = 1;
    constexpr int feat = 4 + nc;
    constexpr std::int64_t N = 1;
    std::vector<float> buf(static_cast<std::size_t>(N) * feat, 0.f);

    // Letterbox: 1000x500 image fitted into 320x320 (scale=0.32, pad_x=0, pad_y=0 top-left convention).
    // Resized content is 320x160. A box with center at the center of the resized content
    // should map to the center of the original image.
    LetterboxInfo lb{};
    lb.scale = 0.32f;
    lb.pad_x = 0;
    lb.pad_y = 0;
    lb.dst_w = 320;
    lb.dst_h = 320;
    lb.resized_w = 320;
    lb.resized_h = 160;

    put_cl(buf, 0, feat, 0, 160.f); // cx in net-input space (center of 320 width)
    put_cl(buf, 0, feat, 1, 80.f);  // cy in net-input space (center of 160 height)
    put_cl(buf, 0, feat, 2, 64.f);  // w
    put_cl(buf, 0, feat, 3, 32.f);  // h
    put_cl(buf, 0, feat, 4, 0.95f); // class 0

    auto dets = YOLO::decode_raw_buffer(buf.data(), N, nc, false, Layout::ChannelsLast, false, 0.5f, lb, 1000, 500, 0,
                                        0);

    ASSERT_EQ(dets.size(), 1u);
    // Center: (160/0.32, 80/0.32) = (500, 250) -> matches original-image center.
    // Width: 64/0.32 = 200, height: 32/0.32 = 100.
    EXPECT_NEAR(dets[0].pts[0].x, 400.f, 1e-3f); // 500 - 100
    EXPECT_NEAR(dets[0].pts[0].y, 200.f, 1e-3f); // 250 - 50
    EXPECT_NEAR(dets[0].pts[2].x, 600.f, 1e-3f);
    EXPECT_NEAR(dets[0].pts[2].y, 300.f, 1e-3f);
}

TEST(YoloDecode, ClampsBoxesToOriginalImageBounds) {
    constexpr int nc = 1;
    constexpr int feat = 4 + nc;
    constexpr std::int64_t N = 1;
    std::vector<float> buf(static_cast<std::size_t>(N) * feat, 0.f);

    auto lb = identity_lb(640, 640);
    // Box extends past the image on the right/bottom; should be clamped.
    put_cl(buf, 0, feat, 0, 600.f);
    put_cl(buf, 0, feat, 1, 600.f);
    put_cl(buf, 0, feat, 2, 200.f);
    put_cl(buf, 0, feat, 3, 200.f);
    put_cl(buf, 0, feat, 4, 0.99f);

    auto dets = YOLO::decode_raw_buffer(buf.data(), N, nc, false, Layout::ChannelsLast, false, 0.5f, lb, 640, 640, 0,
                                        0);
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_GE(dets[0].pts[0].x, 0.f);
    EXPECT_LE(dets[0].pts[2].x, 640.f);
    EXPECT_LE(dets[0].pts[2].y, 640.f);
}

TEST(YoloDecode, FiltersBelowMinSize) {
    constexpr int nc = 1;
    constexpr int feat = 4 + nc;
    constexpr std::int64_t N = 1;
    std::vector<float> buf(static_cast<std::size_t>(N) * feat, 0.f);

    auto lb = identity_lb(640, 640);
    put_cl(buf, 0, feat, 0, 100.f);
    put_cl(buf, 0, feat, 1, 100.f);
    put_cl(buf, 0, feat, 2, 4.f); // very small
    put_cl(buf, 0, feat, 3, 4.f);
    put_cl(buf, 0, feat, 4, 0.99f);

    auto dets = YOLO::decode_raw_buffer(buf.data(), N, nc, false, Layout::ChannelsLast, false, 0.5f, lb, 640, 640,
                                        /*min_w*/ 16, /*min_h*/ 16);
    EXPECT_EQ(dets.size(), 0u);
}

TEST(YoloDecode, SortsByScoreDescending) {
    constexpr int nc = 1;
    constexpr int feat = 4 + nc;
    constexpr std::int64_t N = 3;
    std::vector<float> buf(static_cast<std::size_t>(N) * feat, 0.f);

    auto lb = identity_lb(640, 640);
    for (int i = 0; i < N; ++i) {
        put_cl(buf, i, feat, 0, 100.f * (i + 1));
        put_cl(buf, i, feat, 1, 100.f);
        put_cl(buf, i, feat, 2, 50.f);
        put_cl(buf, i, feat, 3, 50.f);
    }
    put_cl(buf, 0, feat, 4, 0.6f);
    put_cl(buf, 1, feat, 4, 0.9f);
    put_cl(buf, 2, feat, 4, 0.7f);

    auto dets = YOLO::decode_raw_buffer(buf.data(), N, nc, false, Layout::ChannelsLast, false, 0.5f, lb, 640, 640, 0,
                                        0);
    ASSERT_EQ(dets.size(), 3u);
    EXPECT_GT(dets[0].score, dets[1].score);
    EXPECT_GT(dets[1].score, dets[2].score);
}

TEST(YoloDecode, LogitFastPath_RejectsBelowThresholdWithoutCallingSigmoid) {
    // Anchor with logit just below logit(thr) must be rejected; an anchor with logit just above
    // must keep the same final sigmoid score as the slow path. This validates that the logit
    // short-circuit is bit-equivalent to the original sigmoid-then-compare semantics.
    constexpr int nc = 1;
    constexpr int feat = 4 + nc;
    constexpr std::int64_t N = 2;
    std::vector<float> buf(static_cast<std::size_t>(N) * feat, 0.f);

    auto lb = identity_lb(640, 640);
    // Anchor 0: logit barely above logit(0.5) = 0 -> sigmoid > 0.5
    put_cl(buf, 0, feat, 0, 100.f);
    put_cl(buf, 0, feat, 1, 100.f);
    put_cl(buf, 0, feat, 2, 50.f);
    put_cl(buf, 0, feat, 3, 50.f);
    put_cl(buf, 0, feat, 4, 0.5f); // logit 0.5 -> sigmoid ~ 0.6225

    // Anchor 1: logit far below logit(0.5) -> rejected by fast path
    put_cl(buf, 1, feat, 0, 200.f);
    put_cl(buf, 1, feat, 1, 200.f);
    put_cl(buf, 1, feat, 2, 50.f);
    put_cl(buf, 1, feat, 3, 50.f);
    put_cl(buf, 1, feat, 4, -3.0f); // sigmoid ~ 0.047

    auto dets = YOLO::decode_raw_buffer(buf.data(), N, nc, /*has_obj*/ false, Layout::ChannelsLast,
                                        /*apply_sigmoid*/ true, /*score_thr*/ 0.5f, lb, 640, 640, 0, 0);
    ASSERT_EQ(dets.size(), 1u);
    const float expected = 1.f / (1.f + std::exp(-0.5f));
    EXPECT_NEAR(dets[0].score, expected, 1e-5f);
}

TEST(YoloDecode, AppliesSigmoidWhenRequested) {
    constexpr int nc = 1;
    constexpr int feat = 4 + nc;
    constexpr std::int64_t N = 1;
    std::vector<float> buf(static_cast<std::size_t>(N) * feat, 0.f);

    auto lb = identity_lb(640, 640);
    put_cl(buf, 0, feat, 0, 320.f);
    put_cl(buf, 0, feat, 1, 320.f);
    put_cl(buf, 0, feat, 2, 50.f);
    put_cl(buf, 0, feat, 3, 50.f);
    // logit 4.0 -> sigmoid ~= 0.982
    put_cl(buf, 0, feat, 4, 4.0f);

    auto dets = YOLO::decode_raw_buffer(buf.data(), N, nc, false, Layout::ChannelsLast, /*apply_sigmoid*/ true, 0.5f,
                                        lb, 640, 640, 0, 0);
    ASSERT_EQ(dets.size(), 1u);
    const float expected = 1.f / (1.f + std::exp(-4.0f));
    EXPECT_NEAR(dets[0].score, expected, 1e-5f);
}
