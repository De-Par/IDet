/**
 * @file test_letterbox.cpp
 * @brief Unit tests for the aspect-preserving letterbox helper used by DBNet/SCRFD/YOLO.
 *
 * @details
 * Covers:
 * - aspect ratio is preserved across both axes (uniform scale),
 * - resized content fits inside the destination buffer,
 * - padding is placed only outside the resized content,
 * - the reverse mapping ((x_net - pad) / scale) recovers source pixel coordinates,
 * - degenerate inputs (empty source, non-positive sizes) are handled.
 */

#if defined(__has_include) && __has_include(<gtest/gtest.h>)
    #include <gtest/gtest.h>
#elif defined(__has_include) && __has_include(<gtest.h>)
    #include <gtest.h>
#else
    #error "[ERROR] 'gtest.h' header not found"
#endif

#include "internal/letterbox.h"

#include <cmath>

using idet::internal::letterbox_bgr;
using idet::internal::LetterboxInfo;

namespace {

/** @brief Make a synthetic BGR image with a per-pixel signature so we can verify resize behavior. */
static cv::Mat make_test_bgr(int w, int h, std::uint8_t base) {
    cv::Mat m(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y) {
        auto* row = m.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x) {
            row[x] = cv::Vec3b(static_cast<std::uint8_t>(base + (x & 0xFF)),
                               static_cast<std::uint8_t>(base + (y & 0xFF)), static_cast<std::uint8_t>((x ^ y) & 0xFF));
        }
    }
    return m;
}

} // namespace

TEST(Letterbox, PreservesAspectRatio_LandscapeIntoSquare) {
    auto src = make_test_bgr(800, 200, 0);
    cv::Mat dst;
    auto info = letterbox_bgr(src, dst, 320, 320, /*pad=*/114);

    EXPECT_EQ(dst.cols, 320);
    EXPECT_EQ(dst.rows, 320);
    EXPECT_EQ(dst.type(), CV_8UC3);

    // 800x200 into 320x320 => scale = 0.4, content = 320x80
    EXPECT_NEAR(info.scale, 0.4f, 1e-5f);
    EXPECT_EQ(info.resized_w, 320);
    EXPECT_EQ(info.resized_h, 80);
    EXPECT_EQ(info.pad_x, 0);
    EXPECT_EQ(info.pad_y, 0); // top-left padding -> resized content starts at (0, 0)
}

TEST(Letterbox, PreservesAspectRatio_PortraitIntoSquare) {
    auto src = make_test_bgr(200, 800, 0);
    cv::Mat dst;
    auto info = letterbox_bgr(src, dst, 320, 320, /*pad=*/114);

    EXPECT_NEAR(info.scale, 0.4f, 1e-5f);
    EXPECT_EQ(info.resized_w, 80);
    EXPECT_EQ(info.resized_h, 320);
    EXPECT_EQ(info.pad_x, 0);
    EXPECT_EQ(info.pad_y, 0);
}

TEST(Letterbox, PaddingHasExpectedValueOutsideContent) {
    auto src = make_test_bgr(800, 200, 0);
    cv::Mat dst;
    const std::uint8_t pad = 114;
    auto info = letterbox_bgr(src, dst, 320, 320, pad);
    ASSERT_GT(info.resized_h, 0);
    ASSERT_LT(info.resized_h, dst.rows);

    // Inspect the rows below the resized content (top-left convention).
    for (int y = info.resized_h; y < dst.rows; ++y) {
        const auto& px = dst.at<cv::Vec3b>(y, dst.cols / 2);
        EXPECT_EQ(px[0], pad);
        EXPECT_EQ(px[1], pad);
        EXPECT_EQ(px[2], pad);
    }
}

TEST(Letterbox, InverseMappingRecoversSourceCoordinates) {
    auto src = make_test_bgr(640, 480, 0);
    cv::Mat dst;
    auto info = letterbox_bgr(src, dst, 320, 320, /*pad=*/0);

    // 640x480 into 320x320 -> scale = 0.5, content 320x240, no pad needed on x, pad_y = 0 (top-left).
    EXPECT_NEAR(info.scale, 0.5f, 1e-5f);

    // Pick a few source points, project them forward, then recover via the inverse.
    const std::array<std::pair<int, int>, 5> srcs = {{{0, 0}, {639, 479}, {320, 240}, {100, 50}, {500, 300}}};
    for (auto [sx, sy] : srcs) {
        const float xn = sx * info.scale + info.pad_x;
        const float yn = sy * info.scale + info.pad_y;

        const float ix = (xn - info.pad_x) / info.scale;
        const float iy = (yn - info.pad_y) / info.scale;
        EXPECT_NEAR(ix, static_cast<float>(sx), 1e-3f);
        EXPECT_NEAR(iy, static_cast<float>(sy), 1e-3f);
    }
}

TEST(Letterbox, EmptyOrInvalidSourceProducesPaddedBuffer) {
    cv::Mat empty_src;
    cv::Mat dst;
    auto info = letterbox_bgr(empty_src, dst, 64, 32, /*pad=*/7);

    EXPECT_EQ(dst.cols, 64);
    EXPECT_EQ(dst.rows, 32);
    EXPECT_EQ(dst.type(), CV_8UC3);
    EXPECT_EQ(info.scale, 1.0f);
    EXPECT_EQ(info.pad_x, 0);
    EXPECT_EQ(info.pad_y, 0);

    const auto& px = dst.at<cv::Vec3b>(15, 30);
    EXPECT_EQ(px[0], 7);
    EXPECT_EQ(px[1], 7);
    EXPECT_EQ(px[2], 7);
}

TEST(Letterbox, NonPositiveSizeReleasesDst) {
    auto src = make_test_bgr(100, 100, 0);
    cv::Mat dst(10, 10, CV_8UC3);
    auto info = letterbox_bgr(src, dst, 0, 32, /*pad=*/0);
    EXPECT_TRUE(dst.empty());
    EXPECT_EQ(info.dst_w, 0);
}

TEST(Letterbox, SquareIntoSquareIsIdentityShape) {
    auto src = make_test_bgr(256, 256, 0);
    cv::Mat dst;
    auto info = letterbox_bgr(src, dst, 256, 256, /*pad=*/0);

    EXPECT_NEAR(info.scale, 1.0f, 1e-5f);
    EXPECT_EQ(info.resized_w, 256);
    EXPECT_EQ(info.resized_h, 256);
    EXPECT_EQ(info.pad_x, 0);
    EXPECT_EQ(info.pad_y, 0);
}

TEST(Letterbox, ScaleIsTheMinOfPerAxisRatios) {
    auto src = make_test_bgr(1000, 100, 0);
    cv::Mat dst;
    auto info = letterbox_bgr(src, dst, 320, 320, /*pad=*/0);
    // sx = 0.32, sy = 3.2 -> scale = 0.32 (downscale to fit width).
    EXPECT_NEAR(info.scale, 0.32f, 1e-5f);
}
