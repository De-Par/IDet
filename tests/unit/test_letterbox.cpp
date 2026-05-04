/**
 * @file test_letterbox.cpp
 * @brief Unit tests for OpenCV-free aspect-preserving letterbox preprocessing.
 */

#if defined(__has_include) && __has_include(<gtest/gtest.h>)
    #include <gtest/gtest.h>
#elif defined(__has_include) && __has_include(<gtest.h>)
    #include <gtest.h>
#else
    #error "[ERROR] 'gtest.h' header not found"
#endif

#include "internal/chw_preprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

using idet::internal::bgr_u8_to_chw_f32_letterbox;
using idet::internal::BgrImageView;
using idet::internal::LetterboxInfo;

namespace {

struct TestBgr {
    std::vector<std::uint8_t> data;
    int width = 0;
    int height = 0;

    [[nodiscard]] BgrImageView view() const noexcept {
        return {data.data(), width, height, static_cast<std::ptrdiff_t>(width * 3)};
    }
};

static TestBgr make_test_bgr(int w, int h, std::uint8_t base) {
    TestBgr out;
    out.width = w;
    out.height = h;
    out.data.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3U);
    for (int y = 0; y < h; ++y) {
        std::uint8_t* row = out.data.data() + static_cast<std::ptrdiff_t>(y) * w * 3;
        for (int x = 0; x < w; ++x) {
            const int x3 = x * 3;
            row[x3 + 0] = static_cast<std::uint8_t>(base + (x & 0xFF));
            row[x3 + 1] = static_cast<std::uint8_t>(base + (y & 0xFF));
            row[x3 + 2] = static_cast<std::uint8_t>((x ^ y) & 0xFF);
        }
    }
    return out;
}

static LetterboxInfo run_letterbox(const TestBgr& src, std::vector<float>& chw, int dst_w, int dst_h,
                                   std::uint8_t pad) {
    const float mean[3] = {0.0f, 0.0f, 0.0f};
    const float inv_std[3] = {1.0f, 1.0f, 1.0f};
    chw.assign(static_cast<std::size_t>(std::max(dst_w, 0)) * static_cast<std::size_t>(std::max(dst_h, 0)) * 3U, -1.0f);
    return bgr_u8_to_chw_f32_letterbox(src.view(), dst_w, dst_h, pad, chw.data(), mean, inv_std);
}

static float chw_at(const std::vector<float>& chw, int c, int x, int y, int w, int h) {
    const auto plane = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    return chw[static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
               static_cast<std::size_t>(x)];
}

} // namespace

TEST(Letterbox, PreservesAspectRatio_LandscapeIntoSquare) {
    auto src = make_test_bgr(800, 200, 0);
    std::vector<float> chw;
    auto info = run_letterbox(src, chw, 320, 320, /*pad=*/114);

    EXPECT_EQ(info.dst_w, 320);
    EXPECT_EQ(info.dst_h, 320);

    // 800x200 into 320x320 => scale = 0.4, content = 320x80
    EXPECT_NEAR(info.scale, 0.4f, 1e-5f);
    EXPECT_EQ(info.resized_w, 320);
    EXPECT_EQ(info.resized_h, 80);
    EXPECT_EQ(info.pad_x, 0);
    EXPECT_EQ(info.pad_y, 0);
}

TEST(Letterbox, PreservesAspectRatio_PortraitIntoSquare) {
    auto src = make_test_bgr(200, 800, 0);
    std::vector<float> chw;
    auto info = run_letterbox(src, chw, 320, 320, /*pad=*/114);

    EXPECT_NEAR(info.scale, 0.4f, 1e-5f);
    EXPECT_EQ(info.resized_w, 80);
    EXPECT_EQ(info.resized_h, 320);
    EXPECT_EQ(info.pad_x, 0);
    EXPECT_EQ(info.pad_y, 0);
}

TEST(Letterbox, PaddingHasExpectedValueOutsideContent) {
    auto src = make_test_bgr(800, 200, 0);
    std::vector<float> chw;
    constexpr std::uint8_t pad = 114;
    auto info = run_letterbox(src, chw, 320, 320, pad);
    ASSERT_GT(info.resized_h, 0);
    ASSERT_LT(info.resized_h, info.dst_h);

    for (int y = info.resized_h; y < info.dst_h; ++y) {
        const int x = info.dst_w / 2;
        EXPECT_FLOAT_EQ(chw_at(chw, 0, x, y, info.dst_w, info.dst_h), static_cast<float>(pad));
        EXPECT_FLOAT_EQ(chw_at(chw, 1, x, y, info.dst_w, info.dst_h), static_cast<float>(pad));
        EXPECT_FLOAT_EQ(chw_at(chw, 2, x, y, info.dst_w, info.dst_h), static_cast<float>(pad));
    }
}

TEST(Letterbox, InverseMappingRecoversSourceCoordinates) {
    auto src = make_test_bgr(640, 480, 0);
    std::vector<float> chw;
    auto info = run_letterbox(src, chw, 320, 320, /*pad=*/0);

    EXPECT_NEAR(info.scale, 0.5f, 1e-5f);

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
    const float mean[3] = {0.0f, 0.0f, 0.0f};
    const float inv_std[3] = {1.0f, 1.0f, 1.0f};
    std::vector<float> chw(3U * 64U * 32U, -1.0f);

    auto info = bgr_u8_to_chw_f32_letterbox(BgrImageView{}, 64, 32, /*pad_value=*/7, chw.data(), mean, inv_std);

    EXPECT_EQ(info.dst_w, 64);
    EXPECT_EQ(info.dst_h, 32);
    EXPECT_EQ(info.scale, 1.0f);
    EXPECT_EQ(info.pad_x, 0);
    EXPECT_EQ(info.pad_y, 0);
    EXPECT_EQ(info.resized_w, 0);
    EXPECT_EQ(info.resized_h, 0);

    EXPECT_FLOAT_EQ(chw_at(chw, 0, 30, 15, 64, 32), 7.0f);
    EXPECT_FLOAT_EQ(chw_at(chw, 1, 30, 15, 64, 32), 7.0f);
    EXPECT_FLOAT_EQ(chw_at(chw, 2, 30, 15, 64, 32), 7.0f);
}

TEST(Letterbox, NonPositiveSizeLeavesOutputUntouched) {
    auto src = make_test_bgr(100, 100, 0);
    const float mean[3] = {0.0f, 0.0f, 0.0f};
    const float inv_std[3] = {1.0f, 1.0f, 1.0f};
    std::array<float, 3> sentinel{42.0f, 43.0f, 44.0f};

    auto info = bgr_u8_to_chw_f32_letterbox(src.view(), 0, 32, /*pad_value=*/0, sentinel.data(), mean, inv_std);

    EXPECT_EQ(info.dst_w, 0);
    EXPECT_EQ(info.dst_h, 32);
    EXPECT_EQ(info.resized_w, 0);
    EXPECT_EQ(info.resized_h, 0);
    EXPECT_FLOAT_EQ(sentinel[0], 42.0f);
    EXPECT_FLOAT_EQ(sentinel[1], 43.0f);
    EXPECT_FLOAT_EQ(sentinel[2], 44.0f);
}

TEST(Letterbox, SquareIntoSquareIsIdentityShape) {
    auto src = make_test_bgr(256, 256, 0);
    std::vector<float> chw;
    auto info = run_letterbox(src, chw, 256, 256, /*pad=*/0);

    EXPECT_NEAR(info.scale, 1.0f, 1e-5f);
    EXPECT_EQ(info.resized_w, 256);
    EXPECT_EQ(info.resized_h, 256);
    EXPECT_EQ(info.pad_x, 0);
    EXPECT_EQ(info.pad_y, 0);
}

TEST(Letterbox, ScaleIsTheMinOfPerAxisRatios) {
    auto src = make_test_bgr(1000, 100, 0);
    std::vector<float> chw;
    auto info = run_letterbox(src, chw, 320, 320, /*pad=*/0);

    EXPECT_NEAR(info.scale, 0.32f, 1e-5f);
}
