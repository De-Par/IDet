/**
 * @file test_chw_preprocess.cpp
 * @brief Unit tests for OpenCV-free BGR_U8 to CHW preprocessing helpers.
 */

#if defined(__has_include) && __has_include(<gtest/gtest.h>)
    #include <gtest/gtest.h>
#elif defined(__has_include) && __has_include(<gtest.h>)
    #include <gtest.h>
#else
    #error "[ERROR] 'gtest.h' header not found"
#endif

#include "internal/chw_preprocess.h"

#include <array>
#include <cstdint>
#include <vector>

using idet::internal::bgr_u8_to_chw_f32_letterbox;
using idet::internal::bgr_u8_to_chw_f32_same_size;
using idet::internal::BgrImageView;

TEST(ChwPreprocess, SameSizeConvertsInterleavedBgrToPlanarChw) {
    const std::array<std::uint8_t, 8> pixels{
        10, 20, 30, 40, 50, 60, 0xEE, 0xEE, // two pixels + row padding
    };
    const BgrImageView bgr{pixels.data(), 2, 1, 8};

    const float mean[3] = {0.0f, 0.0f, 0.0f};
    const float inv_std[3] = {1.0f, 1.0f, 1.0f};
    std::array<float, 6> chw{};

    bgr_u8_to_chw_f32_same_size(bgr, chw.data(), mean, inv_std);

    EXPECT_FLOAT_EQ(chw[0], 10.0f);
    EXPECT_FLOAT_EQ(chw[1], 40.0f);
    EXPECT_FLOAT_EQ(chw[2], 20.0f);
    EXPECT_FLOAT_EQ(chw[3], 50.0f);
    EXPECT_FLOAT_EQ(chw[4], 30.0f);
    EXPECT_FLOAT_EQ(chw[5], 60.0f);
}

TEST(ChwPreprocess, LetterboxFillsPaddingWithNormalizedPadValue) {
    const std::array<std::uint8_t, 6> pixels{10, 20, 30, 40, 50, 60};
    const BgrImageView bgr{pixels.data(), 2, 1, 6};

    const float mean[3] = {0.0f, 10.0f, 100.0f};
    const float inv_std[3] = {1.0f, 0.5f, 0.25f};
    std::vector<float> chw(3U * 4U * 4U, -1.0f);

    const auto info = bgr_u8_to_chw_f32_letterbox(bgr, 4, 4, /*pad_value=*/8, chw.data(), mean, inv_std);

    EXPECT_EQ(info.dst_w, 4);
    EXPECT_EQ(info.dst_h, 4);
    EXPECT_EQ(info.resized_w, 4);
    EXPECT_EQ(info.resized_h, 2);
    EXPECT_FLOAT_EQ(info.scale, 2.0f);

    const std::size_t plane = 16;
    const std::size_t bottom_row = 3U * 4U;
    for (std::size_t x = 0; x < 4; ++x) {
        EXPECT_FLOAT_EQ(chw[0U * plane + bottom_row + x], 8.0f);
        EXPECT_FLOAT_EQ(chw[1U * plane + bottom_row + x], -1.0f);
        EXPECT_FLOAT_EQ(chw[2U * plane + bottom_row + x], -23.0f);
    }
}

TEST(ChwPreprocess, LetterboxInvalidInputProducesPaddedTensor) {
    const BgrImageView invalid{};
    const float mean[3] = {1.0f, 2.0f, 3.0f};
    const float inv_std[3] = {1.0f, 1.0f, 1.0f};
    std::array<float, 12> chw{};

    const auto info = bgr_u8_to_chw_f32_letterbox(invalid, 2, 2, /*pad_value=*/7, chw.data(), mean, inv_std);

    EXPECT_EQ(info.resized_w, 0);
    EXPECT_EQ(info.resized_h, 0);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(chw[i], 6.0f);
        EXPECT_FLOAT_EQ(chw[4 + i], 5.0f);
        EXPECT_FLOAT_EQ(chw[8 + i], 4.0f);
    }
}
