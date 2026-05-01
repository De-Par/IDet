/**
 * @file test_bgr_image.cpp
 * @brief Unit tests for the OpenCV-free BGR image view/holder boundary.
 */

#if defined(__has_include) && __has_include(<gtest/gtest.h>)
    #include <gtest/gtest.h>
#elif defined(__has_include) && __has_include(<gtest.h>)
    #include <gtest.h>
#else
    #error "[ERROR] 'gtest.h' header not found"
#endif

#include "internal/bgr_image.h"

#include <array>
#include <cstdint>
#include <limits>

using idet::Image;
using idet::ImageView;
using idet::PixelFormat;
using idet::internal::BgrImage;
using idet::internal::BgrImageView;

TEST(BgrImageView, ValidatesStrideAndPointerOffsetRange) {
    std::array<std::uint8_t, 6> pixels{};

    BgrImageView ok{pixels.data(), 2, 1, 6};
    EXPECT_TRUE(ok.is_valid());

    BgrImageView short_stride{pixels.data(), 2, 1, 5};
    EXPECT_FALSE(short_stride.is_valid());

    BgrImageView huge_span{pixels.data(), 1, 2, std::numeric_limits<std::ptrdiff_t>::max()};
    EXPECT_FALSE(huge_span.is_valid());
}

TEST(BgrImageView, RoiRejectsOverflowingRectangleWithoutPointerMath) {
    std::array<std::uint8_t, 1> pixels{};

    BgrImageView huge_width{pixels.data(), std::numeric_limits<int>::max(), 1,
                            static_cast<std::ptrdiff_t>(std::numeric_limits<int>::max()) * 3};
    ASSERT_TRUE(huge_width.is_valid());

    BgrImageView roi = huge_width.roi(std::numeric_limits<int>::max() - 1, 0, 2, 1);
    EXPECT_FALSE(roi.is_valid());
}

TEST(BgrImage, ConvertsRgbToOwnedBgr) {
    std::array<std::uint8_t, 3> rgb{1, 2, 3};
    Image img = Image::view(ImageView{rgb.data(), 1, 1, 3, PixelFormat::RGB_U8});

    auto converted = BgrImage::from(std::move(img));
    ASSERT_TRUE(converted.ok()) << converted.status().message;

    const BgrImageView& bgr = converted.value().view();
    ASSERT_TRUE(bgr.is_valid());
    EXPECT_EQ(bgr.data[0], 3);
    EXPECT_EQ(bgr.data[1], 2);
    EXPECT_EQ(bgr.data[2], 1);
}

TEST(BgrImage, RejectsSourceSpanThatCannotBeRepresentedSafely) {
    std::array<std::uint8_t, 3> rgb{};
    Image img = Image::view(ImageView{
        rgb.data(), 1, 2, static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()), PixelFormat::RGB_U8});

    auto converted = BgrImage::from(std::move(img));
    EXPECT_FALSE(converted.ok());
}

TEST(BgrImage, RejectsBgrStrideThatCannotBeRepresentedSafely) {
    std::array<std::uint8_t, 3> bgr{};
    const std::size_t oversized_stride = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) + 1U;
    Image img = Image::view(ImageView{bgr.data(), 1, 1, oversized_stride, PixelFormat::BGR_U8});

    auto held = BgrImage::from(std::move(img));
    EXPECT_FALSE(held.ok());
}
