#if defined(__has_include) && __has_include(<gtest/gtest.h>)
    #include <gtest/gtest.h>
#elif defined(__has_include) && __has_include(<gtest.h>)
    #include <gtest.h>
#else
    #error "[ERROR] 'gtest.h' header not found"
#endif

#include "internal/ort_tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using idet::internal::TensorDesc;
using idet::internal::TensorLayout;

// ----------------------------- make_desc_probmap -----------------------------------------------

TEST(OrtTensor, MakeDescProbmap_NCHW) {
    const std::vector<int64_t> sh = {1, 1, 7, 9};
    TensorDesc d = idet::internal::make_desc_probmap(sh);

    EXPECT_EQ(d.layout, TensorLayout::NCHW);
    EXPECT_EQ(d.N, 1);
    EXPECT_EQ(d.C, 1);
    EXPECT_EQ(d.H, 7);
    EXPECT_EQ(d.W, 9);
    EXPECT_EQ(d.numel, (std::size_t)1 * 1 * 7 * 9);
}

TEST(OrtTensor, MakeDescProbmap_NHWC) {
    const std::vector<int64_t> sh = {1, 7, 9, 1};
    TensorDesc d = idet::internal::make_desc_probmap(sh);

    EXPECT_EQ(d.layout, TensorLayout::NHWC);
    EXPECT_EQ(d.N, 1);
    EXPECT_EQ(d.C, 1);
    EXPECT_EQ(d.H, 7);
    EXPECT_EQ(d.W, 9);
    EXPECT_EQ(d.numel, (std::size_t)1 * 7 * 9 * 1);
}

TEST(OrtTensor, MakeDescProbmap_N1HW) {
    const std::vector<int64_t> sh = {1, 7, 9};
    TensorDesc d = idet::internal::make_desc_probmap(sh);

    EXPECT_EQ(d.layout, TensorLayout::N1HW);
    EXPECT_EQ(d.N, 1);
    EXPECT_EQ(d.C, 1);
    EXPECT_EQ(d.H, 7);
    EXPECT_EQ(d.W, 9);
    EXPECT_EQ(d.numel, (std::size_t)1 * 7 * 9);
}

TEST(OrtTensor, MakeDescProbmap_HW) {
    const std::vector<int64_t> sh = {7, 9};
    TensorDesc d = idet::internal::make_desc_probmap(sh);

    EXPECT_EQ(d.layout, TensorLayout::HW);
    EXPECT_EQ(d.N, 1);
    EXPECT_EQ(d.C, 1);
    EXPECT_EQ(d.H, 7);
    EXPECT_EQ(d.W, 9);
    EXPECT_EQ(d.numel, (std::size_t)7 * 9);
}

TEST(OrtTensor, MakeDescProbmap_NCHW_MultiChannelSmallC) {
    // typical: prob maps sometimes export C=2 or C=4
    const std::vector<int64_t> sh = {1, 2, 64, 128};
    TensorDesc d = idet::internal::make_desc_probmap(sh);

    EXPECT_EQ(d.layout, TensorLayout::NCHW);
    EXPECT_EQ(d.N, 1);
    EXPECT_EQ(d.C, 2);
    EXPECT_EQ(d.H, 64);
    EXPECT_EQ(d.W, 128);
    EXPECT_EQ(d.numel, (std::size_t)1 * 2 * 64 * 128);
}

TEST(OrtTensor, MakeDescProbmap_NHWC_MultiChannelSmallC) {
    const std::vector<int64_t> sh = {1, 64, 128, 2};
    TensorDesc d = idet::internal::make_desc_probmap(sh);

    EXPECT_EQ(d.layout, TensorLayout::NHWC);
    EXPECT_EQ(d.N, 1);
    EXPECT_EQ(d.C, 2);
    EXPECT_EQ(d.H, 64);
    EXPECT_EQ(d.W, 128);
    EXPECT_EQ(d.numel, (std::size_t)1 * 64 * 128 * 2);
}

TEST(OrtTensor, MakeDescProbmap_Rank1_Unknown) {
    const std::vector<int64_t> sh = {123};
    TensorDesc d = idet::internal::make_desc_probmap(sh);

    EXPECT_EQ(d.layout, TensorLayout::Unknown);
    EXPECT_EQ(d.numel, (std::size_t)123);
}

TEST(OrtTensor, MakeDescProbmap_EmptyShape_Unknown) {
    const std::vector<int64_t> sh = {};
    TensorDesc d = idet::internal::make_desc_probmap(sh);

    EXPECT_EQ(d.layout, TensorLayout::Unknown);
    // safe_numel(empty)=1 by implementation
    EXPECT_EQ(d.numel, (std::size_t)1);
}

TEST(OrtTensor, MakeDescProbmap_DynamicDims_DoNotBreakNumel) {
    // ORT uses -1 for dynamic dims. Your safe_numel treats <=0 as 1
    const std::vector<int64_t> sh = {1, 1, -1, -1};
    TensorDesc d = idet::internal::make_desc_probmap(sh);

    // layout can't be reliably detected because H/W are not >8, so may remain Unknown
    EXPECT_EQ(d.numel, (std::size_t)1 * 1 * 1 * 1);
}

TEST(OrtTensor, MakeDescProbmap_AmbiguousRank4_SmallHW_IsSafeAndConsistent) {
    const std::vector<int64_t> sh = {1, 1, 2, 2};
    TensorDesc d = idet::internal::make_desc_probmap(sh);

    EXPECT_EQ(d.numel, (std::size_t)1 * 1 * 2 * 2);

    if (d.layout == TensorLayout::Unknown) return;

    if (d.layout == TensorLayout::NCHW) {
        EXPECT_EQ(d.N, 1);
        EXPECT_EQ(d.C, 1);
        EXPECT_EQ(d.H, 2);
        EXPECT_EQ(d.W, 2);
    } else if (d.layout == TensorLayout::NHWC) {
        EXPECT_EQ(d.N, 1);
        EXPECT_EQ(d.H, 1);
        EXPECT_EQ(d.W, 2);
        EXPECT_EQ(d.C, 2);
    } else {
        FAIL() << "unexpected layout";
    }
}

// ----------------------------------- extract_hw_channel -----------------------------------------

TEST(OrtTensor, ExtractHWChannel_NCHW_Channel0_NoCopy) {
    // [1,1,2,3]
    const std::vector<int64_t> sh = {1, 1, 2, 3};
    TensorDesc d = idet::internal::make_desc_probmap(sh);
    ASSERT_EQ(d.layout, TensorLayout::NCHW);

    // HW=6 values
    std::vector<float> data = {0, 1, 2, 3, 4, 5};
    std::vector<float> scratch;

    const float* p = idet::internal::extract_hw_channel(data.data(), d, 0, scratch);
    ASSERT_NE(p, nullptr);

    EXPECT_TRUE(scratch.empty());
    EXPECT_EQ(p, data.data());

    for (int i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(p[i], (float)i);
}

TEST(OrtTensor, ExtractHWChannel_NCHW_Channel1_UsesOffset) {
    // [1,2,2,3] => C=2, HW=6, total=12
    const std::vector<int64_t> sh = {1, 2, 2, 3};
    TensorDesc d = idet::internal::make_desc_probmap(sh);
    ASSERT_EQ(d.layout, TensorLayout::NCHW);
    ASSERT_EQ(d.C, 2);

    std::vector<float> data(12);
    for (int i = 0; i < 12; ++i)
        data[(std::size_t)i] = (float)i;

    std::vector<float> scratch;
    const float* p1 = idet::internal::extract_hw_channel(data.data(), d, 1, scratch);
    ASSERT_NE(p1, nullptr);
    EXPECT_TRUE(scratch.empty());

    // channel 1 starts at offset HW=6 => values 6..11
    for (int i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(p1[i], (float)(6 + i));
}

TEST(OrtTensor, ExtractHWChannel_NCHW_ChannelClampedBelowZero) {
    TensorDesc d;
    d.layout = TensorLayout::NCHW;
    d.N = 1;
    d.C = 2;
    d.H = 2;
    d.W = 2;

    // NCHW: C0=[0..3], C1=[4..7]
    std::vector<float> data = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> scratch;

    const float* p = idet::internal::extract_hw_channel(data.data(), d, -999, scratch);
    ASSERT_NE(p, nullptr);
    EXPECT_TRUE(scratch.empty());
    EXPECT_EQ(p, data.data()); // clamp -> channel 0

    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(p[i], (float)i);
}

TEST(OrtTensor, ExtractHWChannel_NCHW_ChannelClampedAboveMax) {
    TensorDesc d;
    d.layout = TensorLayout::NCHW;
    d.N = 1;
    d.C = 2;
    d.H = 2;
    d.W = 2;

    std::vector<float> data = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> scratch;

    const float* p = idet::internal::extract_hw_channel(data.data(), d, 999, scratch);
    ASSERT_NE(p, nullptr);
    EXPECT_TRUE(scratch.empty());

    // clamp -> channel 1 => offset 4
    EXPECT_EQ(p, data.data() + 4);

    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(p[i], (float)(4 + i));
}

TEST(OrtTensor, ExtractHWChannel_NHWC_Channel0_C1_MayCopyOrNotButCorrect) {
    TensorDesc d;
    d.layout = TensorLayout::NHWC;
    d.N = 1;
    d.H = 2;
    d.W = 3;
    d.C = 1;
    d.numel = (std::size_t)1 * 2 * 3 * 1;

    std::vector<float> data = {0, 1, 2, 3, 4, 5};
    std::vector<float> scratch;

    const float* p = idet::internal::extract_hw_channel(data.data(), d, 0, scratch);
    ASSERT_NE(p, nullptr);

    if (p == data.data()) {
        EXPECT_TRUE(scratch.empty());
    } else {
        EXPECT_EQ(p, scratch.data());
        EXPECT_EQ(scratch.size(), (std::size_t)6);
    }

    for (int i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(p[i], (float)i);
}

TEST(OrtTensor, ExtractHWChannel_NHWC_Channel1_CopiesCorrectPlane) {
    // [1,2,2,2] => H=2,W=2,C=2
    const std::vector<int64_t> sh = {1, 2, 2, 2};
    TensorDesc d = idet::internal::make_desc_probmap(sh);
    ASSERT_EQ(d.layout, TensorLayout::NHWC);
    ASSERT_EQ(d.C, 2);

    // pixels 0..3 each has [c0,c1]
    std::vector<float> data = {0, 100, 1, 101, 2, 102, 3, 103};
    std::vector<float> scratch;

    const float* p1 = idet::internal::extract_hw_channel(data.data(), d, 1, scratch);
    ASSERT_NE(p1, nullptr);
    ASSERT_EQ(scratch.size(), (std::size_t)4);

    EXPECT_FLOAT_EQ(p1[0], 100);
    EXPECT_FLOAT_EQ(p1[1], 101);
    EXPECT_FLOAT_EQ(p1[2], 102);
    EXPECT_FLOAT_EQ(p1[3], 103);
}

TEST(OrtTensor, ExtractHWChannel_NHWC_ChannelClamped) {
    TensorDesc d;
    d.layout = TensorLayout::NHWC;
    d.N = 1;
    d.C = 2;
    d.H = 1;
    d.W = 3;

    // NHWC, H=1,W=3,C=2:
    // pixel0: [0,10], pixel1:[1,11], pixel2:[2,12]
    std::vector<float> data = {0, 10, 1, 11, 2, 12};
    std::vector<float> scratch;

    // channel 999 -> clamp to 1 => [10,11,12]
    const float* p_hi = idet::internal::extract_hw_channel(data.data(), d, 999, scratch);
    ASSERT_NE(p_hi, nullptr);
    ASSERT_EQ(scratch.size(), (std::size_t)3);

    EXPECT_FLOAT_EQ(p_hi[0], 10);
    EXPECT_FLOAT_EQ(p_hi[1], 11);
    EXPECT_FLOAT_EQ(p_hi[2], 12);

    // channel -5 -> clamp to 0 => [0,1,2]
    scratch.clear();
    const float* p_lo = idet::internal::extract_hw_channel(data.data(), d, -5, scratch);
    ASSERT_NE(p_lo, nullptr);
    ASSERT_EQ(scratch.size(), (std::size_t)3);

    EXPECT_FLOAT_EQ(p_lo[0], 0);
    EXPECT_FLOAT_EQ(p_lo[1], 1);
    EXPECT_FLOAT_EQ(p_lo[2], 2);
}

TEST(OrtTensor, ExtractHWChannel_NHWC_ReusesScratchBuffer) {
    // Ensure scratch is resized appropriately across calls and content updates
    const std::vector<int64_t> sh = {1, 2, 2, 2};
    TensorDesc d = idet::internal::make_desc_probmap(sh);
    ASSERT_EQ(d.layout, TensorLayout::NHWC);

    std::vector<float> data0 = {0, 10, 1, 11, 2, 12, 3, 13};
    std::vector<float> scratch;

    const float* p0 = idet::internal::extract_hw_channel(data0.data(), d, 0, scratch);
    ASSERT_NE(p0, nullptr);
    ASSERT_EQ(scratch.size(), (std::size_t)4);
    EXPECT_FLOAT_EQ(p0[0], 0);
    EXPECT_FLOAT_EQ(p0[1], 1);
    EXPECT_FLOAT_EQ(p0[2], 2);
    EXPECT_FLOAT_EQ(p0[3], 3);

    std::vector<float> data1 = {100, 200, 101, 201, 102, 202, 103, 203};
    const float* p1 = idet::internal::extract_hw_channel(data1.data(), d, 1, scratch);
    ASSERT_NE(p1, nullptr);
    ASSERT_EQ(scratch.size(), (std::size_t)4);
    EXPECT_FLOAT_EQ(p1[0], 200);
    EXPECT_FLOAT_EQ(p1[1], 201);
    EXPECT_FLOAT_EQ(p1[2], 202);
    EXPECT_FLOAT_EQ(p1[3], 203);
}

TEST(OrtTensor, ExtractHWChannel_N1HW_NoCopy) {
    const std::vector<int64_t> sh = {1, 2, 3};
    TensorDesc d = idet::internal::make_desc_probmap(sh);
    ASSERT_EQ(d.layout, TensorLayout::N1HW);

    std::vector<float> data = {0, 1, 2, 3, 4, 5};
    std::vector<float> scratch;

    const float* p = idet::internal::extract_hw_channel(data.data(), d, 0, scratch);
    ASSERT_NE(p, nullptr);

    EXPECT_TRUE(scratch.empty());
    EXPECT_EQ(p, data.data());
}

TEST(OrtTensor, ExtractHWChannel_HW_NoCopy) {
    const std::vector<int64_t> sh = {2, 3};
    TensorDesc d = idet::internal::make_desc_probmap(sh);
    ASSERT_EQ(d.layout, TensorLayout::HW);

    std::vector<float> data = {0, 1, 2, 3, 4, 5};
    std::vector<float> scratch;

    const float* p = idet::internal::extract_hw_channel(data.data(), d, 0, scratch);
    ASSERT_NE(p, nullptr);

    EXPECT_TRUE(scratch.empty());
    EXPECT_EQ(p, data.data());
}

TEST(OrtTensor, ExtractHWChannel_NullData_ReturnsNull) {
    const std::vector<int64_t> sh = {1, 1, 7, 9};
    TensorDesc d = idet::internal::make_desc_probmap(sh);
    ASSERT_EQ(d.layout, TensorLayout::NCHW);

    std::vector<float> scratch;
    const float* p = idet::internal::extract_hw_channel(nullptr, d, 0, scratch);
    EXPECT_EQ(p, nullptr);
    EXPECT_TRUE(scratch.empty());
}

TEST(OrtTensor, ExtractHWChannel_UnknownLayout_ReturnsNull) {
    TensorDesc d;
    d.layout = TensorLayout::Unknown;
    d.H = 7;
    d.W = 9;
    d.C = 1;

    std::vector<float> data(63, 1.0f);
    std::vector<float> scratch;

    const float* p = idet::internal::extract_hw_channel(data.data(), d, 0, scratch);
    EXPECT_EQ(p, nullptr);
}

TEST(OrtTensor, ExtractHWChannel_ZeroHW_ReturnsNull) {
    TensorDesc d;
    d.layout = TensorLayout::NCHW;
    d.H = 0;
    d.W = 9;
    d.C = 1;

    std::vector<float> data(1, 1.0f);
    std::vector<float> scratch;

    const float* p = idet::internal::extract_hw_channel(data.data(), d, 0, scratch);
    EXPECT_EQ(p, nullptr);
}

// ----------------------------------- make_desc_probmap -----------------------------------------

TEST(OrtTensor, MakeDescProbmap_HeuristicPrefersNCHWWhenBothSeemValid) {
    // Ambiguous-ish but should pick NCHW by our heuristic:
    // [1,1,64,2] => NCHW? yes (C=1, H=64, W=2) but W not > 8 => Unknown
    // So build a case where both ends have small channel but still decide:
    // [1,2,64,64] => clearly NCHW
    const std::vector<int64_t> sh = {1, 2, 64, 64};
    TensorDesc d = idet::internal::make_desc_probmap(sh);
    EXPECT_EQ(d.layout, TensorLayout::NCHW);
    EXPECT_EQ(d.C, 2);
    EXPECT_EQ(d.H, 64);
    EXPECT_EQ(d.W, 64);
}
