#if defined(__has_include) && __has_include(<gtest/gtest.h>)
    #include <gtest/gtest.h>
#elif defined(__has_include) && __has_include(<gtest.h>)
    #include <gtest.h>
#else
    #error "[ERROR] 'gtest.h' header not found"
#endif

#include "algo/tiling.h"
#include "engine/engine.h"
#include "internal/bgr_image.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

struct TestBgr {
    std::vector<std::uint8_t> data;
    int width = 0;
    int height = 0;

    [[nodiscard]] idet::internal::BgrImageView view() const noexcept {
        return {data.data(), width, height, static_cast<std::ptrdiff_t>(width * 3)};
    }
};

static TestBgr make_bgr(int width, int height) {
    TestBgr out;
    out.width = width;
    out.height = height;
    out.data.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U, 0);
    return out;
}

// Use this helper to avoid accidental swaps forever
static inline idet::GridSpec grid(int cols, int rows) {
    idet::GridSpec g{};
    g.cols = cols;
    g.rows = rows;
    return g;
}

static inline bool rect_inside(const idet::internal::RectI& r, int W, int H) {
    if (W <= 0 || H <= 0) return false;
    if (r.width <= 0 || r.height <= 0) return false;
    if (r.x < 0 || r.y < 0) return false;
    if (r.x + r.width > W) return false;
    if (r.y + r.height > H) return false;
    return true;
}

// Discrete coverage checks are robust and catch subtle bugs. Keep them for SMALL images only
static void expect_full_cover_no_overlap_discrete(const std::vector<idet::internal::RectI>& tiles, int W, int H) {
    ASSERT_GT(W, 0);
    ASSERT_GT(H, 0);
    ASSERT_LE((std::size_t)W * (std::size_t)H, (std::size_t)400 * 400) << "too big for discrete cover check";

    std::vector<int> cover((std::size_t)W * (std::size_t)H, 0);

    for (const auto& t : tiles) {
        ASSERT_TRUE(rect_inside(t, W, H));
        for (int y = t.y; y < t.y + t.height; ++y) {
            for (int x = t.x; x < t.x + t.width; ++x) {
                cover[(std::size_t)y * (std::size_t)W + (std::size_t)x] += 1;
            }
        }
    }

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int v = cover[(std::size_t)y * (std::size_t)W + (std::size_t)x];
            EXPECT_EQ(v, 1) << "pixel(" << x << "," << y << ") covered " << v << " times";
        }
    }
}

static void expect_full_cover_discrete(const std::vector<idet::internal::RectI>& tiles, int W, int H) {
    ASSERT_GT(W, 0);
    ASSERT_GT(H, 0);
    ASSERT_LE((std::size_t)W * (std::size_t)H, (std::size_t)400 * 400) << "too big for discrete cover check";

    std::vector<int> cover((std::size_t)W * (std::size_t)H, 0);

    for (const auto& t : tiles) {
        ASSERT_TRUE(rect_inside(t, W, H));
        for (int y = t.y; y < t.y + t.height; ++y) {
            for (int x = t.x; x < t.x + t.width; ++x) {
                cover[(std::size_t)y * (std::size_t)W + (std::size_t)x] += 1;
            }
        }
    }

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int v = cover[(std::size_t)y * (std::size_t)W + (std::size_t)x];
            EXPECT_GE(v, 1) << "pixel(" << x << "," << y << ") uncovered";
        }
    }
}

// --------------------------- Helpers for infer_tiled() verification ---------------------------

static inline bool pt_eq(const idet::Point2f& a, const idet::Point2f& b, float eps = 1e-6f) {
    return std::fabs(a.x - b.x) <= eps && std::fabs(a.y - b.y) <= eps;
}

static inline void expect_det_tl(const idet::algo::Detection& d, float x, float y) {
    EXPECT_TRUE(pt_eq(d.pts[0], idet::Point2f{x, y}))
        << "got TL=(" << d.pts[0].x << "," << d.pts[0].y << ") expected (" << x << "," << y << ")";
}

class DummyEngine final : public idet::engine::IEngine {
  public:
    explicit DummyEngine(const idet::DetectorConfig& cfg) : IEngine(cfg, "dummy") {}

    idet::EngineKind kind() const noexcept override {
        return cfg_.engine;
    }
    idet::Task task() const noexcept override {
        return cfg_.task;
    }

    idet::Status update_hot(const idet::DetectorConfig&) noexcept override {
        return idet::Status::Ok();
    }

    idet::Status setup_binding(int w, int h, int contexts) noexcept override {
        binding_ready_ = true;
        bound_w_ = w;
        bound_h_ = h;
        contexts_ = (contexts > 0) ? contexts : 1;
        used_ctx_mask.store(0, std::memory_order_relaxed);
        calls_unbound.store(0, std::memory_order_relaxed);
        calls_bound.store(0, std::memory_order_relaxed);
        return idet::Status::Ok();
    }

    void unset_binding() noexcept override {
        binding_ready_ = false;
        bound_w_ = bound_h_ = 0;
        contexts_ = 0;
        used_ctx_mask.store(0, std::memory_order_relaxed);
        calls_unbound.store(0, std::memory_order_relaxed);
        calls_bound.store(0, std::memory_order_relaxed);
    }

    idet::Result<std::vector<idet::algo::Detection>>
    infer_unbound(const idet::internal::BgrImageView& bgr) noexcept override {
        calls_unbound.fetch_add(1, std::memory_order_relaxed);
        return idet::Result<std::vector<idet::algo::Detection>>::Ok(make_one_det(bgr.width, bgr.height, 0.5f));
    }

    idet::Result<std::vector<idet::algo::Detection>> infer_bound(const idet::internal::BgrImageView& bgr,
                                                                 int ctx_idx) noexcept override {
        calls_bound.fetch_add(1, std::memory_order_relaxed);

        if (ctx_idx >= 0 && ctx_idx < 64) {
            const std::uint64_t bit = (1ull << (unsigned)ctx_idx);
            used_ctx_mask.fetch_or(bit, std::memory_order_relaxed);
        }

        return idet::Result<std::vector<idet::algo::Detection>>::Ok(make_one_det(bgr.width, bgr.height, 0.6f));
    }

    std::atomic<std::uint64_t> used_ctx_mask{0};
    std::atomic<int> calls_unbound{0};
    std::atomic<int> calls_bound{0};

  private:
    static std::vector<idet::algo::Detection> make_one_det(int w, int h, float score) {
        idet::algo::Detection d;
        d.score = score;
        d.pts[0] = {0.f, 0.f};
        d.pts[1] = {float(w), 0.f};
        d.pts[2] = {float(w), float(h)};
        d.pts[3] = {0.f, float(h)};
        return {d};
    }
};

} // namespace

// --------------------------- make_tiles ---------------------------

TEST(Tiling, MakeTiles_InvalidInput_ReturnsEmpty) {
    EXPECT_TRUE(idet::algo::make_tiles(0, 10, grid(2, 2), 0.0f).empty());
    EXPECT_TRUE(idet::algo::make_tiles(10, 0, grid(2, 2), 0.0f).empty());
    EXPECT_TRUE(idet::algo::make_tiles(-1, 10, grid(2, 2), 0.0f).empty());
    EXPECT_TRUE(idet::algo::make_tiles(10, -1, grid(2, 2), 0.0f).empty());
}

TEST(Tiling, MakeTiles_InvalidGrid_ReturnsEmpty) {
    EXPECT_TRUE(idet::algo::make_tiles(10, 10, grid(0, 2), 0.0f).empty());
    EXPECT_TRUE(idet::algo::make_tiles(10, 10, grid(2, 0), 0.0f).empty());
    EXPECT_TRUE(idet::algo::make_tiles(10, 10, grid(-1, -1), 0.0f).empty());
}

TEST(Tiling, MakeTiles_NoOverlap_2x2_ExactRects_AndPartition) {
    const int W = 100, H = 50;
    auto tiles = idet::algo::make_tiles(W, H, grid(2, 2), 0.0f);
    ASSERT_EQ(tiles.size(), 4u);

    EXPECT_EQ(tiles[0], (idet::internal::RectI{0, 0, 50, 25}));
    EXPECT_EQ(tiles[1], (idet::internal::RectI{50, 0, 50, 25}));
    EXPECT_EQ(tiles[2], (idet::internal::RectI{0, 25, 50, 25}));
    EXPECT_EQ(tiles[3], (idet::internal::RectI{50, 25, 50, 25}));

    expect_full_cover_no_overlap_discrete(tiles, W, H);
}

TEST(Tiling, MakeTiles_NoOverlap_1x1_FullImage) {
    const int W = 77, H = 33;
    auto tiles = idet::algo::make_tiles(W, H, grid(1, 1), 0.0f);
    ASSERT_EQ(tiles.size(), 1u);
    EXPECT_EQ(tiles[0], (idet::internal::RectI{0, 0, W, H}));
}

TEST(Tiling, MakeTiles_NoOverlap_NonDivisibleDims_PartitionsExactly) {
    // Split base+rem must still cover exactly once
    const int W = 101, H = 51;
    auto tiles = idet::algo::make_tiles(W, H, grid(2, 2), 0.0f);
    ASSERT_EQ(tiles.size(), 4u);

    for (const auto& t : tiles)
        EXPECT_TRUE(rect_inside(t, W, H));
    expect_full_cover_no_overlap_discrete(tiles, W, H);
}

TEST(Tiling, MakeTiles_WithOverlap_3x1_ExactRects_ForDivisibleCase) {
    const int W = 300, H = 100;
    auto tiles = idet::algo::make_tiles(W, H, grid(3, 1), 0.2f);
    ASSERT_EQ(tiles.size(), 3u);

    EXPECT_EQ(tiles[0], (idet::internal::RectI{0, 0, 120, 100}));
    EXPECT_EQ(tiles[1], (idet::internal::RectI{80, 0, 140, 100}));
    EXPECT_EQ(tiles[2], (idet::internal::RectI{180, 0, 120, 100}));

    expect_full_cover_discrete(tiles, W, H);
}

TEST(Tiling, MakeTiles_OverlapClamped_AlwaysValidAndCovers) {
    const int W = 120, H = 80;
    const auto g = grid(3, 2);

    for (float overlap : {-10.0f, -1.0f, 0.0f, 0.25f, 0.95f, 1.0f, 2.0f, 10.0f}) {
        auto tiles = idet::algo::make_tiles(W, H, g, overlap);
        ASSERT_EQ(tiles.size(), (std::size_t)g.cols * (std::size_t)g.rows);

        for (const auto& t : tiles) {
            EXPECT_TRUE(rect_inside(t, W, H))
                << "overlap=" << overlap << " rect=(" << t.x << "," << t.y << "," << t.width << "," << t.height << ")";
        }
        expect_full_cover_discrete(tiles, W, H);
    }
}

// --------------------------- infer_tiled ---------------------------

TEST(Tiling, InferTiled_EmptyOrWrongType_ReturnsErr) {
    idet::DetectorConfig cfg{};
    cfg.task = idet::Task::Text;
    cfg.engine = idet::EngineKind::DBNet;
    DummyEngine eng(cfg);

    auto r0 = idet::algo::infer_tiled(eng, idet::internal::BgrImageView{}, false, 0, false, grid(2, 1), 0.0f, 1);
    EXPECT_FALSE(r0.ok());

    std::vector<std::uint8_t> wrong(100, 0);
    idet::internal::BgrImageView wrong_view{wrong.data(), 10, 10, 10};
    auto r1 = idet::algo::infer_tiled(eng, wrong_view, false, 0, false, grid(2, 1), 0.0f, 1);
    EXPECT_FALSE(r1.ok());
}

TEST(Tiling, InferTiled_Unbound_OffsetsApplied_2x1_SerialDeterministicOrder) {
    idet::DetectorConfig cfg{};
    cfg.task = idet::Task::Text;
    cfg.engine = idet::EngineKind::DBNet;
    DummyEngine eng(cfg);

    auto img = make_bgr(100, 50);

    auto r = idet::algo::infer_tiled(eng, img.view(),
                                     /*bound=*/false, /*ctx_idx=*/0, /*parallel_bound=*/false, grid(2, 1),
                                     /*overlap=*/0.0f, /*tile_omp_threads=*/1);
    ASSERT_TRUE(r.ok());
    const auto dets = r.value();
    ASSERT_EQ(dets.size(), 2u);

    // For n_threads=1, order follows rects order: left tile then right tile
    expect_det_tl(dets[0], 0.f, 0.f);
    expect_det_tl(dets[1], 50.f, 0.f);

    EXPECT_EQ(eng.calls_unbound.load(), 2);
    EXPECT_EQ(eng.calls_bound.load(), 0);
}

TEST(Tiling, InferTiled_Unbound_OffsetsApplied_1x2_YOffsets_SerialDeterministicOrder) {
    idet::DetectorConfig cfg{};
    cfg.task = idet::Task::Text;
    cfg.engine = idet::EngineKind::DBNet;
    DummyEngine eng(cfg);

    auto img = make_bgr(50, 100);

    auto r = idet::algo::infer_tiled(eng, img.view(),
                                     /*bound=*/false, 0, false, grid(1, 2), /*overlap=*/0.0f, /*tile_omp_threads=*/1);
    ASSERT_TRUE(r.ok());
    const auto dets = r.value();
    ASSERT_EQ(dets.size(), 2u);

    // top tile then bottom tile
    expect_det_tl(dets[0], 0.f, 0.f);
    expect_det_tl(dets[1], 0.f, 50.f);

    EXPECT_EQ(eng.calls_unbound.load(), 2);
}

TEST(Tiling, InferTiled_Unbound_2x2_OffsetsBothAxes_SerialDeterministicOrder) {
    idet::DetectorConfig cfg{};
    cfg.task = idet::Task::Text;
    cfg.engine = idet::EngineKind::DBNet;
    DummyEngine eng(cfg);

    auto img = make_bgr(80, 60);

    auto r = idet::algo::infer_tiled(eng, img.view(),
                                     /*bound=*/false, 0, false, grid(2, 2), /*overlap=*/0.0f, /*tile_omp_threads=*/1);
    ASSERT_TRUE(r.ok());
    const auto dets = r.value();
    ASSERT_EQ(dets.size(), 4u);

    expect_det_tl(dets[0], 0.f, 0.f);
    expect_det_tl(dets[1], 40.f, 0.f);
    expect_det_tl(dets[2], 0.f, 30.f);
    expect_det_tl(dets[3], 40.f, 30.f);
}

TEST(Tiling, InferTiled_OverlapStillOffsetsMatchTileOrigins) {
    idet::DetectorConfig cfg{};
    cfg.task = idet::Task::Text;
    cfg.engine = idet::EngineKind::DBNet;
    DummyEngine eng(cfg);

    auto img = make_bgr(100, 50);
    const auto g = grid(2, 1);
    const float overlap = 0.25f;

    // Use make_tiles() to define the contract
    const auto tiles = idet::algo::make_tiles(img.width, img.height, g, overlap);
    ASSERT_EQ(tiles.size(), 2u);

    auto r = idet::algo::infer_tiled(eng, img.view(),
                                     /*bound=*/false, 0, false, g, overlap, /*tile_omp_threads=*/1);
    ASSERT_TRUE(r.ok());
    const auto dets = r.value();
    ASSERT_EQ(dets.size(), 2u);

    expect_det_tl(dets[0], (float)tiles[0].x, (float)tiles[0].y);
    expect_det_tl(dets[1], (float)tiles[1].x, (float)tiles[1].y);
}

TEST(Tiling, InferTiled_Bound_WithoutBinding_ReturnsErr) {
    idet::DetectorConfig cfg{};
    cfg.task = idet::Task::Text;
    cfg.engine = idet::EngineKind::DBNet;
    DummyEngine eng(cfg);

    auto img = make_bgr(100, 50);
    auto r = idet::algo::infer_tiled(eng, img.view(),
                                     /*bound=*/true, /*ctx_idx=*/0, /*parallel_bound=*/false, grid(2, 1), 0.0f, 1);
    EXPECT_FALSE(r.ok());
}

TEST(Tiling, InferTiled_Bound_Serial_UsesGivenCtx_AndValidatesRange) {
    idet::DetectorConfig cfg{};
    cfg.task = idet::Task::Text;
    cfg.engine = idet::EngineKind::DBNet;
    DummyEngine eng(cfg);
    ASSERT_TRUE(eng.setup_binding(64, 64, 4).ok());

    auto img = make_bgr(100, 50);

    // good ctx
    {
        auto r = idet::algo::infer_tiled(eng, img.view(),
                                         /*bound=*/true, /*ctx_idx=*/2, /*parallel_bound=*/false, grid(2, 1), 0.0f, 8);
        ASSERT_TRUE(r.ok());
        EXPECT_EQ(eng.calls_bound.load(), 2);

        const std::uint64_t mask = eng.used_ctx_mask.load();
        EXPECT_NE(mask & (1ull << 2), 0ull);
        // serial safe mode -> should not touch other ctx bits
        EXPECT_EQ(mask, (1ull << 2));
    }

    // out of range ctx should error
    {
        auto r = idet::algo::infer_tiled(eng, img.view(),
                                         /*bound=*/true, /*ctx_idx=*/99, /*parallel_bound=*/false, grid(2, 1), 0.0f, 8);
        EXPECT_FALSE(r.ok());
    }
}

TEST(Tiling, InferTiled_Bound_Parallel_UsesOnlyValidCtxIds) {
    idet::DetectorConfig cfg{};
    cfg.task = idet::Task::Text;
    cfg.engine = idet::EngineKind::DBNet;
    DummyEngine eng(cfg);
    ASSERT_TRUE(eng.setup_binding(64, 64, 4).ok());

    auto img = make_bgr(128, 64);
    const auto g = grid(4, 1); // 4 tiles

    auto r = idet::algo::infer_tiled(eng, img.view(),
                                     /*bound=*/true, /*ctx_idx=*/0, /*parallel_bound=*/true, g, 0.0f,
                                     /*tile_omp_threads=*/8);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(eng.calls_bound.load(), 4);

    const std::uint64_t mask = eng.used_ctx_mask.load();
    // Must not use ctx >= contexts (4)
    EXPECT_EQ(mask & ~((1ull << 4) - 1ull), 0ull);

    // At least one ctx used
    EXPECT_NE(mask, 0ull);
}

TEST(Tiling, InferTiled_Unbound_BasicRun_Succeeds) {
    idet::DetectorConfig cfg{};
    cfg.task = idet::Task::Text;
    cfg.engine = idet::EngineKind::DBNet;
    DummyEngine eng(cfg);

    auto img = make_bgr(64, 32);
    auto r = idet::algo::infer_tiled(eng, img.view(), /*bound=*/false, 0, false, grid(4, 1), 0.0f, 1);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(eng.calls_unbound.load(), 4);
}
