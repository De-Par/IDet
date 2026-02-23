/**
 * @file ort_tensor.h
 * @ingroup idet_internal
 * @brief Small production-grade helpers for ORT tensor shape/layout handling.
 *
 * Motivation:
 * - Models are exported with different tensor layouts (NCHW/NHWC/flat).
 * - Binding must use the real output shape (never force {1,1,H,W}).
 * - Decoding must be layout-aware and safe (no UB, no wrong indexing assumptions).
 *
 * Supported "probmap-like" output shapes:
 * - [N, C, H, W]  (NCHW)
 * - [N, H, W, C]  (NHWC)
 * - [N, H, W]     (treated as single-channel)
 * - [H, W]        (treated as single-channel, batch=1)
 *
 * Ambiguous rank-4 cases:
 * - For shapes that can be interpreted as both NCHW and NHWC (e.g., [1,1,H,W] vs [1,H,W,1]),
 *   the heuristic prefers the interpretation with larger spatial area (H*W).
 *   If the areas tie (fully ambiguous), the policy chooses NHWC.
 *
 * @note
 * This is an internal header and is not part of the stable public API.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace idet::internal {

/**
 * @brief Logical layout classification for a tensor shape.
 *
 * The layouts here describe how a contiguous float buffer should be interpreted.
 * They are intentionally minimal and tailored for IDet decoding needs.
 */
enum class TensorLayout : std::uint8_t {
    /** Unknown/unsupported layout. */
    Unknown = 0,

    /** [N, C, H, W] */
    NCHW,

    /** [N, H, W, C] */
    NHWC,

    /** [N, H, W] (probmap-like, implied C=1) */
    N1HW,

    /**
     * @brief Flat "locations x channels" style export.
     *
     * Commonly used by face detectors (e.g., SCRFD), where the shape may resemble:
     * - [N, Nloc, C] or
     * - [N, Nloc, 4] for bounding box regression
     *
     * Interpretation is model-specific; this enum value is only a classification hint.
     */
    FlatNC,

    /** [H, W] (probmap-like, implied N=1, C=1) */
    HW,
};

/**
 * @brief Parsed tensor description used by decoding helpers.
 *
 * Fields @ref N, @ref C, @ref H, @ref W represent a normalized view of the shape:
 * - For NCHW/NHWC: N,C,H,W are extracted from shape.
 * - For N1HW/HW: C is forced to 1 and N is forced to 1 for HW.
 *
 * @note
 * For dynamic/unknown dimensions (<= 0), helpers may substitute 1 to allow safe arithmetic.
 * This is a decoding convenience, not a guarantee about the true runtime dimension.
 */
struct TensorDesc {
    /** @brief Original tensor shape as reported by ORT (may contain -1 for dynamic dims). */
    std::vector<int64_t> shape;

    /** @brief Detected/assumed layout classification. */
    TensorLayout layout = TensorLayout::Unknown;

    /** @brief Batch dimension (normalized; may be 1 for unknown). */
    int64_t N = 1;

    /** @brief Channel dimension (normalized; may be 1 for unknown). */
    int64_t C = 1;

    /** @brief Height dimension (normalized; 0 if unknown/unsupported). */
    int64_t H = 0;

    /** @brief Width dimension (normalized; 0 if unknown/unsupported). */
    int64_t W = 0;

    /**
     * @brief Product of dimensions with "safe" substitution for dynamic values.
     *
     * Computed using @ref safe_numel(), where non-positive dimensions are treated as 1.
     * This is useful for bounds checks on contiguous buffers, but should not be interpreted
     * as the true runtime element count when shape contains dynamic values.
     */
    std::size_t numel = 0;
};

/**
 * @brief Substitute a non-positive dimension with 1 for safe arithmetic.
 *
 * ORT commonly reports dynamic dimensions as -1. Treating them as 1 allows computing
 * conservative products without underflow/UB.
 *
 * @param v Dimension value.
 * @return @p v if @p v > 0, otherwise 1.
 */
static inline int64_t safe_dim(int64_t v) noexcept {
    return (v > 0) ? v : 1;
}

/**
 * @brief Compute a "safe" element count for a shape vector.
 *
 * Each dimension is first passed through @ref safe_dim(), therefore dynamic/unknown
 * dimensions (<= 0) are treated as 1.
 *
 * @param sh Shape vector.
 * @return Product of dimensions with safe substitution.
 */
static inline std::size_t safe_numel(const std::vector<int64_t>& sh) noexcept {
    std::size_t n = 1;
    for (auto v : sh) {
        n *= static_cast<std::size_t>(safe_dim(v));
    }
    return n;
}

/**
 * @brief Heuristic: does this dimension look like a small channel count?
 *
 * Probmap-like outputs typically have a small number of channels (often 1 or 2; sometimes 4/6/8).
 * This helper is used to disambiguate rank-4 shapes between NCHW and NHWC.
 *
 * @param x Candidate channel dimension.
 * @return True if the value looks like a plausible channel count.
 */
static inline bool looks_small_channel(int64_t x) noexcept {
    // Score/prob channels usually 1/2, sometimes 4/6/8; keep the heuristic generous.
    return x > 0 && x <= 16;
}

/**
 * @brief Compute spatial area H*W safely.
 *
 * @param h Height.
 * @param w Width.
 * @return H*W if both dimensions are positive, otherwise 0.
 */
static inline std::size_t safe_area(int64_t h, int64_t w) noexcept {
    if (h <= 0 || w <= 0) return 0;
    return static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
}

/**
 * @brief Build a @ref TensorDesc for "probmap-like" outputs.
 *
 * The function classifies supported shapes and fills a normalized (N,C,H,W) view for decoding.
 *
 * Supported ranks:
 * - rank 4: attempts NCHW/NHWC disambiguation using channel heuristic and spatial-area policy
 * - rank 3: treated as [N,H,W] with C=1
 * - rank 2: treated as [H,W] with N=1, C=1
 *
 * @param sh Shape vector (as reported by ORT).
 * @return Parsed descriptor. If unsupported/unknown, @ref TensorDesc::layout remains Unknown.
 *
 * @note
 * For shapes with dynamic/unknown dims (<= 0), dimensions are normalized using @ref safe_dim().
 * This allows decoding helpers to operate safely, but does not guarantee correct results unless
 * runtime shapes are concrete and consistent with the chosen layout.
 */
static inline TensorDesc make_desc_probmap(const std::vector<int64_t>& sh) noexcept {
    TensorDesc d;
    d.shape = sh;
    d.numel = safe_numel(sh);

    const int r = static_cast<int>(sh.size());
    if (r == 4) {
        const int64_t N0 = safe_dim(sh[0]);

        // Candidate A: NCHW = [N,C,H,W]
        const int64_t Cn = safe_dim(sh[1]);
        const int64_t Hn = safe_dim(sh[2]);
        const int64_t Wn = safe_dim(sh[3]);
        const bool cand_nchw = looks_small_channel(Cn);

        // Candidate B: NHWC = [N,H,W,C]
        const int64_t Hh = safe_dim(sh[1]);
        const int64_t Wh = safe_dim(sh[2]);
        const int64_t Ch = safe_dim(sh[3]);
        const bool cand_nhwc = looks_small_channel(Ch);

        if (cand_nchw && !cand_nhwc) {
            d.layout = TensorLayout::NCHW;
            d.N = N0;
            d.C = Cn;
            d.H = Hn;
            d.W = Wn;
            return d;
        }
        if (cand_nhwc && !cand_nchw) {
            d.layout = TensorLayout::NHWC;
            d.N = N0;
            d.C = Ch;
            d.H = Hh;
            d.W = Wh;
            return d;
        }
        if (cand_nchw && cand_nhwc) {
            // Ambiguous (e.g., [1,1,H,W] vs [1,H,W,1]).
            // Choose the interpretation with larger spatial area (H*W).
            const std::size_t area_nchw = safe_area(Hn, Wn);
            const std::size_t area_nhwc = safe_area(Hh, Wh);

            if (area_nhwc > area_nchw) {
                d.layout = TensorLayout::NHWC;
                d.N = N0;
                d.C = Ch;
                d.H = Hh;
                d.W = Wh;
            } else if (area_nchw > area_nhwc) {
                d.layout = TensorLayout::NCHW;
                d.N = N0;
                d.C = Cn;
                d.H = Hn;
                d.W = Wn;
            } else {
                // Fully ambiguous (e.g., [1,2,2,2]). Choose NHWC by policy.
                d.layout = TensorLayout::NHWC;
                d.N = N0;
                d.C = Ch;
                d.H = Hh;
                d.W = Wh;
            }
            return d;
        }

        // Unknown rank-4 layout (not probmap-like or cannot be classified safely).
        return d;
    }

    if (r == 3) {
        // [N,H,W]
        d.layout = TensorLayout::N1HW;
        d.N = safe_dim(sh[0]);
        d.C = 1;
        d.H = safe_dim(sh[1]);
        d.W = safe_dim(sh[2]);
        return d;
    }

    if (r == 2) {
        // [H,W]
        d.layout = TensorLayout::HW;
        d.N = 1;
        d.C = 1;
        d.H = safe_dim(sh[0]);
        d.W = safe_dim(sh[1]);
        return d;
    }

    return d;
}

/**
 * @brief Extract a contiguous HxW float plane for a given channel.
 *
 * This helper returns a pointer to a contiguous plane of size @c H*W floats that represents
 * the requested channel for batch 0.
 *
 * Behavior by layout:
 * - @ref TensorLayout::NCHW:
 *   returns a pointer into the original buffer at the beginning of the channel plane (batch 0).
 * - @ref TensorLayout::NHWC:
 *   gathers the channel plane into @p scratch and returns @c scratch.data().
 * - @ref TensorLayout::N1HW and @ref TensorLayout::HW:
 *   returns the original buffer pointer (single-channel).
 *
 * Preconditions:
 * - @p data points to a contiguous float buffer containing at least the elements described by @p desc.
 * - @p desc must describe a supported probmap-like layout with valid positive H/W.
 * - For NCHW/NHWC, this function decodes batch 0 only (N>1 is ignored for indexing).
 *
 * @param data Pointer to contiguous tensor data (float).
 * @param desc Parsed tensor descriptor.
 * @param channel Requested channel index. Clamped to [0, C-1] when C>0.
 * @param scratch Temporary storage used only for NHWC gathering. Contents are overwritten.
 * @return Pointer to contiguous HxW float plane, or null if @p desc is invalid/unsupported.
 *
 * @note
 * The returned pointer remains valid as long as:
 * - for NCHW/N1HW/HW: the caller keeps @p data alive,
 * - for NHWC: the caller keeps @p scratch alive and does not resize it before use.
 */
static inline const float* extract_hw_channel(const float* data, const TensorDesc& desc, int channel,
                                              std::vector<float>& scratch) {
    if (!data) return nullptr;
    if (desc.H <= 0 || desc.W <= 0) return nullptr;

    const std::size_t hw = static_cast<std::size_t>(desc.H) * static_cast<std::size_t>(desc.W);

    channel = std::max(0, channel);
    if (desc.C > 0) channel = std::min<int>(channel, static_cast<int>(desc.C - 1));

    switch (desc.layout) {
    case TensorLayout::NCHW: {
        // [N,C,H,W] — return pointer into channel plane (batch 0).
        return data + static_cast<std::size_t>(channel) * hw;
    }
    case TensorLayout::NHWC: {
        // [N,H,W,C] — gather plane into scratch (batch 0).
        const std::size_t C = static_cast<std::size_t>(std::max<int64_t>(1, desc.C));
        scratch.resize(hw);
        for (std::size_t i = 0; i < hw; ++i) {
            scratch[i] = data[i * C + static_cast<std::size_t>(channel)];
        }
        return scratch.data();
    }
    case TensorLayout::N1HW:
    case TensorLayout::HW:
        // Single-channel layouts: the buffer already represents an HxW plane.
        return data;
    default:
        return nullptr;
    }
}

} // namespace idet::internal
