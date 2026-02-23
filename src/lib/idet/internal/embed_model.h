/**
 * @file embed_model.h
 * @ingroup idet_internal
 * @brief Embedded model blobs (optional) for engines shipped with IDet.
 *
 * This header provides a tiny abstraction (@ref idet::internal::ModelBlob) for accessing model
 * data embedded into the binary at build time.
 *
 * Embedded models are optional and controlled by compile-time macros:
 * - @c IDET_HAVE_DBNET_EMBED enables embedding for the DBNet text detection model
 * - @c IDET_HAVE_SCRFD_EMBED enables embedding for the SCRFD face detection model
 *
 * When the corresponding macro is not defined, the accessor returns an empty blob
 * (@c {nullptr, 0}) and the caller is expected to use an external model path.
 *
 * Typical usage:
 * - Determine engine kind from @ref idet::DetectorConfig::engine
 * - Call @ref idet::internal::get_model_blob
 * - If empty, fall back to loading from @ref idet::DetectorConfig::model_path
 *
 * @note
 * The actual embedded symbol definitions (e.g., @c dbnet_model, @c scrfd_model) are expected to be
 * provided by a generated/compiled translation unit (for example, produced from an ONNX file).
 *
 * @note
 * This is an internal header and is not part of the stable public API.
 */

#pragma once

#include "idet.h"

#include <cstddef>
#include <cstdint>

namespace idet::internal {

/**
 * @brief Describes a contiguous binary model payload stored in memory.
 *
 * This type is a non-owning view over embedded bytes and does not manage lifetime.
 * Embedded arrays typically have static storage duration and remain valid for the
 * lifetime of the process.
 */
struct ModelBlob {
    /** @brief Pointer to the first byte of the model payload (may be null). */
    const void* data = nullptr;

    /** @brief Payload size in bytes (0 means empty). */
    std::size_t size = 0;

    /**
     * @brief Checks whether this blob is empty.
     * @return True if @ref data is null or @ref size is zero, otherwise false.
     */
    [[nodiscard]] constexpr bool empty() const noexcept {
        return data == nullptr || size == 0;
    }
};

// Text detection DBNet model
#if defined(IDET_HAVE_DBNET_EMBED)
/**
 * @brief Embedded DBNet model bytes (generated symbol).
 *
 * The symbol must be defined in exactly one translation unit when embedding is enabled.
 */
extern const unsigned char dbnet_model[];

/** @brief Size of @ref dbnet_model in bytes. */
extern const std::size_t dbnet_model_len;

/**
 * @brief Returns the embedded DBNet model blob.
 * @return Non-empty blob when embedding is enabled, otherwise empty.
 */
inline constexpr ModelBlob get_dbnet_model_blob() noexcept {
    return {dbnet_model, dbnet_model_len};
}
#else
/**
 * @brief Returns an empty blob when DBNet embedding is disabled.
 * @return Empty blob @c {nullptr, 0}.
 */
inline constexpr ModelBlob get_dbnet_model_blob() noexcept {
    return {nullptr, 0};
}
#endif

// Face detection SCRFD model
#if defined(IDET_HAVE_SCRFD_EMBED)
/**
 * @brief Embedded SCRFD model bytes (generated symbol).
 *
 * The symbol must be defined in exactly one translation unit when embedding is enabled.
 */
extern const unsigned char scrfd_model[];

/** @brief Size of @ref scrfd_model in bytes. */
extern const std::size_t scrfd_model_len;

/**
 * @brief Returns the embedded SCRFD model blob.
 * @return Non-empty blob when embedding is enabled, otherwise empty.
 */
inline constexpr ModelBlob get_scrfd_model_blob() noexcept {
    return {scrfd_model, scrfd_model_len};
}
#else
/**
 * @brief Returns an empty blob when SCRFD embedding is disabled.
 * @return Empty blob @c {nullptr, 0}.
 */
inline constexpr ModelBlob get_scrfd_model_blob() noexcept {
    return {nullptr, 0};
}
#endif

/**
 * @brief Returns an embedded model blob for the specified engine kind.
 *
 * Dispatches to engine-specific embedded blob accessors. If embedding is disabled for the
 * requested engine, returns an empty blob.
 *
 * @param engine_kind Engine kind (DBNet, SCRFD, etc.).
 * @return Embedded model blob or an empty blob when unavailable/unsupported.
 */
inline ModelBlob get_model_blob(EngineKind engine_kind) noexcept {
    switch (engine_kind) {
    case EngineKind::DBNet:
        return get_dbnet_model_blob();
    case EngineKind::SCRFD:
        return get_scrfd_model_blob();
    default:
        return {nullptr, 0};
    }
}

} // namespace idet::internal
