/**
 * @file ort_headers.h
 * @ingroup idet_internal
 * @brief Centralized include shim for ONNX Runtime (ORT) headers used by IDet internals.
 *
 * This header provides a single stable include point for ONNX Runtime within the IDet codebase.
 * It supports multiple common header layouts:
 * - source/build-tree style includes: @c <onnxruntime/core/session/onnxruntime_cxx_api.h>
 * - packaged installs exposing the umbrella header directly: @c <onnxruntime_cxx_api.h>
 *
 * Optional execution providers (e.g., Arm Compute Library / ACL) can be enabled via compile-time
 * toggles, allowing the implementation to include provider factory headers when available.
 *
 * Rationale:
 * - avoids scattering conditional include logic throughout the project,
 * - keeps include paths consistent across platforms and packaging systems,
 * - enables optional providers without hard-coding a single ORT distribution layout.
 *
 * @note
 * This is an internal header and is not part of the stable public API. Prefer including this shim
 * instead of including ORT headers directly within IDet sources.
 */

#pragma once

/**
 * @def USE_ACL
 * @brief Build-time toggle for the Arm Compute Library (ACL) execution provider.
 *
 * Set this macro to @c 1 to enable including the ACL provider factory header and allow the
 * implementation to register the ACL execution provider with an ORT session.
 *
 * This is a compile-time knob. ACL-specific code paths must be guarded behind this macro.
 *
 * @warning
 * Enabling this macro requires an ORT build/package that provides ACL provider headers.
 * If the headers are not present, compilation will fail.
 *
 * @note
 * Default value is @c 0.
 */
#ifndef USE_ACL
    #define USE_ACL 0
#endif

// -----------------------------------------------------------------------------
// onnxruntime_cxx_api.h
// -----------------------------------------------------------------------------
#if defined(__has_include) && __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
    #include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#elif defined(__has_include) && __has_include(<onnxruntime/onnxruntime_cxx_api.h>)
    #include <onnxruntime/onnxruntime_cxx_api.h>
#elif defined(__has_include) && __has_include(<onnxruntime_cxx_api.h>)
    #include <onnxruntime_cxx_api.h>
#else
    #error "[IDet] ONNX Runtime header not found: onnxruntime_cxx_api.h"
#endif

// -----------------------------------------------------------------------------
// acl_provider_factory.h
// -----------------------------------------------------------------------------
#if USE_ACL
    #if defined(__has_include) && __has_include(<onnxruntime/core/providers/acl/acl_provider_factory.h>)
        #include <onnxruntime/core/providers/acl/acl_provider_factory.h>
    #elif defined(__has_include) && __has_include(<acl_provider_factory.h>)
        #include <acl_provider_factory.h>
    #else
        #error "[IDet] ONNX Runtime header not found: acl_provider_factory.h (ACL provider enabled)"
    #endif
#endif
