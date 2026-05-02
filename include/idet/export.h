/**
 * @file export.h
 * @brief Public symbol export/import macro for the IDet library (shared/static builds).
 *
 * This header defines the @c IDET_API macro controlling symbol visibility for public API
 * entities (classes, functions, global variables) depending on how the library is built
 * and consumed:
 *
 * - **Static build**: no special attributes are required, so @c IDET_API is empty.
 * - **Shared build (Windows)**: uses @c __declspec(dllexport/dllimport) for DLLs.
 * - **Shared build (ELF: Linux/macOS)**: uses @c __attribute__((visibility("default")))
 *   with GCC/Clang to export symbols.
 *
 * Typical usage:
 * @code
 * class IDET_API Detector { ... };
 * IDET_API idet::Status create_detector(...);
 * @endcode
 *
 * @note
 * On ELF platforms, this is commonly combined with compiler/linker settings such as
 * @c -fvisibility=hidden to hide all symbols by default and export only those explicitly
 * marked with @c IDET_API. This reduces the exported ABI surface and can improve load times.
 *
 * @warning
 * Inconsistent combinations of IDET_BUILD_SHARED / IDET_USE_SHARED may cause missing exports
 * (consumer link errors), or incorrect imports on Windows. Ensure your build system defines
 * these macros consistently for each target.
 */

#pragma once

/**
 * @name Build Selection Macros
 * @brief Optional build-system macros that select how @ref IDET_API expands.
 *
 * These symbols are normally provided by Meson or by a downstream consumer:
 * - @c IDET_BUILD_STATIC: static build/usage, no import/export attribute.
 * - @c IDET_BUILD_SHARED: compiling the shared library itself, export public symbols.
 * - @c IDET_USE_SHARED: consuming the shared library, import public symbols where required.
 *
 * @{
 */
/** @} */

/**
 * @def IDET_API
 * @brief Visibility/export attribute for public IDet API symbols.
 *
 * Expands to a platform-specific attribute:
 * - static build → empty,
 * - Windows shared build → @c __declspec(dllexport),
 * - Windows shared use   → @c __declspec(dllimport),
 * - GCC/Clang shared build on ELF → @c __attribute__((visibility("default"))),
 * - otherwise → empty.
 *
 * @note
 * On Windows, exporting/importing classes is often required to ensure vtables/RTTI symbols
 * are available to consumers.
 */

#if defined(IDET_BUILD_STATIC)
    #define IDET_API
#else
    #if defined(_WIN32) || defined(__CYGWIN__)
        #if defined(IDET_BUILD_SHARED)
            #define IDET_API __declspec(dllexport)
        #elif defined(IDET_USE_SHARED)
            #define IDET_API __declspec(dllimport)
        #else
            #define IDET_API
        #endif
    #else
        #if defined(IDET_BUILD_SHARED) && (defined(__GNUC__) || defined(__clang__))
            #define IDET_API __attribute__((visibility("default")))
        #else
            #define IDET_API
        #endif
    #endif
#endif
