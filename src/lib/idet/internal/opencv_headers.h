/**
 * @file opencv_headers.h
 * @ingroup idet_internal
 * @brief Centralized include shim for OpenCV headers used by IDet internals.
 *
 * This header provides a single stable include point for OpenCV within the IDet codebase.
 * It supports both common installation layouts:
 * - system-wide OpenCV installations that expose headers under @c <opencv2/...>
 * - distributions that install under an @c opencv4/ prefix: @c <opencv4/opencv2/...>
 *
 * If OpenCV headers cannot be found, this header triggers a compile-time error with a clear
 * diagnostic message.
 *
 * Rationale:
 * - avoids scattering conditional include logic throughout the project,
 * - keeps include paths consistent across platforms and packaging systems,
 * - reduces build friction when OpenCV is provided by different distributions.
 *
 * @note
 * This is an internal header and is not part of the stable public API. Prefer including this
 * shim instead of including OpenCV headers directly within IDet sources.
 */

#pragma once

// -----------------------------------------------------------------------------
// opencv.hpp (umbrella header)
// -----------------------------------------------------------------------------
#if defined(__has_include) && __has_include(<opencv4/opencv2/opencv.hpp>)
    #include <opencv4/opencv2/opencv.hpp>
#elif defined(__has_include) && __has_include(<opencv2/opencv.hpp>)
    #include <opencv2/opencv.hpp>
#else
    #error "[IDet] OpenCV header not found: opencv2/opencv.hpp (or opencv4/opencv2/opencv.hpp)"
#endif

// -----------------------------------------------------------------------------
// core.hpp
// -----------------------------------------------------------------------------
#if defined(__has_include) && __has_include(<opencv4/opencv2/core.hpp>)
    #include <opencv4/opencv2/core.hpp>
#elif defined(__has_include) && __has_include(<opencv2/core.hpp>)
    #include <opencv2/core.hpp>
#else
    #error "[IDet] OpenCV header not found: opencv2/core.hpp (or opencv4/opencv2/core.hpp)"
#endif

// -----------------------------------------------------------------------------
// imgcodecs.hpp
// -----------------------------------------------------------------------------
#if defined(__has_include) && __has_include(<opencv4/opencv2/imgcodecs.hpp>)
    #include <opencv4/opencv2/imgcodecs.hpp>
#elif defined(__has_include) && __has_include(<opencv2/imgcodecs.hpp>)
    #include <opencv2/imgcodecs.hpp>
#else
    #error "[IDet] OpenCV header not found: opencv2/imgcodecs.hpp (or opencv4/opencv2/imgcodecs.hpp)"
#endif

// -----------------------------------------------------------------------------
// imgproc.hpp
// -----------------------------------------------------------------------------
#if defined(__has_include) && __has_include(<opencv4/opencv2/imgproc.hpp>)
    #include <opencv4/opencv2/imgproc.hpp>
#elif defined(__has_include) && __has_include(<opencv2/imgproc.hpp>)
    #include <opencv2/imgproc.hpp>
#else
    #error "[IDet] OpenCV header not found: opencv2/imgproc.hpp (or opencv4/opencv2/imgproc.hpp)"
#endif
