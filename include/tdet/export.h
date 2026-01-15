#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)

    #if defined(TDET_BUILD_SHARED)
        #define TDET_API __declspec(dllexport)
    #else
        #define TDET_API
    #endif

#else // Linux / macOS / etc.

    #if defined(TDET_BUILD_SHARED) && __GNUC__ >= 4
        #define TDET_API __attribute__((visibility("default")))
    #else
        #define TDET_API
    #endif

#endif