#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)

#if defined(YUVV_BUILD_SHARED)
#define YUVV_API __declspec(dllexport)
#else
#define YUVV_API
#endif

#else // Linux / macOS / etc.

#if defined(YUVV_BUILD_SHARED) && __GNUC__ >= 4
#define YUVV_API __attribute__((visibility("default")))
#else
#define YUVV_API
#endif

#endif