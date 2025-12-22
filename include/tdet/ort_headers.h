#pragma once

#define USE_ACL 0

// onnxruntime_cxx_api.h
#if defined(__has_include) && __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#elif defined(__has_include) && __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#else
#error "[ERROR] ONNX Runtime 'onnxruntime_cxx_api.h' header not found"
#endif

// acl_provider_factory.h
#if USE_ACL
#if defined(__has_include) && __has_include(<onnxruntime/core/providers/acl/acl_provider_factory.h>)
#include <onnxruntime/core/providers/acl/acl_provider_factory.h>
#elif defined(__has_include) && __has_include(<acl_provider_factory.h>)
#include <acl_provider_factory.h>
#else
#error "[ERROR] ONNX Runtime 'acl_provider_factory.h' header not found"
#endif
#endif