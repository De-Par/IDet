#pragma once
#include <cstddef>

struct ModelBlob {
    const unsigned char* data = nullptr;
    std::size_t size = 0;
};

#if defined(TDET_HAVE_FACE_BLOB)
extern unsigned char scrfd_model[];
extern unsigned int scrfd_model_len;
inline ModelBlob get_face_blob() { return {scrfd_model, scrfd_model_len}; }
#else
inline ModelBlob get_face_blob() { return {nullptr, 0}; }
#endif

#if defined(TDET_HAVE_TEXT_BLOB)
extern unsigned char dbnet_model[];
extern unsigned int dbnet_model_len;
inline ModelBlob get_text_blob() { return {dbnet_model, dbnet_model_len}; }
#else
inline ModelBlob get_text_blob() { return {nullptr, 0}; }
#endif
