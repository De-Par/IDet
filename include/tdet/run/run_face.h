#pragma once
/**
 * @file run_face.h
 * @brief Точки входа single/bench для детектора лиц (SCRFD).
 */
#include "tdet.h"

bool run_face_single(const tdet::FaceDetectorConfig& opt);
bool run_face_bench(const tdet::FaceDetectorConfig& opt);
