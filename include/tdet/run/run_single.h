#pragma once
/**
 * @file run_single.h
 * @brief Точка входа single-режима для текстового детектора (DBNet).
 */
#include "tdet.h"

bool run_single(const tdet::TextDetectorConfig& opt);
