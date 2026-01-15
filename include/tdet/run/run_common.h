#pragma once
/**
 * @file run_common.h
 * @brief Общие раннеры single/bench для любых детекторов, реализующих IDetector.
 *
 * Функции принимают конфигурацию и фабрику детектора, подготавливают тайлинг/bind_io,
 * выполняют инференс, NMS и при необходимости отрисовку/бенчмарк.
 */
#include "tdet.h"
#include "detector.h"

#include <functional>
#include <memory>

/**
 * @brief Запуск детектора в single-режиме.
 * @param cfg конфигурация (может быть модифицирована для подгонки fixed_wh при тайлинге)
 * @param make_detector фабрика, создающая конкретную реализацию IDetector по cfg
 */
bool run_detector_single(
    tdet::DetectorConfig& cfg,
    const std::function<std::unique_ptr<IDetector>(const tdet::DetectorConfig&)>& make_detector);

/**
 * @brief Запуск детектора в бенч-режиме (warmup + метрики p50/p90/...).
 * @param cfg конфигурация (может быть модифицирована для подгонки fixed_wh при тайлинге)
 * @param make_detector фабрика, создающая конкретную реализацию IDetector по cfg
 */
bool run_detector_bench(
    tdet::DetectorConfig& cfg,
    const std::function<std::unique_ptr<IDetector>(const tdet::DetectorConfig&)>& make_detector);
