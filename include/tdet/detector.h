#pragma once
/**
 * @file detector.h
 * @brief Базовый интерфейс детектора для унификации text/face и новых моделей.
 *
 * Конкретные реализации должны определить detect(). Дополнительно можно поддержать I/O binding
 * через prepare_binding()/detect_bound(), чтобы ускорять повторные вызовы на фиксированном размере входа.
 */
#include "geometry.h"

#include <vector>

namespace cv {
class Mat;
}

class IDetector {
  public:
    virtual ~IDetector() = default;

    /** @brief Обычное выполнение без привязки буферов. */
    virtual std::vector<Detection> detect(const cv::Mat& img_bgr, double* ms_out = nullptr) = 0;

    /** @brief Поддерживает ли детектор I/O binding. */
    virtual bool supports_binding() const { return false; }

    /**
     * @brief Подготовка биндинга для фиксированного WxH и указанного числа контекстов (потоков).
     * @return true при успешной подготовке.
     */
    virtual bool prepare_binding(int w, int h, int contexts) { return false; }

    /** @brief Максимальное число потоков при работе через биндинг. */
    virtual int binding_thread_limit() const { return 1; }

    /**
     * @brief Запуск на уже подготовленном биндинге (ctx_idx — индекс контекста/потока).
     * @note Реализация по умолчанию вызывает detect().
     */
    virtual std::vector<Detection> detect_bound(const cv::Mat& img_bgr, int ctx_idx, double* ms_out = nullptr) {
        (void)ctx_idx;
        return detect(img_bgr, ms_out);
    }
};
