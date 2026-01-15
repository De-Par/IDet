#pragma once
/**
 * @file tiling.h
 * @brief Разбиение изображения на тайлы и запуск детекторов (DBNet/SCRFD/любой IDetector) по тайлам.
 */
#include "dbnet.h"
#include "face_detector.h"
#include "opencv_headers.h"

#include <string>
#include <vector>

/** @brief Размер сетки тайлов (rows x cols). */
struct GridSpec {
    int rows{1};
    int cols{1};
};

/** @brief Парсинг строки RxC (или off/0) в GridSpec; возвращает true, если тайлинг включён. */
bool parse_tiles(const std::string& s, GridSpec& g);

/** @brief Тайлинг без I/O binding для DBNet. */
std::vector<Detection> infer_tiled_unbound(const cv::Mat& img, DBNet& det, const GridSpec& g, const float overlap,
                                           double* ms_out, const int tile_omp_threads);

/** @brief Тайлинг с I/O binding для DBNet (fixed_w/fixed_h обязательны). */
std::vector<Detection> infer_tiled_bound(const cv::Mat& img, DBNet& det, const GridSpec& g, const float overlap,
                                         double* ms_out, const int tile_omp_threads, int fixed_w, int fixed_h);

/** @brief Тайлинг для SCRFD (face), поддерживает bind_io при наличии fixed_w/fixed_h. */
std::vector<Detection> infer_tiled_face(const cv::Mat& img, FaceDetector& det, const GridSpec& g, const float overlap,
                                        double* ms_out, const int tile_omp_threads, bool bind_io = false,
                                        int fixed_w = 0, int fixed_h = 0);

/** @brief Универсальный путь для любых детекторов, реализующих IDetector (используется в общих раннерах). */
std::vector<Detection> infer_tiled_generic(const cv::Mat& img, IDetector& det, const GridSpec& g, const float overlap,
                                           double* ms_out, const int tile_omp_threads, bool use_bind, int fixed_w,
                                           int fixed_h);
