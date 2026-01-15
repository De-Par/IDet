#pragma once
/**
 * @file tdet.h
 * @brief Публичные конфиги детекторов и точки входа API (InitEnvironment, RunDetection, ParseArgs).
 *
 * Структуры в этом файле описывают параметры путей, инференса, потоков, тайлинга, бенчмаркинга и вывода
 * для всех детекторов. DetectorConfig — базовый родитель; TextDetectorConfig и FaceDetectorConfig задают
 * специфические поля. Файл — основной контракт для приложений, использующих библиотеку tdet.
 */
#include "export.h"

#include <memory>
#include <string>

namespace tdet {

enum class DetectorKind {
    Text,
    Face,
    Unknown
};

/** @brief Пути к модели, входному изображению и выходному файлу. */
struct Paths {
    std::string model_path;
    std::string image_path;
    std::string out_path = "out.png";
};

/** @brief Общие параметры инференса (пороги, ограничение размера, fixed input). */
struct InferenceParams {
    float bin_thresh = 0.3f;
    float box_thresh = 0.3f;
    float unclip = 1.0f;
    int limit_side_len = 960;
    bool apply_sigmoid = false;
    int fixed_W = 0;
    int fixed_H = 0;
};

/** @brief Настройки потоков ORT и OpenMP. */
struct Threading {
    int ort_intra_threads = 1;
    int ort_inter_threads = 1;
    int tile_omp_threads = 1;
    std::string omp_places_cli;
    std::string omp_bind_cli;
};

/** @brief Параметры тайлинга и биндинга I/O. */
struct TilingParams {
    float overlap = 0.1f;
    std::string grid;
    int bind_io = 0;
    std::string fixed_wh;
};

/** @brief Параметры бенчмарка. */
struct Benchmarking {
    int bench_iters = 0;
    int warmup = 0;
};

/** @brief Параметры постобработки и логирования. */
struct OutputParams {
    float nms_iou = 0.3f;
    bool is_draw = false;
    bool verbose = true;
};

/** @brief Базовый конфиг детектора (родитель для текстового и face-конфигов). */
struct DetectorConfig {
    DetectorKind kind = DetectorKind::Unknown;
    Paths paths;
    InferenceParams infer;
    Threading threads;
    TilingParams tiling;
    Benchmarking bench;
    OutputParams output;

    virtual ~DetectorConfig() = default;
};

/** @brief Конфиг DBNet (детекция текста). */
struct TextDetectorConfig : public DetectorConfig {
    int min_text_size = 3;

    TextDetectorConfig() {
        kind = DetectorKind::Text;
    }
};

/** @brief Конфиг SCRFD (детекция лиц). */
struct FaceDetectorConfig : public DetectorConfig {
    int min_size_w = 10;
    int min_size_h = 10;

    FaceDetectorConfig() {
        kind = DetectorKind::Face;
        infer.apply_sigmoid = false; // SCRFD outputs are already probabilities
        infer.box_thresh = 0.6f;
        output.nms_iou = 0.4f;
    }
};

/** @brief Инициализация окружения: пиннинг CPU/памяти, настройка OpenMP. */
TDET_API bool InitEnvironment(DetectorConfig& cfg);

/** @brief Запуск детектора (single или bench в зависимости от cfg.bench). */
TDET_API bool RunDetection(DetectorConfig& cfg);

/** @brief Разбор CLI-аргументов в конкретный конфиг (DetectorKind определяется по --mode). */
TDET_API bool ParseArgs(int argc, char** argv, std::unique_ptr<DetectorConfig>& cfg_out);

/** @brief Вывод справки по CLI. */
TDET_API void PrintUsage(const char* app);

} // namespace tdet
