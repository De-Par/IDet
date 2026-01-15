#pragma once
/**
 * @file cli.h
 * @brief Разбор аргументов командной строки и вывод справки для приложения tdet.
 */
#include "tdet.h"

/** @brief Печать справки по ключам CLI. */
void print_usage(const char* app);

/**
 * @brief Разобрать аргументы CLI и создать конкретный DetectorConfig.
 * @param argc количество аргументов
 * @param argv массив аргументов
 * @param cfg_out результат (TextDetectorConfig или FaceDetectorConfig по --mode)
 * @return true при успешном разборе и наличии обязательных параметров (--model, --image)
 */
bool parse_arguments(int argc, char** argv, std::unique_ptr<tdet::DetectorConfig>& cfg_out);
