#pragma once

#include <yuvv.h>

/**
 * @file cli.h
 * @brief Public command-line interface helpers for the YUV Viewer application
 *
 * This header exposes only the minimal CLI-facing API used by the app entrypoints
 */

/**
 * @brief Prints CLI usage/help text for the YUV viewer
 *
 * @param argv0 Executable name (typically argv[0]) used in usage output
 */
void print_usage(const char* argv0);

/**
 * @brief Parses CLI arguments into a @ref ViewerConfig structure
 *
 * @param argc Standard argument count
 * @param argv Standard argument vector
 * @param cfg Output configuration
 * @return true if parsing succeeded and @p cfg is usable; false otherwise
 */
bool parse_args(int argc, char** argv, yuvv::ViewerConfig& cfg);
