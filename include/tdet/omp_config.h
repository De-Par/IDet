#pragma once

#include <string>

void configure_openmp_affinity(const std::string& omp_places_cli, const std::string& omp_bind_cli,
                               const int tile_omp_threads, bool verbose = true);