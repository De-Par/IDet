#pragma once
#include "tdet.h"

void print_usage(const char* app);

bool parse_arguments(int argc, char** argv, std::unique_ptr<tdet::DetectorConfig>& cfg_out);
