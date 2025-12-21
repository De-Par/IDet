#pragma once
#include "tdet.h"

void print_usage(const char *app);

bool parse_arguments(int argc, char **argv, tdet::Options &opt);
