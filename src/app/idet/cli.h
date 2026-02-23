#pragma once

#include <idet.h>
#include <iosfwd>
#include <string>

namespace cli {

struct AppConfig {
    std::string image_path;
    std::string out_path = "result.png";
    int bench_iters = 100;
    int warmup_iters = 20;
    bool is_draw = true;
    bool is_dump = true;
    bool setup_runtime_policy = true;
};

void print_config(std::ostream& os, const AppConfig& ac, const idet::DetectorConfig& dc, bool color = true);

[[nodiscard]] bool parse_arguments(int argc, char** argv, AppConfig& ac, idet::DetectorConfig& dc);

} // namespace cli
