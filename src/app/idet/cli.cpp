#include "cli.h"

#include "printer.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace cli {

namespace {

inline std::string_view trim_view(std::string_view s) noexcept {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };

    while (!s.empty() && is_space(static_cast<unsigned char>(s.front())))
        s.remove_prefix(1);
    while (!s.empty() && is_space(static_cast<unsigned char>(s.back())))
        s.remove_suffix(1);
    return s;
}

inline void to_lower_inplace(std::string& s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
}

inline std::string lower_copy(std::string_view sv) {
    std::string s(sv);
    to_lower_inplace(s);
    return s;
}

inline bool parse_int(std::string_view sv, int& v) noexcept {
    sv = trim_view(sv);
    if (sv.empty()) return false;

    int tmp = 0;
    const char* first = sv.data();
    const char* last = sv.data() + sv.size();
    auto r = std::from_chars(first, last, tmp);
    if (r.ec != std::errc{} || r.ptr != last) return false;

    v = tmp;
    return true;
}

inline bool parse_float(std::string_view sv, float& v) noexcept {
    sv = trim_view(sv);
    if (sv.empty()) return false;

    std::string tmp(sv);
    char* end = nullptr;
    errno = 0;
    const float val = std::strtof(tmp.c_str(), &end);

    if (end == tmp.c_str() || *end != '\0') return false;
    if (errno == ERANGE) return false;

    v = val;
    return true;
}

inline bool parse_bool(std::string_view sv, bool& v) noexcept {
    std::string s = lower_copy(trim_view(sv));

    if (s == "true" || s == "yes" || s == "on" || s == "1") {
        v = true;
        return true;
    }
    if (s == "false" || s == "no" || s == "off" || s == "0") {
        v = false;
        return true;
    }

    int x = 0;
    if (parse_int(s, x)) {
        v = (x != 0);
        return true;
    }
    return false;
}

inline bool parse_grid_int(std::string_view s_in, idet::GridSpec& g) {
    std::string s = lower_copy(trim_view(s_in));
    if (s.empty() || s == "off" || s == "no" || s == "false" || s == "0") {
        g.rows = 0;
        g.cols = 0;
        return true;
    }

    std::replace(s.begin(), s.end(), '*', 'x');

    const auto xpos = s.find('x');
    if (xpos == std::string::npos) return false;
    if (s.find('x', xpos + 1) != std::string::npos) return false; // строго один разделитель

    const auto lhs = trim_view(std::string_view{s}.substr(0, xpos));
    const auto rhs = trim_view(std::string_view{s}.substr(xpos + 1));
    if (lhs.empty() || rhs.empty()) return false;

    int r = 0, c = 0;
    if (!parse_int(lhs, r) || !parse_int(rhs, c)) return false;
    if (r <= 0 || c <= 0) return false;

    g.rows = r;
    g.cols = c;
    return true;
}

inline idet::EngineKind get_default_engine(idet::Task t) {
    switch (t) {
    case idet::Task::Text:
        return idet::EngineKind::DBNet;
    case idet::Task::Face:
        return idet::EngineKind::SCRFD;
    default:
        return idet::EngineKind::None;
    }
}

inline idet::Task string_to_task(std::string_view task_s) {
    if (task_s == "text") return idet::Task::Text;
    if (task_s == "face") return idet::Task::Face;
    return idet::Task::None;
}

inline std::string task_to_string(idet::Task t) {
    switch (t) {
    case idet::Task::None:
        return "none";
    case idet::Task::Text:
        return "text";
    case idet::Task::Face:
        return "face";
    default:
        return "unknown";
    }
}

inline std::string engine_to_string(idet::EngineKind e) {
    switch (e) {
    case idet::EngineKind::None:
        return "none";
    case idet::EngineKind::DBNet:
        return "dbnet";
    case idet::EngineKind::SCRFD:
        return "scrfd";
    default:
        return "unknown";
    }
}

inline std::string grid_to_string(const idet::GridSpec& g, bool treat_zeros_as_auto = false) {
    if (treat_zeros_as_auto && (g.rows == 0 || g.cols == 0)) return "auto";
    std::ostringstream oss;
    oss << g.rows << "x" << g.cols;
    return oss.str();
}

static void print_usage(const char* app) {
    std::cerr << "Usage:\n"
              << "  " << app << " --model <path.onnx> --mode [text|face] --image <path> [options]\n\n"
              << "Required:\n"
              << "  --model             STR      ONNX model path\n"
              << "  --mode              STR      Detector mode: text | face\n"
              << "  --image             STR      Input image path\n\n"
              << "Generic:\n"
              << "  --is_draw           0|1      Draw image detections. Default: 1\n"
              << "  --is_dump           0|1      Write output image detections. Default: 1\n"
              << "  --output            STR      Output image path (when --is_draw=1). Default: result.png\n"
              << "  --verbose           0|1      Verbose logging. Default: 0\n\n"
              << "Inference:\n"
              << "  --bin_thresh         F       Binarization threshold. Default: 0.3\n"
              << "  --box_thresh         F       Box score threshold. Default: 0.5\n"
              << "  --unclip             F       Unclip ratio. Default: 1.0\n"
              << "  --max_img_size       N       Max side length (no-tiling). Default: 960\n"
              << "  --min_roi_size_w     N       Minimal ROI width. Default: 5\n"
              << "  --min_roi_size_h     N       Minimal ROI height. Default: 5\n"
              << "  --tiles_rc          RxC      Enable tiling grid, e.g. 2x2 / 3x4. Disable: off|no|0\n"
              << "  --tile_overlap       F       Tile overlap fraction. Default: 0.1\n"
              << "  --nms_iou            F       NMS IoU threshold. Default: 0.3\n"
              << "  --use_fast_iou      0|1      Fast IoU option for NMS / overlap checks. Default: 0\n"
              << "  --sigmoid           0|1      Apply sigmoid on output map. Default: 0\n"
              << "  --bind_io           0|1      Use ORT I/O binding. Default: 0\n"
              << "  --fixed_hw          HxW      Fixed input size, e.g. 480x480. Disable: off|no|0\n\n"
              << "Runtime:\n"
              << "  --threads_intra      N       Internal pull of ORT for graph operations (inside node). Default: 1\n"
              << "  --threads_inter      N       Prallelism between nodes of graph. Default: 1\n"
              << "  --tile_omp           N       OpenMP threads for tiling. Default: 1\n"
              << "  --runtime_policy    0|1      Setup runtime policy for session (mem/cpus binding + opencv "
                 "suppression). Default: 1\n"
              << "  --soft_mem_bind     0|1      Apply best-effort memory locality (when supported). Default: 1\n"
              << "  --suppress_opencv   0|1      Globally limit the OpenCV number of threads to single. Default: 1\n\n"
              << "Benchmark:\n"
              << "  --bench_iters        N       Benchmark iterations. Default: 100\n"
              << "  --warmup_iters       N       Warmup iterations. Default: 20\n\n"
              << "Examples:\n"
              << "  " << app << " --mode text --model det.onnx --image img.png --output out.png --is_draw 1\n"
              << "  " << app
              << " --mode text --model det.onnx --image img.png --tiles_rc 2x2 --tile_overlap 0.1 --tile_omp 4\n"
              << "  " << app
              << " --mode face --model scrfd.onnx --image img.jpg --threads_intra 2 --threads_inter 1\n\n";
}

inline bool missing_value(const char* flag) {
    std::cerr << "[ERROR] " << flag << " expects a value\n";
    return false;
}

inline bool invalid_value(const char* flag, const std::string& v, const char* hint = nullptr) {
    std::cerr << "[ERROR] Invalid value for " << flag << ": '" << v << "'";
    if (hint) std::cerr << " (" << hint << ")";
    std::cerr << "\n";
    return false;
}

} // namespace

void print_config(std::ostream& os, const AppConfig& ac, const idet::DetectorConfig& dc, bool color) {
    printer::Printer p{os};
    p.a.enable = color;

    os << "\n========================================================\n\n";
    p.section("Detector&App Configuration");
    os << "\n";

    p.section("Generic", 2);
    p.kv("task", task_to_string(dc.task), 4, p.a.yellow());
    p.kv("engine", engine_to_string(dc.engine), 4, p.a.yellow());
    p.kv_path("model_path", dc.model_path, 4);
    p.kv_path("image_path", ac.image_path, 4);
    p.kv_path("output_path", ac.out_path, 4);

    os << "\n";

    p.section("IO", 2);
    p.kv_bool("verbose", dc.verbose, 4);
    p.kv_bool("is_draw", ac.is_draw, 4);
    p.kv_bool("is_dump", ac.is_dump, 4);

    os << "\n";

    p.section("Bench", 2);
    p.kv("warmup_iters", ac.warmup_iters, 4, p.a.cyan());
    p.kv("bench_iters", ac.bench_iters, 4, p.a.cyan());

    os << "\n";

    p.section("Inference", 2);
    p.kv("bin_thresh", dc.infer.bin_thresh, 4, p.a.cyan());
    p.kv("box_thresh", dc.infer.box_thresh, 4, p.a.cyan());
    p.kv("unclip", dc.infer.unclip, 4, p.a.cyan());

    p.kv("max_img_size", dc.infer.max_img_size, 4, p.a.cyan());
    p.kv("min_roi_size_w", dc.infer.min_roi_size_w, 4, p.a.cyan());
    p.kv("min_roi_size_h", dc.infer.min_roi_size_h, 4, p.a.cyan());

    p.kv("fixed_input_dim", grid_to_string(dc.infer.fixed_input_dim, /*treat_zeros_as_auto=*/true), 4, p.a.cyan());

    const bool tiling_off = (dc.infer.tiles_dim.rows <= 1 && dc.infer.tiles_dim.cols <= 1);
    p.kv("tiles_dim", tiling_off ? std::string("off") : grid_to_string(dc.infer.tiles_dim), 4, p.a.cyan());
    p.kv("tile_overlap", dc.infer.tile_overlap, 4, p.a.cyan());
    p.kv("nms_iou", dc.infer.nms_iou, 4, p.a.cyan());

    p.kv_bool("use_fast_iou", dc.infer.use_fast_iou, 4);
    p.kv_bool("apply_sigmoid", dc.infer.apply_sigmoid, 4);
    p.kv_bool("bind_io", dc.infer.bind_io, 4);

    os << "\n";

    p.section("Runtime", 2);
    p.kv("ort_intra_threads", dc.runtime.ort_intra_threads, 4, p.a.cyan());
    p.kv("ort_inter_threads", dc.runtime.ort_inter_threads, 4, p.a.cyan());
    p.kv("tile_omp_threads", dc.runtime.tile_omp_threads, 4, p.a.cyan());

    p.kv_bool("runtime_policy", ac.setup_runtime_policy, 4);
    if (ac.setup_runtime_policy) {
        p.kv_bool(" - soft_mem_bind", dc.runtime.soft_mem_bind, 4);
        p.kv_bool(" - suppress_opencv", dc.runtime.suppress_opencv, 4);
    }

    os << "\n========================================================\n\n";
}

bool parse_arguments(int argc, char** argv, AppConfig& ac, idet::DetectorConfig& dc) {
    if (argc <= 1) {
        print_usage(argv[0]);
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return false;
        }

        // support --flag=value
        std::string inline_val;
        if (auto eq = a.find('='); eq != std::string::npos) {
            inline_val = a.substr(eq + 1);
            a = a.substr(0, eq);
        }

        auto next = [&](std::string& out) -> bool {
            if (!inline_val.empty()) {
                out = inline_val;
                inline_val.clear();
                return true;
            }
            if (i + 1 >= argc) return false;
            out = argv[++i];
            return true;
        };

        if (a == "--model") {
            std::string v;
            if (!next(v)) return missing_value("--model");
            dc.model_path = v;

        } else if (a == "--mode") {
            std::string v;
            if (!next(v)) return missing_value("--mode");
            v = lower_copy(trim_view(v));
            dc.task = string_to_task(v);
            if (dc.task == idet::Task::None) return invalid_value("--mode", v, "expected text|face");
            dc.engine = get_default_engine(dc.task);

        } else if (a == "--image") {
            std::string v;
            if (!next(v)) return missing_value("--image");
            ac.image_path = v;

        } else if (a == "--output") {
            std::string v;
            if (!next(v)) return missing_value("--output");
            ac.out_path = v;

        } else if (a == "--bin_thresh") {
            std::string v;
            if (!next(v)) return missing_value("--bin_thresh");
            if (!parse_float(v, dc.infer.bin_thresh)) return invalid_value("--bin_thresh", v);

        } else if (a == "--box_thresh") {
            std::string v;
            if (!next(v)) return missing_value("--box_thresh");
            if (!parse_float(v, dc.infer.box_thresh)) return invalid_value("--box_thresh", v);

        } else if (a == "--unclip") {
            std::string v;
            if (!next(v)) return missing_value("--unclip");
            if (!parse_float(v, dc.infer.unclip)) return invalid_value("--unclip", v);

        } else if (a == "--max_img_size") {
            std::string v;
            if (!next(v)) return missing_value("--max_img_size");
            if (!parse_int(v, dc.infer.max_img_size) || dc.infer.max_img_size <= 0)
                return invalid_value("--max_img_size", v, "expected positive integer");

        } else if (a == "--min_roi_size_h") {
            std::string v;
            if (!next(v)) return missing_value("--min_roi_size_h");
            if (!parse_int(v, dc.infer.min_roi_size_h) || dc.infer.min_roi_size_h < 0)
                return invalid_value("--min_roi_size_h", v, "expected integer >= 0");

        } else if (a == "--min_roi_size_w") {
            std::string v;
            if (!next(v)) return missing_value("--min_roi_size_w");
            if (!parse_int(v, dc.infer.min_roi_size_w) || dc.infer.min_roi_size_w < 0)
                return invalid_value("--min_roi_size_w", v, "expected integer >= 0");

        } else if (a == "--threads_intra") {
            std::string v;
            if (!next(v)) return missing_value("--threads_intra");
            if (!parse_int(v, dc.runtime.ort_intra_threads) || dc.runtime.ort_intra_threads <= 0)
                return invalid_value("--threads_intra", v, "expected positive integer");

        } else if (a == "--threads_inter") {
            std::string v;
            if (!next(v)) return missing_value("--threads_inter");
            if (!parse_int(v, dc.runtime.ort_inter_threads) || dc.runtime.ort_inter_threads <= 0)
                return invalid_value("--threads_inter", v, "expected positive integer");

        } else if (a == "--tiles_rc") {
            std::string v;
            if (!next(v)) return missing_value("--tiles_rc");
            if (!parse_grid_int(v, dc.infer.tiles_dim))
                return invalid_value("--tiles_rc", v, "expected RxC or off|no|0");

        } else if (a == "--tile_overlap") {
            std::string v;
            if (!next(v)) return missing_value("--tile_overlap");
            if (!parse_float(v, dc.infer.tile_overlap) || dc.infer.tile_overlap < 0.0f || dc.infer.tile_overlap >= 1.0f)
                return invalid_value("--tile_overlap", v, "expected 0 <= x < 1");

        } else if (a == "--tile_omp") {
            std::string v;
            if (!next(v)) return missing_value("--tile_omp");
            if (!parse_int(v, dc.runtime.tile_omp_threads) || dc.runtime.tile_omp_threads <= 0)
                return invalid_value("--tile_omp", v, "expected positive integer");

        } else if (a == "--nms_iou") {
            std::string v;
            if (!next(v)) return missing_value("--nms_iou");
            if (!parse_float(v, dc.infer.nms_iou) || dc.infer.nms_iou < 0.0f || dc.infer.nms_iou > 1.0f)
                return invalid_value("--nms_iou", v, "expected 0 <= x <= 1");

        } else if (a == "--use_fast_iou") {
            std::string v;
            if (!next(v)) return missing_value("--use_fast_iou");
            if (!parse_bool(v, dc.infer.use_fast_iou))
                return invalid_value("--use_fast_iou", v, "expected 0|1|true|false");

        } else if (a == "--sigmoid") {
            std::string v;
            if (!next(v)) return missing_value("--sigmoid");
            if (!parse_bool(v, dc.infer.apply_sigmoid)) return invalid_value("--sigmoid", v, "expected 0|1|true|false");

        } else if (a == "--soft_mem_bind") {
            std::string v;
            if (!next(v)) return missing_value("--soft_mem_bind");
            if (!parse_bool(v, dc.runtime.soft_mem_bind))
                return invalid_value("--soft_mem_bind", v, "expected 0|1|true|false");

        } else if (a == "--suppress_opencv") {
            std::string v;
            if (!next(v)) return missing_value("--suppress_opencv");
            if (!parse_bool(v, dc.runtime.suppress_opencv))
                return invalid_value("--suppress_opencv", v, "expected 0|1|true|false");

        } else if (a == "--bind_io") {
            std::string v;
            if (!next(v)) return missing_value("--bind_io");
            if (!parse_bool(v, dc.infer.bind_io)) return invalid_value("--bind_io", v, "expected 0|1|true|false");

        } else if (a == "--verbose") {
            std::string v;
            if (!next(v)) return missing_value("--verbose");
            if (!parse_bool(v, dc.verbose)) return invalid_value("--verbose", v, "expected 0|1|true|false");

        } else if (a == "--fixed_hw") {
            std::string v;
            if (!next(v)) return missing_value("--fixed_hw");
            if (!parse_grid_int(v, dc.infer.fixed_input_dim))
                return invalid_value("--fixed_hw", v, "expected HxW or off|no|0");

        } else if (a == "--bench_iters") {
            std::string v;
            if (!next(v)) return missing_value("--bench_iters");
            if (!parse_int(v, ac.bench_iters) || ac.bench_iters <= 0)
                return invalid_value("--bench_iters", v, "expected positive integer");

        } else if (a == "--warmup_iters") {
            std::string v;
            if (!next(v)) return missing_value("--warmup_iters");
            if (!parse_int(v, ac.warmup_iters) || ac.warmup_iters < 0)
                return invalid_value("--warmup_iters", v, "expected integer >= 0");

        } else if (a == "--is_draw") {
            std::string v;
            if (!next(v)) return missing_value("--is_draw");
            if (!parse_bool(v, ac.is_draw)) return invalid_value("--is_draw", v, "expected 0|1|true|false");

        } else if (a == "--is_dump") {
            std::string v;
            if (!next(v)) return missing_value("--is_dump");
            if (!parse_bool(v, ac.is_dump)) return invalid_value("--is_dump", v, "expected 0|1|true|false");

        } else if (a == "--runtime_policy") {
            std::string v;
            if (!next(v)) return missing_value("--runtime_policy");
            if (!parse_bool(v, ac.setup_runtime_policy))
                return invalid_value("--runtime_policy", v, "expected 0|1|true|false");

        } else {
            std::cerr << "[ERROR] Unknown argument: " << a << "\n";
            print_usage(argv[0]);
            return false;
        }
    }

    if (ac.image_path.empty()) {
        std::cerr << "[ERROR] Missing required argument: --image\n";
        print_usage(argv[0]);
        return false;
    }

    if (dc.task == idet::Task::None) {
        std::cerr << "[ERROR] Missing required argument: --mode\n";
        print_usage(argv[0]);
        return false;
    }

    if (dc.engine == idet::EngineKind::None) {
        dc.engine = get_default_engine(dc.task);
    }

    if (dc.model_path.empty()) {
        std::cerr << "[WARN] Missing required argument: --model (will fallback to blob model if available)\n";
    }

    return true;
}

} // namespace cli
