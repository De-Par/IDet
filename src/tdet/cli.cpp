#include "cli.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

static bool parse_int(const std::string& s, int& v) {
    try {
        v = std::stoi(s);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_float(const std::string& s, float& v) {
    try {
        v = std::stof(s);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_bool(const std::string& s, bool& v) {
    if (s == "true" || s == "false") {
        v = (s == "true");
        return true;
    }
    try {
        if (std::stoi(s))
            v = true;
        else
            v = false;
        return true;
    } catch (...) {
        return false;
    }
}

void print_usage(const char* app) {
    std::cerr << "Usage:\n"
                 "  "
              << app
              << "--model det.onnx --image img.<jpg, png> [options]\n\n"
                 "Options:\n\n"
                 "|    parameter   | Value |                    Description                        |\n"
                 "  --model           PTH      ONNX model path (required)\n"
                 "  --image           PTH      Input image (required)\n"
                 "  --out             PTH      Output image (draw boxes), default: out.png\n"
                 "  --bin_thresh       F       Binarization threshold, default: 0.3\n"
                 "  --box_thresh       F       Box score threshold, default: 0.3\n"
                 "  --unclip           F       Unclip ratio, default: 1.0\n"
                 "  --side             N       Limit side (no-tiles), default: 960\n"
                 "  --threads_intra    N       ORT intra-op threads, default: 1\n"
                 "  --threads_inter    N       ORT inter-op threads, default: 1\n"
                 "  --min_text_size    N       Minimal diagonal size of box with text, default: 3 px\n"
                 "  --tiles           RxC      Enable tiling (e.g., 3x3)\n"
                 "  --tile_overlap     F       Overlap fraction [0..0.5], default: 0.1\n"
                 "  --tile_omp         N       OpenMP threads for tiles, default: 1\n"
                 "  --nms_iou          F       NMS IoU threshold, default: 0.3\n"
                 "  --apply_sigmoid   0|1      Apply sigmoid on output map, default: 0\n"
                 "  --bind_io         0|1      Use I/O binding (recommended), default: 0\n"
                 "  --fixed_wh        WxH      Fix tile input size (e.g. 480x480). Auto if tiles\n"
                 "  --omp_places      STR      e.g., cores | threads\n"
                 "  --omp_bind        STR      e.g., close | spread\n"
                 "  --bench            N       Benchmark iterations\n"
                 "  --warmup           N       Warmup iterations (bench)\n"
                 "  --is_draw         0|1      Write output image, default: 1\n"
                 "  --verbose         0|1      Log in terminall all information, default: 1\n\n";
}

bool parse_arguments(int argc, char** argv, tdet::Options& o) {
    if (argc < 3) return false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        auto next = [&](std::string& out) -> bool {
            if (i + 1 >= argc) return false;
            out = argv[++i];
            return true;
        };

        if (a == "--model") {
            std::string v;
            if (!next(v)) return false;
            o.model_path = v;
        } else if (a == "--image") {
            std::string v;
            if (!next(v)) return false;
            o.image_path = v;
        } else if (a == "--out") {
            std::string v;
            if (!next(v)) return false;
            o.out_path = v;
        } else if (a == "--bin_thresh") {
            std::string v;
            if (!next(v) || !parse_float(v, o.bin_thresh)) return false;
        } else if (a == "--box_thresh") {
            std::string v;
            if (!next(v) || !parse_float(v, o.box_thresh)) return false;
        } else if (a == "--unclip") {
            std::string v;
            if (!next(v) || !parse_float(v, o.unclip)) return false;
        } else if (a == "--side") {
            std::string v;
            if (!next(v) || !parse_int(v, o.side)) return false;
        } else if (a == "--min_text_size") {
            std::string v;
            if (!next(v) || !parse_int(v, o.min_text_size)) return false;
        } else if (a == "--threads_intra") {
            std::string v;
            if (!next(v) || !parse_int(v, o.ort_intra_threads)) return false;
        } else if (a == "--threads_inter") {
            std::string v;
            if (!next(v) || !parse_int(v, o.ort_inter_threads)) return false;
        } else if (a == "--tiles") {
            std::string v;
            if (!next(v)) return false;
            o.tiles_arg = v;
        } else if (a == "--tile_overlap") {
            std::string v;
            if (!next(v) || !parse_float(v, o.tile_overlap)) return false;
        } else if (a == "--tile_omp") {
            std::string v;
            if (!next(v) || !parse_int(v, o.tile_omp_threads)) return false;
        } else if (a == "--nms_iou") {
            std::string v;
            if (!next(v) || !parse_float(v, o.nms_iou)) return false;
        } else if (a == "--apply_sigmoid") {
            std::string v;
            if (!next(v) || !parse_int(v, o.apply_sigmoid)) return false;
        } else if (a == "--omp_places") {
            std::string v;
            if (!next(v)) return false;
            o.omp_places_cli = v;
        } else if (a == "--omp_bind") {
            std::string v;
            if (!next(v)) return false;
            o.omp_bind_cli = v;
        } else if (a == "--bind_io") {
            std::string v;
            if (!next(v) || !parse_int(v, o.bind_io)) return false;
        } else if (a == "--verbose") {
            std::string v;
            if (!next(v) || !parse_bool(v, o.verbose)) return false;
        } else if (a == "--fixed_wh") {
            std::string v;
            if (!next(v)) return false;
            o.fixed_wh = v;
        } else if (a == "--bench") {
            std::string v;
            if (!next(v) || !parse_int(v, o.bench_iters)) return false;
        } else if (a == "--warmup") {
            std::string v;
            if (!next(v) || !parse_int(v, o.warmup)) return false;
        } else if (a == "--is_draw") {
            std::string v;
            if (!next(v) || !parse_bool(v, o.is_draw)) return false;
        } else {
            std::cerr << "[ERROR] Unknown argument: " << a << "\n";
            return false;
        }
    }

    if (o.model_path.empty() || o.image_path.empty()) return false;

    if (!o.fixed_wh.empty()) {
        size_t pos = o.fixed_wh.find_first_of("xX*");
        if (pos == std::string::npos) {
            std::cerr << "[ERROR] Bad --fixed_wh format. Use WxH\n";
            return false;
        }
        try {
            o.fixedW = std::stoi(o.fixed_wh.substr(0, pos));
            o.fixedH = std::stoi(o.fixed_wh.substr(pos + 1));
        } catch (...) {
            std::cerr << "[ERROR] Bad --fixed_wh numbers\n";
            return false;
        }
        if (o.fixedW < 32 || o.fixedH < 32) {
            std::cerr << "[ERROR] Parameter values of --fixed_wh are too small (<32)\n";
            return false;
        }
        o.fixedW = (o.fixedW + 31) & ~31;
        o.fixedH = (o.fixedH + 31) & ~31;
    }

    auto clampf = [](float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); };

    o.bin_thresh = clampf(o.bin_thresh, 0.f, 1.f);
    o.box_thresh = clampf(o.box_thresh, 0.f, 1.f);
    o.nms_iou = clampf(o.nms_iou, 0.f, 1.f);
    o.tile_overlap = clampf(o.tile_overlap, 0.f, 0.5f);

    o.ort_intra_threads = (o.ort_intra_threads > 0 ? o.ort_intra_threads : 1);
    o.ort_inter_threads = (o.ort_inter_threads > 0 ? o.ort_inter_threads : 1);
    o.tile_omp_threads = (o.tile_omp_threads > 0 ? o.tile_omp_threads : 1);

    if (o.side < 32) o.side = 32;

    if (o.min_text_size < 0) o.min_text_size = 1;

    return true;
}