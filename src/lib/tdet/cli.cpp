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
              << "--model det.onnx --image img.<jpg, png> --mode <text|face> [options]\n\n"
                 "Options:\n\n"
                 "|    parameter   | Value |                    Description                        |\n"
                 "  --model           PTH      ONNX model path (required)\n"
                 "  --image           PTH      Input image (required)\n"
                 "  --out             PTH      Output image (draw boxes), default: out.png\n"
                 "  --mode            STR      text | face (default: text)\n"
                 "  --bin_thresh       F       Binarization threshold, default: 0.3\n"
                 "  --box_thresh       F       Box score threshold, default: 0.3\n"
                 "  --unclip           F       Unclip ratio, default: 1.0\n"
                 "  --side             N       Limit side (no-tiles), default: 960\n"
                 "  --threads_intra    N       ORT intra-op threads, default: 1\n"
                 "  --threads_inter    N       ORT inter-op threads, default: 1\n"
                 "  --min_text_size    N       Minimal diagonal size of box with text, default: 3 px (text)\n"
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
                 "  --verbose         0|1      Log in terminall all information, default: 1\n"
                 "  --face_min_wh     WxH      SCRFD min face size (default 30x30) (face)\n\n";
}

bool parse_arguments(int argc, char** argv, std::unique_ptr<tdet::DetectorConfig>& cfg_out) {
    std::string mode = "text";
    // First pass for mode
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--mode" && i + 1 < argc) {
            mode = argv[i + 1];
            break;
        }
    }

    if (mode == "face")
        cfg_out = std::make_unique<tdet::FaceDetectorConfig>();
    else
        cfg_out = std::make_unique<tdet::TextDetectorConfig>();

    auto& o = *cfg_out;
    o.kind = (mode == "face") ? tdet::DetectorKind::Face : tdet::DetectorKind::Text;

    if (argc < 3) return false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        auto next = [&](std::string& out) -> bool {
            if (i + 1 >= argc) return false;
            out = argv[++i];
            return true;
        };

        if (a == "--mode") {
            // already handled in pre-pass
            std::string dummy;
            if (!next(dummy)) return false;
            continue;
        } else if (a == "--model") {
            std::string v;
            if (!next(v)) return false;
            o.paths.model_path = v;
        } else if (a == "--image") {
            std::string v;
            if (!next(v)) return false;
            o.paths.image_path = v;
        } else if (a == "--out") {
            std::string v;
            if (!next(v)) return false;
            o.paths.out_path = v;
        } else if (a == "--bin_thresh") {
            std::string v;
            if (!next(v) || !parse_float(v, o.infer.bin_thresh)) return false;
        } else if (a == "--box_thresh") {
            std::string v;
            if (!next(v) || !parse_float(v, o.infer.box_thresh)) return false;
        } else if (a == "--unclip") {
            std::string v;
            if (!next(v) || !parse_float(v, o.infer.unclip)) return false;
        } else if (a == "--side") {
            std::string v;
            if (!next(v) || !parse_int(v, o.infer.limit_side_len)) return false;
        } else if (a == "--min_text_size") {
            std::string v;
            if (!next(v)) return false;
            if (auto* t = dynamic_cast<tdet::TextDetectorConfig*>(&o)) {
                if (!parse_int(v, t->min_text_size)) return false;
            }
        } else if (a == "--threads_intra") {
            std::string v;
            if (!next(v) || !parse_int(v, o.threads.ort_intra_threads)) return false;
        } else if (a == "--threads_inter") {
            std::string v;
            if (!next(v) || !parse_int(v, o.threads.ort_inter_threads)) return false;
        } else if (a == "--tiles") {
            std::string v;
            if (!next(v)) return false;
            o.tiling.grid = v;
        } else if (a == "--tile_overlap") {
            std::string v;
            if (!next(v) || !parse_float(v, o.tiling.overlap)) return false;
        } else if (a == "--tile_omp") {
            std::string v;
            if (!next(v) || !parse_int(v, o.threads.tile_omp_threads)) return false;
        } else if (a == "--nms_iou") {
            std::string v;
            if (!next(v) || !parse_float(v, o.output.nms_iou)) return false;
        } else if (a == "--apply_sigmoid") {
            std::string v;
            if (!next(v) || !parse_bool(v, o.infer.apply_sigmoid)) return false;
        } else if (a == "--omp_places") {
            std::string v;
            if (!next(v)) return false;
            o.threads.omp_places_cli = v;
        } else if (a == "--omp_bind") {
            std::string v;
            if (!next(v)) return false;
            o.threads.omp_bind_cli = v;
        } else if (a == "--bind_io") {
            std::string v;
            if (!next(v) || !parse_int(v, o.tiling.bind_io)) return false;
        } else if (a == "--verbose") {
            std::string v;
            if (!next(v) || !parse_bool(v, o.output.verbose)) return false;
        } else if (a == "--fixed_wh") {
            std::string v;
            if (!next(v)) return false;
            o.tiling.fixed_wh = v;
        } else if (a == "--bench") {
            std::string v;
            if (!next(v) || !parse_int(v, o.bench.bench_iters)) return false;
        } else if (a == "--warmup") {
            std::string v;
            if (!next(v) || !parse_int(v, o.bench.warmup)) return false;
        } else if (a == "--is_draw") {
            std::string v;
            if (!next(v) || !parse_bool(v, o.output.is_draw)) return false;
        } else if (a == "--face_min_wh") {
            std::string v;
            if (!next(v)) return false;
            size_t pos = v.find_first_of("xX*");
            if (pos == std::string::npos) return false;
            if (auto* f = dynamic_cast<tdet::FaceDetectorConfig*>(cfg_out.get())) {
                try {
                    f->min_size_w = std::stoi(v.substr(0, pos));
                    f->min_size_h = std::stoi(v.substr(pos + 1));
                } catch (...) {
                    return false;
                }
            }
        } else {
            std::cerr << "[ERROR] Unknown argument: " << a << "\n";
            return false;
        }
    }

    if (o.paths.image_path.empty()) return false;

    if (!o.tiling.fixed_wh.empty()) {
        size_t pos = o.tiling.fixed_wh.find_first_of("xX*");
        if (pos == std::string::npos) {
            std::cerr << "[ERROR] Bad --fixed_wh format. Use WxH\n";
            return false;
        }
        try {
            o.infer.fixed_W = std::stoi(o.tiling.fixed_wh.substr(0, pos));
            o.infer.fixed_H = std::stoi(o.tiling.fixed_wh.substr(pos + 1));
        } catch (...) {
            std::cerr << "[ERROR] Bad --fixed_wh numbers\n";
            return false;
        }
        if (o.infer.fixed_W < 32 || o.infer.fixed_H < 32) {
            std::cerr << "[ERROR] Parameter values of --fixed_wh are too small (<32)\n";
            return false;
        }
        o.infer.fixed_W = (o.infer.fixed_W + 31) & ~31;
        o.infer.fixed_H = (o.infer.fixed_H + 31) & ~31;
    }

    auto clampf = [](float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); };

    o.infer.bin_thresh = clampf(o.infer.bin_thresh, 0.f, 1.f);
    o.infer.box_thresh = clampf(o.infer.box_thresh, 0.f, 1.f);
    o.output.nms_iou = clampf(o.output.nms_iou, 0.f, 1.f);
    o.tiling.overlap = clampf(o.tiling.overlap, 0.f, 0.5f);

    o.threads.ort_intra_threads = (o.threads.ort_intra_threads > 0 ? o.threads.ort_intra_threads : 1);
    o.threads.ort_inter_threads = (o.threads.ort_inter_threads > 0 ? o.threads.ort_inter_threads : 1);
    o.threads.tile_omp_threads = (o.threads.tile_omp_threads > 0 ? o.threads.tile_omp_threads : 1);

    if (o.infer.limit_side_len < 32) o.infer.limit_side_len = 32;

    if (auto* t = dynamic_cast<tdet::TextDetectorConfig*>(&o)) {
        if (t->min_text_size < 0) t->min_text_size = 1;
    }

    return true;
}
