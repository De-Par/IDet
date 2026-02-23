#include "bench.h"
#include "cli.h"
#include "io.h"

#include <idet.h>
#include <iostream>

int main(int argc, char** argv) {
    // Create timer
    bench::Timer timer{};

    // Create configs
    idet::DetectorConfig det_config{};
    cli::AppConfig app_config{};

    // Parse arguments and fill configs
    if (!cli::parse_arguments(argc, argv, app_config, det_config)) {
        throw std::runtime_error("[ERROR] Failed to parse arguments!");
    }

    // Setup runtime policy BEFORE hard calculations
    if (app_config.setup_runtime_policy) {
        auto rp_res = idet::setup_runtime_policy(det_config.runtime, /*verbose=*/det_config.verbose);
        if (!rp_res.ok()) {
            throw std::runtime_error("[ERROR] Failed to setup runtime policy: " + rp_res.message);
        }
    }

    // Create detector
    auto det_res = idet::create_detector(det_config);
    if (!det_res.ok()) {
        throw std::runtime_error("[ERROR] Failed to create detector: " + det_res.status().message);
    }
    idet::Detector detector = std::move(det_res.value());

    // Bind io
    if (det_config.infer.bind_io) {
        const int fixed_w = det_config.infer.fixed_input_dim.cols;
        const int fixed_h = det_config.infer.fixed_input_dim.rows;
        const int tile_threads = det_config.runtime.tile_omp_threads;

        auto bind_res = detector.prepare_binding(fixed_w, fixed_h, tile_threads);
        if (!bind_res.ok()) {
            throw std::runtime_error("[ERROR] Failed to bind input/output buffers: " + bind_res.message);
        }
    }

    // Load image
    timer.tic();
    auto img_res = idet::load_image(app_config.image_path, idet::PixelFormat::BGR_U8);
    if (!img_res.ok()) {
        throw std::runtime_error("[ERROR] Failed to load image: " + img_res.status().message);
    }
    idet::Image img = std::move(img_res.value());
    const double img_load_ms = timer.toc_ms();

    // Pre-warmup (catching early errors)
    {
        auto warm_res = detector.detect(img);
        if (!warm_res.ok()) {
            throw std::runtime_error("[ERROR] Cold start of detector failed: " + warm_res.status().message);
        }
    }

    // Display config
    if (det_config.verbose) {
        cli::print_config(std::cout, app_config, det_config);
    }

    // Bench
    if (app_config.bench_iters > 0) {
        std::size_t warm_it = static_cast<std::size_t>(app_config.warmup_iters);
        std::size_t bench_it = static_cast<std::size_t>(app_config.bench_iters);

        auto det_func = [&]() {
            auto det_res = detector.detect(img);
            if (!det_res.ok()) {
                throw std::runtime_error("[ERROR] Failed to detect: " + det_res.status().message);
            }
            return det_res.value().size();
        };

        std::vector<double> samples;
        bench::measure_ms(warm_it, bench_it, samples, det_func, /*progress_bar=*/det_config.verbose);

        auto benc_stat = bench::compute_bench_stat(std::move(samples));

        bench::print_bench_stat(std::cout, benc_stat, /*verbose=*/det_config.verbose, /*use_color=*/true);
    }

    // Combat launch for results
    auto r = detector.detect(img);
    if (!r.ok()) {
        throw std::runtime_error("[ERROR] Failed to detect: " + r.status().message);
    }
    std::vector<idet::Quad> quads = std::move(r.value());

    // Show useful app info
    if (det_config.verbose) {
        std::cout << "[app_info] load image time, ms : " << img_load_ms << "\n";
        std::cout << "[app_info] num detection quads : " << quads.size() << "\n";
        std::cout << "\n";
    } else {
        std::cout << "dets_n: " << quads.size() << "\n";
    }

    // Dump quads points
    if (app_config.is_dump) io::dump_detections(quads);

    // Draw results
    if (app_config.is_draw) io::draw_detections(img, quads, det_config.infer.tiles_dim, app_config.out_path);

    return 0;
}
