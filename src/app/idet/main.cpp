#include "bench.h"
#include "cli.h"
#include "io.h"

#include <idet.h>
#include <iostream>

int main(int argc, char** argv) {
    try {
        // Create timer
        bench::Timer timer{};

        // Create configs
        idet::DetectorConfig det_config{};
        cli::AppConfig app_config{};

        // Parse arguments and fill configs
        if (!cli::parse_arguments(argc, argv, app_config, det_config)) {
            if (app_config.help_requested) return 0;
            std::cerr << "[ERROR] Failed to parse arguments!\n";
            return 1;
        }

        // Setup runtime policy BEFORE hard calculations
        if (app_config.setup_runtime_policy) {
            auto rp_res = idet::setup_runtime_policy(det_config.runtime, /*verbose=*/det_config.verbose);
            if (!rp_res.ok()) {
                std::cerr << "[ERROR] Failed to setup runtime policy: " << rp_res.message << "\n";
                return 1;
            }
        }

        // Create detector
        auto det_res = idet::create_detector(det_config);
        if (!det_res.ok()) {
            std::cerr << "[ERROR] Failed to create detector: " << det_res.status().message << "\n";
            return 1;
        }
        idet::Detector detector = std::move(det_res.value());

        // Load image
        timer.tic();
        auto img_res = idet::load_image(app_config.image_path, idet::PixelFormat::BGR_U8);
        if (!img_res.ok()) {
            std::cerr << "[ERROR] Failed to load image: " << img_res.status().message << "\n";
            return 1;
        }
        idet::Image img = std::move(img_res.value());
        const double img_load_ms = timer.toc_ms();

        if (det_config.verbose) {
            std::cout << "[app_info] load image time, ms : " << img_load_ms << "\n";
        }

        // Pre-warmup (catching early errors)
        {
            auto warm_res = detector.detect(img);
            if (!warm_res.ok()) {
                std::cerr << "[ERROR] Cold start of detector failed: " << warm_res.status().message << "\n";
                return 1;
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
                auto bench_res = detector.detect(img);
                if (!bench_res.ok()) {
                    throw std::runtime_error("[ERROR] Failed to detect: " + bench_res.status().message);
                }
                return bench_res.value().size();
            };

            std::vector<double> samples;
            bench::measure_ms(warm_it, bench_it, samples, det_func, /*progress_bar=*/det_config.verbose);

            auto benc_stat = bench::compute_bench_stat(std::move(samples));

            bench::print_bench_stat(std::cout, benc_stat, /*verbose=*/det_config.verbose, /*use_color=*/true);
        }

        // Combat launch for results
        auto combat_res = detector.detect(img);
        if (!combat_res.ok()) {
            std::cerr << "[ERROR] Failed to detect: " << combat_res.status().message << "\n";
            return 1;
        }
        std::vector<idet::Quad> quads = std::move(combat_res.value());

        // Dump quads points
        if (app_config.is_dump) io::dump_detections(quads);

        // Draw results
        if (app_config.is_draw) io::draw_detections(img, quads, det_config.infer.tiles_dim, app_config.out_path);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "[ERROR] Unknown exception\n";
        return 1;
    }
}
