#pragma once

#include "printer.h"
#include "progress_bar.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef BENCH_COMPILER_FENCE
    #define BENCH_COMPILER_FENCE 1
#endif

namespace bench {

using ClockType = std::chrono::steady_clock;

struct Timer final {
    ClockType::time_point t0{};

    inline void tic() noexcept {
        t0 = ClockType::now();
    }

    inline double toc_ms() const noexcept {
        return std::chrono::duration<double, std::milli>(ClockType::now() - t0).count();
    }
};

struct BenchStat {
    double min_ms = 0, max_ms = 0, avg_ms = 0;
    double p50_ms = 0, p90_ms = 0, p95_ms = 0, p99_ms = 0;
    double geomean_ms = 0;
    double stddev_ms = 0;
    double fps_p50 = 0;
    std::size_t n = 0;
};

namespace detail {

inline double percentile_sorted(const std::vector<double>& x_sorted, double p01) noexcept {
    if (x_sorted.empty()) return 0.0;
    if (p01 <= 0.0) return x_sorted.front();
    if (p01 >= 1.0) return x_sorted.back();

    const double idx = p01 * (static_cast<double>(x_sorted.size()) - 1.0);
    const std::size_t i0 = static_cast<std::size_t>(std::floor(idx));
    const std::size_t i1 = static_cast<std::size_t>(std::ceil(idx));
    if (i0 == i1) return x_sorted[i0];
    const double w = idx - static_cast<double>(i0);
    return x_sorted[i0] * (1.0 - w) + x_sorted[i1] * w;
}

// Anti-optimization helpers (similar spirit to Google Benchmark)
// compiler fence / clobber_memory prevents compiler from reordering memory ops across it
// do_not_optimize tries to keep a value "observable" to the optimizer
#if defined(__GNUC__) || defined(__clang__)

inline void clobber_memory() noexcept {
    asm volatile("" ::: "memory");
}

template <class T> inline void do_not_optimize(T const& value) noexcept {
    asm volatile("" : : "g"(value) : "memory");
}

#else

inline void clobber_memory() noexcept {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

template <class T> inline void do_not_optimize(T const& value) noexcept {
    (void)value;
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

#endif

// Invoke func and try to prevent the compiler from "seeing through" it too much
template <class Func> inline void invoke_bench(Func&& func) {
    using R = std::invoke_result_t<Func&&>;

#if BENCH_COMPILER_FENCE
    clobber_memory();
#endif

    if constexpr (std::is_void_v<R>) {
        std::invoke(std::forward<Func>(func));
#if BENCH_COMPILER_FENCE
        clobber_memory();
#endif
    } else {
        // Keep result alive to prevent DCE on pure-ish functions
        auto result = std::invoke(std::forward<Func>(func));
        do_not_optimize(result);
#if BENCH_COMPILER_FENCE
        clobber_memory();
#endif
    }
}

template <bool WithProgress, class Func>
inline void measure_ms_into(std::size_t warmup, std::size_t iters, std::vector<double>& out_ms, Func&& func) {
    out_ms.assign(iters, 0.0);

    bar::ProgressBar progress_bar{};

    if constexpr (WithProgress) progress_bar.setup(warmup, "Warmup: ", bar::Color::yellow, 45);

    for (std::size_t i = 0; i < warmup; ++i) {
        invoke_bench(func);
        if constexpr (WithProgress) {
            progress_bar.tick();
        }
    }

    if constexpr (WithProgress) {
        progress_bar.done();
        progress_bar.setup(iters, "Bench:  ", bar::Color::green, 45);
    }

    Timer timer{};

    for (std::size_t i = 0; i < iters; ++i) {
        timer.tic();
        invoke_bench(func);
        const double dt_ms = timer.toc_ms();
        out_ms[i] = dt_ms;

        if constexpr (WithProgress) {
            progress_bar.tick();
        }
    }

    if constexpr (WithProgress) progress_bar.done();
}

} // namespace detail

inline BenchStat compute_bench_stat(std::vector<double> vec_ms) {
    BenchStat s{};
    s.n = vec_ms.size();
    if (vec_ms.empty()) return s;

    auto [mn_it, mx_it] = std::minmax_element(vec_ms.begin(), vec_ms.end());
    s.min_ms = *mn_it;
    s.max_ms = *mx_it;

    const double sum = std::accumulate(vec_ms.begin(), vec_ms.end(), 0.0);
    s.avg_ms = sum / static_cast<double>(vec_ms.size());

    // stddev
    {
        double acc = 0.0;
        for (double v : vec_ms) {
            const double d = v - s.avg_ms;
            acc += d * d;
        }
        s.stddev_ms = std::sqrt(acc / static_cast<double>(vec_ms.size()));
    }

    // latency metrics
    std::sort(vec_ms.begin(), vec_ms.end());
    s.p50_ms = detail::percentile_sorted(vec_ms, 0.50);
    s.p90_ms = detail::percentile_sorted(vec_ms, 0.90);
    s.p95_ms = detail::percentile_sorted(vec_ms, 0.95);
    s.p99_ms = detail::percentile_sorted(vec_ms, 0.99);

    // fps@p50
    s.fps_p50 = (s.p50_ms > 0.0) ? (1000.0 / s.p50_ms) : 0.0; // 1s = 1000ms

    // geomean via log (ignore non-positive)
    {
        double log_sum = 0.0;
        std::size_t cnt = 0;
        for (double v : vec_ms) {
            if (v > 0.0) {
                log_sum += std::log(v);
                ++cnt;
            }
        }
        s.geomean_ms = (cnt > 0) ? std::exp(log_sum / static_cast<double>(cnt)) : 0.0;
    }

    return s;
}

inline void print_bench_stat(std::ostream& os, const BenchStat& s, bool verbose, bool use_color = true) {
    if (!verbose) {
        os << "p50_ms: " << s.p50_ms << "\n";
        os << "p90_ms: " << s.p90_ms << "\n";
        os << "p95_ms: " << s.p95_ms << "\n";
        os << "p99_ms: " << s.p99_ms << "\n";
        return;
    }
    printer::Printer p{os};
    p.a.enable = use_color;
    p.key_w = 10;

    os << "\n========================================================\n\n";
    p.section("Benchmark Results");
    os << "\n";

    p.kv("min_ms", s.min_ms, 4, p.a.green());
    p.kv("max_ms", s.max_ms, 4, p.a.red());
    p.kv("avg_ms", s.avg_ms, 4, p.a.yellow());
    p.kv("geo_ms", s.geomean_ms, 4, p.a.cyan());
    p.kv("std_ms", s.stddev_ms, 4, p.a.cyan());

    os << "\n";

    p.kv("p50_ms", s.p50_ms, 4, p.a.cyan());
    p.kv("p90_ms", s.p90_ms, 4, p.a.cyan());
    p.kv("p95_ms", s.p95_ms, 4, p.a.cyan());
    p.kv("p99_ms", s.p99_ms, 4, p.a.cyan());

    os << "\n";

    p.kv("iters", s.n, 4, p.a.bold());
    p.kv("fps@p50", s.fps_p50, 4, p.a.bold());

    os << "\n========================================================\n\n";
}

template <class Func>
inline void measure_ms(std::size_t warmup, std::size_t iters, std::vector<double>& out_ms, Func&& func,
                       bool progress_bar = true) {
    if (progress_bar)
        detail::measure_ms_into<true>(warmup, iters, out_ms, std::forward<Func>(func));
    else
        detail::measure_ms_into<false>(warmup, iters, out_ms, std::forward<Func>(func));
}

} // namespace bench
