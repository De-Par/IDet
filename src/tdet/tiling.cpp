#include "tiling.h"

#include "nms.h"
#include "timer.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <opencv2/opencv.hpp>
#else
#include <opencv4/opencv2/opencv.hpp>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

static inline std::string trim_copy(std::string s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](unsigned char c) { return !std::isspace(c); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [&](unsigned char c) { return !std::isspace(c); }).base(), s.end());
    return s;
}

static inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static inline void replace_all(std::string& s, char from, char to) {
    std::replace(s.begin(), s.end(), from, to);
}

static inline void replace_all_str(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// Create tiles (with overlap in relative [0..1]) and return rects
static std::vector<cv::Rect> make_tiles(const cv::Size& img_sz, const GridSpec& g, const float overlap_rel) {
    const int W = img_sz.width, H = img_sz.height;
    if (W <= 0 || H <= 0 || g.cols <= 0 || g.rows <= 0) return {};

    const float ov = std::clamp(overlap_rel, 0.0f, 0.9f);

    const int tw = (W + g.cols - 1) / g.cols;
    const int th = (H + g.rows - 1) / g.rows;
    const int ow = int(std::round(tw * ov));
    const int oh = int(std::round(th * ov));

    std::vector<cv::Rect> rects;
    rects.reserve((size_t)g.rows * (size_t)g.cols);

    for (int r = 0; r < g.rows; ++r) {
        for (int c = 0; c < g.cols; ++c) {
            const int x0 = std::max(0, c * tw - (c > 0 ? ow : 0));
            const int y0 = std::max(0, r * th - (r > 0 ? oh : 0));
            const int x1 = std::min(W, (c + 1) * tw + (c + 1 < g.cols ? ow : 0));
            const int y1 = std::min(H, (r + 1) * th + (r + 1 < g.rows ? oh : 0));
            rects.emplace_back(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));
        }
    }
    return rects;
}

static inline void offset_detection(Detection& d, const int dx, const int dy) {
    for (int k = 0; k < 4; ++k) {
        d.pts[k].x += dx;
        d.pts[k].y += dy;
    }
}

bool parse_tiles(const std::string& s_in, GridSpec& g) {
    g = GridSpec{1, 1};
    std::string s = to_lower(trim_copy(s_in));
    if (s.empty() || s == "off" || s == "no" || s == "0") return false;

    replace_all(s, '*', 'x');
    replace_all_str(s, u8"×", "x"); // UTF-8 times sign

    const auto xpos = s.find('x');
    if (xpos == std::string::npos) return false;

    const auto a = trim_copy(s.substr(0, xpos));
    const auto b = trim_copy(s.substr(xpos + 1));

    int r = 1, c = 1;
    try {
        c = std::max(1, std::stoi(a));
        r = std::max(1, std::stoi(b));
    } catch (...) {
        return false;
    }

    g.rows = r;
    g.cols = c;
    return (r * c > 1);
}

std::vector<Detection> infer_tiled_bound(const cv::Mat& img, DBNet& det, const GridSpec& g, const float overlap,
                                         double* ms_out, const int tile_omp_threads) {
    const auto rects = make_tiles(img.size(), g, overlap);
    const int num_tiles = (int)rects.size();
    if (num_tiles == 0) {
        if (ms_out) *ms_out = 0.0;
        return {};
    }

    int n_threads = 1;
#if defined(_OPENMP)
    n_threads = (tile_omp_threads > 0) ? tile_omp_threads : omp_get_max_threads();
#endif

    std::vector<std::vector<Detection>> tls_dets((size_t)n_threads);
    for (auto& v : tls_dets)
        v.reserve((size_t)(num_tiles * 4 / std::max(1, n_threads) + 8));

    Timer t;
    t.tic();

#if defined(_OPENMP)
#pragma omp parallel num_threads(n_threads)
    {
        const int tid = omp_get_thread_num();
        auto& local = tls_dets[(size_t)tid];

#pragma omp for schedule(static)
        for (int i = 0; i < num_tiles; ++i) {
            const cv::Rect& rc = rects[(size_t)i];
            cv::Mat tile = img(rc); // ROI view, no clone

            auto dets = det.infer_bound(tile, tid, nullptr);
            for (auto& d : dets)
                offset_detection(d, rc.x, rc.y);

            local.insert(local.end(), std::make_move_iterator(dets.begin()), std::make_move_iterator(dets.end()));
        }
    }
#else
    auto& local = tls_dets[0];
    for (int i = 0; i < num_tiles; ++i) {
        const cv::Rect& rc = rects[(size_t)i];
        cv::Mat tile = img(rc);

        auto dets = det.infer_bound(tile, 0, nullptr);
        for (auto& d : dets)
            offset_detection(d, rc.x, rc.y);

        local.insert(local.end(), std::make_move_iterator(dets.begin()), std::make_move_iterator(dets.end()));
    }
#endif

    std::vector<Detection> all;
    {
        size_t total = 0;
        for (const auto& v : tls_dets)
            total += v.size();
        all.reserve(total);
        for (auto& v : tls_dets)
            all.insert(all.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
    }

    if (ms_out) *ms_out = t.toc_ms();
    return all;
}

std::vector<Detection> infer_tiled_unbound(const cv::Mat& img, DBNet& det, const GridSpec& g, const float overlap,
                                           double* ms_out, const int tile_omp_threads) {
    const auto rects = make_tiles(img.size(), g, overlap);
    const int n_tiles = (int)rects.size();
    if (n_tiles == 0) {
        if (ms_out) *ms_out = 0.0;
        return {};
    }

    int n_threads = 1;
#if defined(_OPENMP)
    n_threads = (tile_omp_threads > 0) ? tile_omp_threads : omp_get_max_threads();
#endif

    std::vector<std::vector<Detection>> per_thread((size_t)n_threads);

    Timer t;
    t.tic();

#if defined(_OPENMP)
#pragma omp parallel num_threads(n_threads)
#endif
    {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        std::vector<Detection> local;
        local.reserve((size_t)(n_tiles * 4 / std::max(1, n_threads) + 8));

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < n_tiles; ++i) {
            const cv::Rect rc = rects[(size_t)i];
            cv::Mat tile = img(rc);

            auto dets = det.infer_unbound(tile, nullptr);
            for (auto& d : dets)
                offset_detection(d, rc.x, rc.y);

            local.insert(local.end(), std::make_move_iterator(dets.begin()), std::make_move_iterator(dets.end()));
        }

        per_thread[(size_t)tid] = std::move(local);
    }

    std::vector<Detection> all;
    {
        size_t total = 0;
        for (const auto& v : per_thread)
            total += v.size();
        all.reserve(total);
        for (auto& v : per_thread)
            all.insert(all.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
    }

    if (ms_out) *ms_out = t.toc_ms();
    return all;
}