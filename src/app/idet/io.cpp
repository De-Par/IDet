#include "io.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(__has_include) && __has_include(<opencv4/opencv2/opencv.hpp>)
    #include <opencv4/opencv2/opencv.hpp> // IWYU pragma: keep
#elif defined(__has_include) && __has_include(<opencv2/opencv.hpp>)
    #include <opencv2/opencv.hpp> // IWYU pragma: keep
#else
    #error "[ERROR] OpenCV 'opencv.hpp' header not found"
#endif

namespace io {

namespace {

static cv::Mat to_cv_mat_bgr_copy(const idet::Image& image) {
    const auto& v = image.view();
    if (!v.is_valid()) {
        throw std::runtime_error("to_cv_mat_bgr_copy: invalid idet::Image");
    }

    const int ch = v.channels();
    if (ch != 3 && ch != 4) {
        throw std::runtime_error("to_cv_mat_bgr_copy: unsupported PixelFormat");
    }

    const int type = (ch == 3) ? CV_8UC3 : CV_8UC4;

    cv::Mat src(v.height, v.width, type, const_cast<std::uint8_t*>(v.data), static_cast<size_t>(v.stride_bytes));

    cv::Mat out = src.clone();

    if (v.format == idet::PixelFormat::RGB_U8) {
        cv::cvtColor(out, out, cv::COLOR_RGB2BGR);
    } else if (v.format == idet::PixelFormat::RGBA_U8) {
        cv::cvtColor(out, out, cv::COLOR_RGBA2BGRA);
    }

    return out;
}

} // namespace

void dump_detections(const idet::VecQuad& quads) {
    int count = 0;
    std::cout << "Quads:\n";
    for (const auto& d : quads) {
        std::cout << "    " << ++count << " -> " << d[0].x << "," << d[0].y << " " << d[1].x << "," << d[1].y << " "
                  << d[2].x << "," << d[2].y << " " << d[3].x << "," << d[3].y << "\n";
    }
    std::cout << "\n";
}

void draw_detections(const idet::Image& image, const idet::VecQuad& quads, const idet::GridSpec& tiles_rc,
                     const std::string& out_path) {

    cv::Mat bgr = to_cv_mat_bgr_copy(image);

    const cv::Scalar green(0, 255, 0);
    const int box_thickness = 2;

    for (const auto& d : quads) {
        cv::Point points[4] = {
            cv::Point(cvRound(d[0].x), cvRound(d[0].y)),
            cv::Point(cvRound(d[1].x), cvRound(d[1].y)),
            cv::Point(cvRound(d[2].x), cvRound(d[2].y)),
            cv::Point(cvRound(d[3].x), cvRound(d[3].y)),
        };

        for (int i = 0; i < 4; ++i) {
            cv::line(bgr, points[i], points[(i + 1) % 4], green, box_thickness, cv::LINE_AA);
        }
    }

    const int t_rows = tiles_rc.rows;
    const int t_cols = tiles_rc.cols;

    if (t_rows > 0 && t_cols > 0 && t_rows * t_cols > 1) {
        const int img_w = bgr.cols;
        const int img_h = bgr.rows;

        const double cell_w = static_cast<double>(img_w) / static_cast<double>(t_cols);
        const double cell_h = static_cast<double>(img_h) / static_cast<double>(t_rows);

        const cv::Scalar red(0, 0, 255);
        const int grid_thickness = 2;

        for (int c = 1; c < t_cols; ++c) {
            int x = cvRound(c * cell_w);
            x = std::max(0, std::min(x, img_w));
            cv::line(bgr, cv::Point(x, 0), cv::Point(x, img_h), red, grid_thickness, cv::LINE_AA);
        }

        for (int r = 1; r < t_rows; ++r) {
            int y = cvRound(r * cell_h);
            y = std::max(0, std::min(y, img_h));
            cv::line(bgr, cv::Point(0, y), cv::Point(img_w, y), red, grid_thickness, cv::LINE_AA);
        }
    }

    if (!out_path.empty()) {
        if (!cv::imwrite(out_path, bgr)) {
            throw std::runtime_error("draw_detections: cv::imwrite failed: " + out_path);
        }
    }
}

} // namespace io
