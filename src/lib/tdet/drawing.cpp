#include "drawing.h"

#include <iostream>

void draw_and_dump(cv::Mat& img, const std::vector<Detection>& dets, int cols, int rows, bool is_draw, bool is_dump) {
    if (is_dump) std::cout << "Boxes:\n";

    int cnt = 0;
    const cv::Scalar green(0, 255, 0);
    const int box_thickness = 2;

    for (const auto& d : dets) {
        cv::Point p[4] = {
            cv::Point(cvRound(d.pts[0].x), cvRound(d.pts[0].y)),
            cv::Point(cvRound(d.pts[1].x), cvRound(d.pts[1].y)),
            cv::Point(cvRound(d.pts[2].x), cvRound(d.pts[2].y)),
            cv::Point(cvRound(d.pts[3].x), cvRound(d.pts[3].y)),
        };

        // Draw
        if (is_draw) {
            for (int i = 0; i < 4; ++i)
                cv::line(img, p[i], p[(i + 1) % 4], green, box_thickness, cv::LINE_AA);
        }

        // Dump
        if (is_dump) {
            std::cout << "\t" << ++cnt << " -> " << p[0].x << "," << p[0].y << " " << p[1].x << "," << p[1].y << " "
                      << p[2].x << "," << p[2].y << " " << p[3].x << "," << p[3].y << " "
                      << "-> score=" << d.score << std::endl;
        }
    }

    // Draw tiling grid
    if (is_draw && rows > 0 && cols > 0 && rows * cols > 1) {
        const int W = img.cols;
        const int H = img.rows;

        const double cell_w = static_cast<double>(W) / static_cast<double>(cols);
        const double cell_h = static_cast<double>(H) / static_cast<double>(rows);

        const cv::Scalar red(0, 0, 255);
        const int grid_thickness = 2;

        // Vertical grid lines
        for (int c = 1; c < cols; ++c) {
            int x = cvRound(c * cell_w);
            if (x < 0) x = 0;
            if (x > W) x = W;
            cv::line(img, cv::Point(x, 0), cv::Point(x, H), red, grid_thickness, cv::LINE_AA);
        }

        // Horizontal grid lines
        for (int r = 1; r < rows; ++r) {
            int y = cvRound(r * cell_h);
            if (y < 0) y = 0;
            if (y > H) y = H;
            cv::line(img, cv::Point(0, y), cv::Point(W, y), red, grid_thickness, cv::LINE_AA);
        }
    }
}