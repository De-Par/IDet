#pragma once
#include "opencv_headers.h"

#include <array>
#include <vector>

#define USE_FAST_IOU 0

struct Detection {
    std::array<cv::Point2f, 4> pts;
    float score;
};

void order_quad(cv::Point2f pts[4]);

float contour_score(const cv::Mat& prob, const std::vector<cv::Point>& contour);

float poly_area(const std::vector<cv::Point2f>& p);

float quad_iou(const std::array<cv::Point2f, 4>& A, const std::array<cv::Point2f, 4>& B);

float aabb_iou(const std::array<cv::Point2f, 4>& A, const std::array<cv::Point2f, 4>& B);

std::pair<int, int> aspect_fit32(const int iw, const int ih, const int side);

std::pair<int, int> aspect_fit32(const int iw, const int ih, const float scale);