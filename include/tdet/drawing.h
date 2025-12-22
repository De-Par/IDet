#pragma once
#include "geometry.h"
#include "opencv_headers.h"

#include <vector>

void draw_and_dump(cv::Mat& img, const std::vector<Detection>& dets, int cols = 1, int rows = 1, bool is_draw = true,
                   bool is_dump = true);