#pragma once

#include <vector>

#include "geometry.h"

struct AABB {
    float minx, miny, maxx, maxy;
};

std::vector<Detection> nms_poly(const std::vector<Detection> &dets, float iou_thr);