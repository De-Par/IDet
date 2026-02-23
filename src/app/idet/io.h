#pragma once

#include <idet.h>
#include <string>

namespace io {

void draw_detections(const idet::Image& image, const idet::VecQuad& quads, const idet::GridSpec& tiles_rc,
                     const std::string& out_path);

void dump_detections(const idet::VecQuad& quads);

} // namespace io
