#include <iostream>
#include <yuvv.h>

int main(int argc, char** argv) {

    yuv::ViewerConfig cfg;
    if (!yuv::ParseArgs(argc, argv, cfg)) {
        yuv::PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    yuv::YuvViewer viewer(cfg);

    // Optional: hook after preview (you can do logging/benchmarking/etc.)
    // viewer.set_post_preview_callback([](const cv::Mat& bgr, int64_t idx) {
    //     (void)bgr; (void)idx;
    // });

    return viewer.run();
}