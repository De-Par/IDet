#include "cli.h"

#include <yuvv.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {

    yuvv::ViewerConfig cfg;
    bool help_requested = false;

    // Parse options from user
    if (!parse_args(argc, argv, cfg, help_requested)) {
        if (help_requested) return EXIT_SUCCESS;
        print_usage(std::cerr, argv[0]);
        return EXIT_FAILURE;
    }

    // Setup viewer
    yuvv::YuvViewer viewer(cfg);

    /*
        Optional: hook after preview (you can do logging/benchmarking/etc.)

        Example:

        viewer.set_post_preview_callback([](const BgrFrameView& frame, int64_t idx) {
            (void)frame; (void)idx;
        });
    */

    return viewer.run();
}
