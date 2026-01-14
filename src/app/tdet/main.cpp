#include <iostream>
#include <tdet.h>

int main(int argc, char** argv) {
    // Read options from user
    tdet::Options opt;
    if (!tdet::ParseArgs(argc, argv, opt)) {
        tdet::PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    // Init detector env
    if (!tdet::InitEnvironment(opt)) {
        std::cerr << "[ERROR] Can not setup detector env"
                  << "\n";
        return EXIT_FAILURE;
    }

    if (!tdet::RunDetection(opt)) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
