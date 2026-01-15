#include <iostream>
#include <tdet.h>

int main(int argc, char** argv) {
    // Read options from user
    std::unique_ptr<tdet::DetectorConfig> cfg;
    if (!tdet::ParseArgs(argc, argv, cfg) || !cfg) {
        tdet::PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    // Init detector env
    if (!tdet::InitEnvironment(*cfg)) {
        std::cerr << "[ERROR] Can not setup detector env"
                  << "\n";
        return EXIT_FAILURE;
    }

    if (!tdet::RunDetection(*cfg)) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
