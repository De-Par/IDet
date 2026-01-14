#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "omp_config.h"

namespace {

inline void set_env_overwrite(const char* key, const std::string& value) {
#if defined(_WIN32)
    _putenv_s(key, value.c_str());
#else
    setenv(key, value.c_str(), 1);
#endif
}

inline void set_env_if_empty(const char* key, const std::string& value) {
    const char* cur = std::getenv(key);
    if (!cur || !*cur) {
        set_env_overwrite(key, value);
    }
}

inline const char* safe_getenv(const char* key) {
    const char* v = std::getenv(key);
    return (v && *v) ? v : nullptr;
}

} // namespace

// Main function: call it AFTER bind_for_threads()! But BEFORE OMP manipulations!
void configure_openmp_affinity(const std::string& omp_places_cli, const std::string& omp_bind_cli, int tile_omp_threads,
                               bool verbose) {
    int threads = tile_omp_threads;
    if (threads <= 0) {
#if defined(_OPENMP)
        threads = omp_get_num_procs();
#else
        unsigned hc = std::thread::hardware_concurrency();
        threads = hc > 0 ? static_cast<int>(hc) : 1;
#endif
    }

    // OMP_NUM_THREADS
    set_env_overwrite("OMP_NUM_THREADS", std::to_string(threads));

    // OMP_PLACES
    if (!omp_places_cli.empty()) {
        set_env_overwrite("OMP_PLACES", omp_places_cli);
    } else {
        set_env_if_empty("OMP_PLACES", "cores");
    }

    // OMP_PROC_BIND
    if (!omp_bind_cli.empty()) {
        set_env_overwrite("OMP_PROC_BIND", omp_bind_cli);
    } else {
        set_env_if_empty("OMP_PROC_BIND", "close");
    }

    // Force setup of OpenMP
#if defined(_OPENMP)
    // 1. disable dynamic commands
    omp_set_dynamic(0);

    // 2. disable neseted parallelism
#if _OPENMP >= 200805 // OpenMP 3.0+
    omp_set_max_active_levels(1);
#else
    omp_set_nested(0);
#endif

    // 3. set number of threads
    omp_set_num_threads(threads);

    // 4. set schedule type
#if _OPENMP >= 200805 // OpenMP 3.0+
    omp_sched_t cur_kind;
    int cur_chunk;
    omp_get_schedule(&cur_kind, &cur_chunk);
    if (cur_kind != omp_sched_static) {
        // chunk==0 => auto block size detection
        omp_set_schedule(omp_sched_static, cur_chunk > 0 ? cur_chunk : 0);
    }
#endif
#endif // _OPENMP

    // Display info
    if (verbose) {
        const char* places = safe_getenv("OMP_PLACES");
        const char* bind = safe_getenv("OMP_PROC_BIND");

        std::cout << "\n=== OpenMP Affinity ===\n";
        std::cout << "Target threads        : " << tile_omp_threads << "\n";
        std::cout << "OMP_NUM_THREADS       : " << threads << "\n";
        std::cout << "OMP_PLACES            : " << (places ? places : "<not set>") << "\n";
        std::cout << "OMP_PROC_BIND         : " << (bind ? bind : "<not set>") << "\n";
#if defined(_OPENMP)
        std::cout << "omp_get_num_procs()   : " << omp_get_num_procs() << "\n";
        std::cout << "omp_get_max_threads() : " << omp_get_max_threads() << "\n";
#endif
        std::cout << "\n";
    }
}