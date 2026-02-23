/**
 * @file omp_config.cpp
 * @ingroup idet_platform
 * @brief OpenMP affinity and determinism configuration implementation.
 *
 * @details
 * Implements @ref idet::platform::configure_openmp_affinity and related diagnostics.
 * Focus:
 * - determine an effective OpenMP thread count (requested value or current affinity mask),
 * - set process-global OMP/GOMP/KMP environment knobs early (best-effort),
 * - optionally apply runtime API calls when OpenMP is enabled (_OPENMP defined),
 * - provide a verbose dump of effective configuration for troubleshooting.
 *
 * Platform notes:
 * - Linux uses sched_getaffinity() to derive allowed CPU count.
 * - Non-Linux uses std::thread::hardware_concurrency() fallback.
 */

#include "platform/omp_config.h"

#include <climits>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#if defined(__linux__)
    #include <sched.h>
    #include <unistd.h>
#endif

#if defined(_OPENMP)
    #include <dlfcn.h>
    #include <omp.h>
#endif

namespace idet::platform {

static inline void set_env_overwrite(const char* key, const std::string& value) {
#if defined(_WIN32)
    _putenv_s(key, value.c_str());
#else
    (void)setenv(key, value.c_str(), 1);
#endif
}

static inline void unset_env(const char* key) {
#if !defined(_WIN32)
    unsetenv(key);
#else
    _putenv_s(key, "");
#endif
}

static inline const char* safe_getenv(const char* key) {
    const char* v = std::getenv(key);
    return (v && *v) ? v : nullptr;
}

#if defined(__linux__)
static int effective_cpu_count_from_affinity() {
    long nconf = sysconf(_SC_NPROCESSORS_CONF);
    int max_cpu = (nconf > 0) ? static_cast<int>(nconf) : CPU_SETSIZE;
    if (max_cpu <= 0) max_cpu = CPU_SETSIZE;

    #if defined(CPU_ALLOC)
    size_t setsize = CPU_ALLOC_SIZE(max_cpu);
    cpu_set_t* set = CPU_ALLOC(max_cpu);
    if (!set) return 1;
    CPU_ZERO_S(setsize, set);

    if (sched_getaffinity(0, setsize, set) != 0) {
        CPU_FREE(set);
        return 1;
    }

    int cnt = 0;
    for (int c = 0; c < max_cpu; ++c)
        if (CPU_ISSET_S(c, setsize, set)) ++cnt;
    CPU_FREE(set);

    return cnt > 0 ? cnt : 1;
    #else
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return 1;
    int cnt = 0;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &set)) ++cnt;
    return cnt > 0 ? cnt : 1;
    #endif
}
#endif // __linux__

static inline void dump_env_kv(const char* key, const char* label = nullptr) {
    const char* v = safe_getenv(key);
    std::cout << "  " << (label ? label : key) << " = " << (v ? v : "<not set>") << "\n";
}

/**
 * @brief Prints OpenMP runtime and environment diagnostics to stdout.
 *
 * @details
 * This helper prints:
 * - the value of @_OPENMP (when available),
 * - best-effort identification of the shared library providing OpenMP symbols,
 * - common OpenMP environment variables,
 * - selected OpenMP runtime query values.
 *
 * @warning
 * Querying the OpenMP runtime via @c omp_get_* APIs may trigger runtime initialization on some
 * systems. Use this primarily for diagnostics, and prefer calling it after all environment
 * variables have already been configured.
 */
void dump_openmp_runtime() {
#if !defined(_OPENMP)
    std::cout << "\n[OMP] OpenMP is not enabled in this build (_OPENMP not defined)\n" << std::flush;
#else
    std::cout << "\n[OMP] OpenMP Runtime\n";
    std::cout << "  _OPENMP = " << std::to_string(_OPENMP) << "\n";

    // Best-effort: resolve which shared library provides omp_* symbols.
    Dl_info info{};
    if (dladdr((void*)&omp_get_max_threads, &info) && info.dli_fname) {
        std::cout << "  runtime_so = " << info.dli_fname << "\n";
    } else {
        std::cout << "  runtime_so = <unknown>\n";
    }

    // Standard OpenMP environment knobs.
    dump_env_kv("OMP_NUM_THREADS");
    dump_env_kv("OMP_PLACES");
    dump_env_kv("OMP_PROC_BIND");
    dump_env_kv("OMP_DYNAMIC");
    dump_env_kv("OMP_MAX_ACTIVE_LEVELS");
    dump_env_kv("OMP_WAIT_POLICY");
    dump_env_kv("OMP_SCHEDULE");
    dump_env_kv("OMP_THREAD_LIMIT");

    // Optional: query runtime API (may trigger initialization in some scenarios).
    std::cout << "  omp_get_max_threads = " << omp_get_max_threads() << "\n";
    std::cout << "  omp_get_num_procs = " << omp_get_num_procs() << "\n";
    std::cout << "  omp_get_dynamic = " << (omp_get_dynamic() ? "true" : "false") << "\n";
    #if _OPENMP >= 200805
    std::cout << "  omp_get_max_active_levels = " << omp_get_max_active_levels() << "\n";
    #endif
    std::cout << "\n" << std::flush;
#endif
}

/**
 * @brief Configures OpenMP placement/binding and determinism settings.
 *
 * @details
 * Thread count selection:
 * - If @p omp_threads > 0, the value is used (clamped to @c INT_MAX).
 * - Otherwise:
 *   - on Linux: the effective CPU count is derived from the current process affinity mask,
 *   - elsewhere: @c std::thread::hardware_concurrency() is used as a fallback.
 *
 * Environment variables (process-global):
 * - @c OMP_NUM_THREADS is always overwritten based on the selected thread count.
 * - Determinism defaults are always overwritten:
 *   - @c OMP_DYNAMIC=FALSE
 *   - @c OMP_MAX_ACTIVE_LEVELS=1
 * - To keep CPU binding controlled by an external affinity step, this function:
 *   - forces @c OMP_PROC_BIND=false,
 *   - clears @c OMP_PLACES,
 *   - clears common runtime-specific affinity variables (@c GOMP_CPU_AFFINITY, @c KMP_*).
 *
 * Runtime API (when @_OPENMP is defined):
 * - disables dynamic adjustment via @c omp_set_dynamic(0),
 * - disables nesting (or limits active levels to 1, depending on OpenMP version),
 * - sets @c omp_set_num_threads(threads),
 * - requests a static schedule when supported.
 *
 * Spin/wait tuning:
 * - When @c threads > 1, additional wait-policy knobs are set to reduce sleep/wake overhead
 *   in tight parallel workloads (best-effort and runtime-dependent).
 *
 * @param omp_threads Requested number of OpenMP threads for tiling (0 means auto).
 * @param verbose If true, prints the effective OpenMP configuration to stdout.
 *
 * @warning
 * This function modifies process-global environment variables and may affect other libraries.
 * If the OpenMP runtime has been initialized before this call, settings may not fully apply.
 */
void configure_openmp_affinity(std::size_t omp_threads, bool verbose) {
    int threads = 1;

    if (omp_threads > 0) {
        threads = (omp_threads > static_cast<std::size_t>(INT_MAX)) ? INT_MAX : static_cast<int>(omp_threads);
    } else {
#if defined(__linux__)
        threads = effective_cpu_count_from_affinity();
#else
        unsigned hc = std::thread::hardware_concurrency();
        threads = hc ? static_cast<int>(hc) : 1;
#endif
    }

    // Deterministic baseline and explicit thread count.
    set_env_overwrite("OMP_NUM_THREADS", std::to_string(threads));
    set_env_overwrite("OMP_DYNAMIC", "FALSE");
    set_env_overwrite("OMP_MAX_ACTIVE_LEVELS", "1");

    // Keep pinning/binding controlled by an external affinity step (not by the OpenMP runtime).
    set_env_overwrite("OMP_PROC_BIND", "false");
    unset_env("OMP_PLACES");
    unset_env("GOMP_CPU_AFFINITY");
    unset_env("KMP_AFFINITY");
    unset_env("KMP_PLACE_THREADS");
    unset_env("KMP_HW_SUBSET");

#if defined(_OPENMP)
    omp_set_dynamic(0);

    #if _OPENMP >= 200805
    omp_set_max_active_levels(1);
    #else
    omp_set_nested(0);
    #endif

    omp_set_num_threads(threads);

    #if _OPENMP >= 200805
    omp_set_schedule(omp_sched_static, 0);
    #endif
#endif // _OPENMP

    if (threads > 1) {
        set_env_overwrite("OMP_WAIT_POLICY", "ACTIVE");
        set_env_overwrite("GOMP_SPINCOUNT", "INFINITE");
        set_env_overwrite("KMP_BLOCKTIME", "100");
    }

    if (verbose) dump_openmp_runtime();
}

} // namespace idet::platform
