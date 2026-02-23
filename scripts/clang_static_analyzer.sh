#!/usr/bin/env bash

set -euo pipefail

# ------------------------- paths -------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd -- "${ROOT_DIR}"

# ------------------------- logging -------------------------

log()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "[WARN] $*" >&2; }
die()  { printf '%s\n' "[ERROR] $*" >&2; exit 1; }

need_cmd() { command -v -- "$1" >/dev/null 2>&1 || die "Command '$1' not found in PATH"; }

usage() {
    cat <<EOF
Clang Static Analyzer with two modes:
    - soft      Run clang-tidy in a lightweight mode to analyze source code
    - hard      Run scan-build for a more in-depth static analysis and generate HTML reports 

Usage:
    ./scripts/clang_static_analyzer.sh [soft|hard]

Examples:
    source toolchain/scripts/activate.sh 
    ./scripts/clang_static_analyzer.sh soft
    ./scripts/clang_static_analyzer.sh hard
EOF
}

# ------------------------- mode -------------------------

MODE="${1:-soft}"
case "${MODE}" in
    soft|hard) ;;
    -h|--help|help) usage; exit 0 ;;
    *) die "Unknown mode '${MODE}' (use: soft|hard|help)" ;;
esac

# ------------------------- require activated env -------------------------

if [[ "${TC_ACTIVE:-0}" != "1" ]]; then
    die "Toolchain environment is not active. Run: source toolchain/scripts/activate.sh <profile>"
fi

# ------------------------- helpers -------------------------

os_nproc() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu 2>/dev/null || echo 8
    else
        echo 8
    fi
}

mktemp_dir() {
    if mktemp -d >/dev/null 2>&1; then
        mktemp -d
    else
        mktemp -d -t idet_tmp
    fi
}

# ------------------------- resolve env -------------------------

BUILD_DIR="${BUILD_DIR:-build}"
MESON_BIN="${MESON:-meson}"
NINJA_BIN="${NINJA:-ninja}"

CLANG_TIDY_BIN="${CLANG_TIDY:-clang-tidy}"
RUN_TIDY_BIN="${RUN_CLANG_TIDY:-run-clang-tidy}"
SCAN_BUILD_BIN="${SCAN_BUILD:-scan-build}"

JOBS_EFF="${JOBS:-0}"
if [[ -z "${JOBS_EFF}" || "${JOBS_EFF}" == "0" ]]; then
    JOBS_EFF="$(os_nproc)"
fi

need_cmd "${MESON_BIN}"
need_cmd "${CLANG_TIDY_BIN}"
need_cmd "${RUN_TIDY_BIN}"

if [[ "${MODE}" == "hard" ]]; then
    need_cmd "${SCAN_BUILD_BIN}"
    need_cmd "${NINJA_BIN}"
fi

# ------------------------- ensure build dir + compdb -------------------------

ensure_compdb() {
    local bdir="$1"
    if [[ -f "${bdir}/compile_commands.json" ]]; then
        return 0
    fi
    # If build dir exists and has build.ninja, generate compdb.
    if [[ -d "${bdir}" && -f "${bdir}/build.ninja" ]]; then
        log "[INFO] Generating compilation database via ninja: ${bdir}/compile_commands.json"
        "${NINJA_BIN}" -C "${bdir}" -t compdb > "${bdir}/compile_commands.json"
    fi
    [[ -f "${bdir}/compile_commands.json" ]] || die "Missing '${bdir}/compile_commands.json' (build first)"
}

[[ -d "${BUILD_DIR}" ]] || die "Build dir '${BUILD_DIR}' not found (run ./scripts/build.sh first)"
ensure_compdb "${BUILD_DIR}"

# ------------------------- sources list -------------------------

CPP_FILES=()
if command -v git >/dev/null 2>&1 && git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    while IFS= read -r f; do
        [[ -n "$f" ]] && CPP_FILES+=("$f")
    done < <(git -C "${ROOT_DIR}" ls-files -- 'src/**/*.cpp' 'src/**/*.cc' 'src/**/*.cxx' 2>/dev/null || true)
fi

if ((${#CPP_FILES[@]} == 0)); then
    while IFS= read -r f; do
        [[ -n "$f" ]] && CPP_FILES+=("$f")
    done < <(find src -type f \( -name '*.cpp' -o -name '*.cc' -o -name '*.cxx' \) 2>/dev/null | sort || true)
fi

# ------------------------- extra args -------------------------

extra_args=()
# CLANG_INCLUDE_DIR can be ':' separated
if [[ -n "${CLANG_INCLUDE_DIR:-}" ]]; then
    IFS=':' read -r -a _dirs <<< "${CLANG_INCLUDE_DIR}"
    for d in "${_dirs[@]}"; do
        [[ -n "${d}" ]] || continue
        extra_args+=("-extra-arg=-isystem${d}")
    done
fi

# Optional user extra args
if [[ -n "${CSA_EXTRA_ARGS:-}" ]]; then
    extra_args+=( ${CSA_EXTRA_ARGS} )
fi

# ------------------------- print effective -------------------------

print_effective() {
    log "----------------------------------------------"
    log "[CSA] MODE           : ${MODE}"
    log "[CSA] TC_PROFILE     : ${TC_PROFILE:-<none>}"
    log "[CSA] BUILD_DIR      : ${BUILD_DIR}"
    log "[CSA] JOBS           : ${JOBS_EFF}"
    log "[CSA] MESON          : ${MESON_BIN}"
    log "[CSA] CLANG_TIDY     : ${CLANG_TIDY_BIN}"
    log "[CSA] RUN_CLANG_TIDY : ${RUN_TIDY_BIN}"
    if [[ "${MODE}" == "hard" ]]; then
        log "[CSA] SCAN_BUILD     : ${SCAN_BUILD_BIN}"
    fi
    if [[ -n "${CLANG_INCLUDE_DIR:-}" ]]; then
        log "[CSA] INCLUDE_DIRS   : ${CLANG_INCLUDE_DIR}"
    fi
    if [[ -n "${CSA_EXTRA_ARGS:-}" ]]; then
        log "[CSA] EXTRA_ARGS     : ${CSA_EXTRA_ARGS}"
    fi
    log "----------------------------------------------"
}

# ------------------------- run -------------------------

case "${MODE}" in
    soft)
        ((${#CPP_FILES[@]} > 0)) || die "No source files found under src/** (cpp/cc/cxx)"
        print_effective
        log "[INFO] Files: ${#CPP_FILES[@]}"

        "${RUN_TIDY_BIN}" \
            -p "${BUILD_DIR}" \
            -clang-tidy-binary "${CLANG_TIDY_BIN}" \
            -use-color \
            -j "${JOBS_EFF}" \
            "${extra_args[@]-}" \
            "${CPP_FILES[@]}"
        ;;

    hard)
        print_effective

        if [[ "${CSA_CLEAN:-1}" == "1" && -f "${BUILD_DIR}/build.ninja" ]]; then
            log "[INFO] Cleaning Ninja build (CSA_CLEAN=1)"
            "${NINJA_BIN}" -C "${BUILD_DIR}" -t clean || true
        fi

        OUT_BASE="${CSA_OUT_DIR:-${BUILD_DIR}/csa-reports}"
        TS="$(date +%Y%m%d_%H%M%S)"
        OUT_DIR="${OUT_BASE}/scan-build_${TS}"
        mkdir -p "${OUT_DIR}"

        log "[INFO] Output: ${OUT_DIR}"

        "${SCAN_BUILD_BIN}" \
            -analyze-headers \
            -plist-html \
            --status-bugs \
            -o "${OUT_DIR}" \
            "${MESON_BIN}" compile -C "${BUILD_DIR}" -j "${JOBS_EFF}"

        REPORT_HTML="$(find "${OUT_DIR}" -type f -name index.html 2>/dev/null | head -n 1 || true)"
        if [[ -n "${REPORT_HTML}" ]]; then
            log "[INFO] Reports generated: ${REPORT_HTML}"
        else
            log "[INFO] No reports generated: no bugs found ✅"
            log "[INFO] Output dir kept for logs: ${OUT_DIR}"
        fi
        ;;
esac
