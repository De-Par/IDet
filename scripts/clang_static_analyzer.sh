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
    cat <<'EOF'
Clang Static Analyzer with two modes:
    - soft      Run run-clang-tidy over first-party code (src/include only)
    - hard      Run scan-build (HTML report)

Usage:
    ./scripts/clang_static_analyzer.sh [soft|hard]

Env knobs:
    BUILD_DIR (default: build)
    JOBS (0/empty -> auto)
    CLANG_TIDY, RUN_CLANG_TIDY, SCAN_BUILD, MESON, NINJA

    CSA_HEADER_FILTER (default: ^(.*/)?(src|include)/)
    CSA_LINE_FILTER   (default: [{"name":".*/src/.*"},{"name":".*/include/.*"}])

    CSA_EXTRA_ARGS    (extra args forwarded to run-clang-tidy)
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
    die "Toolchain environment is not active. Run: source toolchain/activate.sh <profile>"
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

# Remove any diagnostics lines that reference subprojects/
# (also removes "included from .../subprojects/..." cascades)
filter_out_subprojects() {
    # match both /subprojects/ and \subprojects\ (in case of weird paths)
    grep -vE '(^|[[:space:]])([^[:space:]]*/)?subprojects(/|\\)' || true
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
need_cmd "${NINJA_BIN}"
need_cmd "${CLANG_TIDY_BIN}"
need_cmd "${RUN_TIDY_BIN}"

if [[ "${MODE}" == "hard" ]]; then
    need_cmd "${SCAN_BUILD_BIN}"
fi

# ------------------------- ensure build dir + compdb -------------------------

ensure_compdb() {
    local bdir="$1"
    [[ -d "${bdir}" ]] || die "Build dir '${bdir}' not found (run ./scripts/build.sh first)"

    if [[ -f "${bdir}/compile_commands.json" ]]; then
        return 0
    fi

    if [[ -f "${bdir}/build.ninja" ]]; then
        log "[INFO] Generating compilation database via ninja: ${bdir}/compile_commands.json"
        "${NINJA_BIN}" -C "${bdir}" -t compdb > "${bdir}/compile_commands.json"
    fi

    [[ -f "${bdir}/compile_commands.json" ]] || die "Missing '${bdir}/compile_commands.json' (build first)"
}

ensure_compdb "${BUILD_DIR}"

# ------------------------- sources list (first-party only) -------------------------

declare -a CPP_FILES=()

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

# ------------------------- filters (src/include only) -------------------------

HEADER_FILTER="${CSA_HEADER_FILTER:-^(.*/)?(src|include)/}"
LINE_FILTER="${CSA_LINE_FILTER:-}"
if [[ -z "${LINE_FILTER}" ]]; then
    LINE_FILTER='[{"name":".*/src/.*"},{"name":".*/include/.*"}]'
fi

# ------------------------- extra args -------------------------

declare -a extra_args=()

# Optional user extra args (space-separated tokens)
if [[ -n "${CSA_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args+=( ${CSA_EXTRA_ARGS} )
fi

# ------------------------- print effective -------------------------

print_effective() {
    log "----------------------------------------------"
    log "[CSA] MODE          : ${MODE}"
    log "[CSA] BUILD_DIR     : ${BUILD_DIR}"
    log "[CSA] JOBS          : ${JOBS_EFF}"
    log "[CSA] MESON         : ${MESON_BIN}"
    log "[CSA] NINJA         : ${NINJA_BIN}"
    log "[CSA] CLANG_TIDY    : ${CLANG_TIDY_BIN}"
    log "[CSA] RUN_CLANG_TIDY: ${RUN_TIDY_BIN}"
    if [[ "${MODE}" == "hard" ]]; then
        log "[CSA] SCAN_BUILD    : ${SCAN_BUILD_BIN}"
    fi
    log "[CSA] HEADER_FILTER : ${HEADER_FILTER}"
    log "[CSA] LINE_FILTER   : ${LINE_FILTER}"
    if [[ -n "${CSA_EXTRA_ARGS:-}" ]]; then
        log "[CSA] EXTRA_ARGS    : ${CSA_EXTRA_ARGS}"
    fi
    log "----------------------------------------------"
}

# ------------------------- run -------------------------

case "${MODE}" in
    soft)
        ((${#CPP_FILES[@]} > 0)) || die "No source files found under src/** (cpp/cc/cxx)"
        print_effective
        log "[INFO] Files: ${#CPP_FILES[@]}"

        # Стримим вывод, вырезаем subprojects/, и считаем "error:" только в first-party выводе
        set +e
        "${RUN_TIDY_BIN}" \
            -p "${BUILD_DIR}" \
            -clang-tidy-binary "${CLANG_TIDY_BIN}" \
            -use-color \
            -j "${JOBS_EFF}" \
            -header-filter="${HEADER_FILTER}" \
            -line-filter="${LINE_FILTER}" \
            "${extra_args[@]-}" \
            "${CPP_FILES[@]}" \
            2>&1 | awk '
                # полностью игнорируем любые строки, где фигурирует subprojects/
                /(^|[[:space:]])([^[:space:]]*\/)?subprojects\// { next }

                { print }

                # если в first-party выводе встретили error: -> отметим ошибку
                /(^|[[:space:]])error:/ { err=1 }

                END { exit (err ? 1 : 0) }
            '
        awk_rc=$?
        run_rc=${PIPESTATUS[0]}
        set -e

        if [[ "${awk_rc}" -ne 0 ]]; then
            log "[INFO] clang-tidy: ❌ errors found"
            exit 1
        fi

        if [[ "${run_rc}" -ne 0 ]]; then
            warn "run-clang-tidy exited with rc=${run_rc}, but no first-party errors after subprojects filter"
        fi

        log "[INFO] clang-tidy: ✅ no errors"
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
