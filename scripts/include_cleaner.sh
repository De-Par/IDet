#!/usr/bin/env bash

set -euo pipefail

# ------------------------- paths -------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${ROOT_DIR}"

# ------------------------- utils -------------------------

log()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "[WARN] $*" >&2; }
die()  { printf '%s\n' "[ERROR] $*" >&2; exit 1; }

bool_str() {
    if [[ "$1" == "1" ]]; then
        printf 'true'
    else
        printf 'false'
    fi
}

need_cmd() { command -v -- "$1" >/dev/null 2>&1 || die "Command '$1' not found in PATH"; }

usage() {
    cat <<EOF
Include directives cleaner util

Usage:
    scripts/include_cleaner.sh [--fix] [--misinc] [--] [paths...]

Options:
    --fix        apply fixes in-place
    --misinc     enable MissingIncludes mode 
    -h,--help    show this help

Examples:
    source toolchain/activate.sh
    scripts/include_cleaner.sh --fix -- src/app
EOF
}

# ------------------------- defaults -------------------------

DO_FIX=0
DO_MISSING=0
TARGET_PATHS=()

# ------------------------- CLI parsing -------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fix) DO_FIX=1; shift ;;
        --misinc) DO_MISSING=1; shift ;;
        -h|--help|help) usage; exit 0 ;;
        --) shift; TARGET_PATHS+=("$@"); break ;;
        *) TARGET_PATHS+=("$1"); shift ;;
    esac
done

# Default target paths
if [[ ${#TARGET_PATHS[@]} -eq 0 ]]; then
    TARGET_PATHS=("src/lib" "src/app")
    warn "Target paths not set, using defaults: ${TARGET_PATHS[*]}"
fi

# ------------------------- load environment -------------------------

TC_SH="${ROOT_DIR}/toolchain/tc.sh"
[[ -f "${TC_SH}" ]] || die "Missing toolchain loader: toolchain/tc.sh"
source "${TC_SH}"

# Load environment from the currently active profile (if available)
if [[ "${TC_ACTIVE:-0}" != "1" ]]; then
    die "Toolchain environment is not active. Please run: source toolchain/activate.sh <profile>"
fi

# Apply CLI overrides
BUILD_DIR="${BUILD_DIR:-build}"

# Tools (user-controlled)
need_cmd "${CLANG_TIDY}"
need_cmd "${RUN_CLANG_TIDY}"

# ------------------------- compdb -------------------------

ensure_compdb() {
    if [[ -f "${BUILD_DIR}/compile_commands.json" ]]; then
        return 0
    fi
    if command -v -- "${NINJA}" >/dev/null 2>&1 && [[ -f "${BUILD_DIR}/build.ninja" ]]; then
        log "[INFO] Generating compilation database via ninja: ${BUILD_DIR}/compile_commands.json"
        "${NINJA}" -C "${BUILD_DIR}" -t compdb > "${BUILD_DIR}/compile_commands.json"
    fi
    [[ -f "${BUILD_DIR}/compile_commands.json" ]] || die "Missing '${BUILD_DIR}/compile_commands.json'"
}

ensure_compdb "${BUILD_DIR}"

# ------------------------- collect source files -------------------------

collect_tidy_files() {
    local target

    for target in "${TARGET_PATHS[@]}"; do
        if [[ -d "${target}" ]]; then
            find "${target}" -type f \( \
                -name "*.c" -o \
                -name "*.cc" -o \
                -name "*.cpp" -o \
                -name "*.cxx" -o \
                -name "*.c++" -o \
                -name "*.m" -o \
                -name "*.mm" \
            \) -print0
        elif [[ -f "${target}" ]]; then
            case "${target}" in
                *.c|*.cc|*.cpp|*.cxx|*.c++|*.m|*.mm)
                    printf '%s\0' "${target}"
                    ;;
                *)
                    warn "Skipping non-source file: ${target}"
                    ;;
            esac
        else
            warn "Target path does not exist: ${target}"
        fi
    done
}

mapfile -d '' -t TIDY_FILES < <(collect_tidy_files)

if [[ ${#TIDY_FILES[@]} -eq 0 ]]; then
    die "No source files matched target paths: ${TARGET_PATHS[*]}"
fi

# ------------------------- clang-tidy arguments -------------------------

missing_val="false"
(( DO_MISSING == 1 )) && missing_val="true"

ARGS=()
ARGS+=("-config={CheckOptions: [{key: misc-include-cleaner.MissingIncludes, value: '${missing_val}'}]}")
(( DO_FIX == 1 )) && ARGS+=("-fix")

EXTRA=()
if [[ -n "${CLANG_INCLUDE_DIR:-}" ]]; then
    EXTRA+=("-extra-arg=-isystem${CLANG_INCLUDE_DIR}")
fi

fix_enabled="$(bool_str "${DO_FIX}")"
missing_includes_enabled="$(bool_str "${DO_MISSING}")"

log "[INFO] include-cleaner via clang-tidy"
log "----------------------------------------------"
log "[INFO] Build dir         : ${BUILD_DIR}"
log "[INFO] clang-tidy        : ${CLANG_TIDY}"
log "[INFO] run-clang-tidy    : ${RUN_CLANG_TIDY}"
log "[INFO] Fix enabled       : ${fix_enabled}"
log "[INFO] Missing includes  : ${missing_includes_enabled}"
log "[INFO] Matched TUs       : ${#TIDY_FILES[@]}"
log "[INFO] Target paths      : ${TARGET_PATHS[*]}"

if [[ -n "${CLANG_INCLUDE_DIR:-}" ]]; then
    log "[INFO] Extra include dir : ${CLANG_INCLUDE_DIR}"
fi

log "----------------------------------------------"

"${RUN_CLANG_TIDY}" \
    -p "${BUILD_DIR}" \
    -clang-tidy-binary "${CLANG_TIDY}" \
    -checks='-*,misc-include-cleaner' \
    -header-filter='^(.*/)?(src|include)/' \
    "${EXTRA[@]}" \
    "${ARGS[@]}" \
    "${TIDY_FILES[@]}"

log "✅ Done"
