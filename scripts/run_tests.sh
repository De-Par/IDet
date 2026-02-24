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
need_cmd() { command -v -- "$1" >/dev/null 2>&1 || die "Required command '$1' not found in PATH"; }

usage() {
    cat <<'EOF'
Run tests (Meson)

Usage:
    ./scripts/run_tests.sh [--no-rebuild] [--suite NAME] [--repeat N] [--gdb] [--] [extra meson test args...]

Examples:
    source toolchain/activate.sh 
    ./scripts/run_tests.sh
    ./scripts/run_tests.sh --suite unit
EOF
}

# ------------------------- parse args -------------------------

NO_REBUILD=0
SUITE=""
REPEAT=""
USE_GDB=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-rebuild)
            NO_REBUILD=1; shift
            ;;
        --suite)
            shift; [[ $# -gt 0 ]] || die "--suite requires argument"
            SUITE="$1"; shift
            ;;
        --repeat)
            shift; [[ $# -gt 0 ]] || die "--repeat requires argument"
            REPEAT="$1"; shift
            ;;
        --gdb)
            USE_GDB=1; shift
            ;;
        -h|--help|help)
            usage; exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                EXTRA_ARGS+=("$1"); shift
            done
            ;;
        *)
            # allow passing meson args without '--' too
            EXTRA_ARGS+=("$1"); shift
            ;;
    esac
done

# ------------------------- env (from activate.sh) -------------------------

if [[ "${TC_ACTIVE:-0}" != "1" ]]; then
    warn "No active toolchain detected (TC_ACTIVE!=1)."
    warn "Recommended: source toolchain/activate.sh <profile>"
fi

BUILD_DIR="${BUILD_DIR:-build}"
MESON_BIN="${MESON:-meson}"
JOBS="${JOBS:-0}"

need_cmd "${MESON_BIN}"

if [[ ! -d "${BUILD_DIR}" ]]; then
    die "Build directory '${BUILD_DIR}' not found! Run: ./scripts/build.sh setup (after activating toolchain)."
fi

# Portable CPU count
os_nproc() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu 2>/dev/null || echo 8
    else
        echo 8
    fi
}

# ------------------------- meson test args -------------------------

args=(test -C "${BUILD_DIR}" --print-errorlogs -v)

# Rebuild control
if (( NO_REBUILD == 1 )); then
    args+=(--no-rebuild)
fi

# Suite filter
if [[ -n "${SUITE}" ]]; then
    args+=(--suite "${SUITE}")
fi

# Repeat
if [[ -n "${REPEAT}" ]]; then
    args+=(--repeat "${REPEAT}")
fi

# GDB
if (( USE_GDB == 1 )); then
    args+=(--gdb)
fi

# Parallelism:
# meson test uses --num-processes (NOT -j)
nprocs="${JOBS:-0}"
if [[ -z "${nprocs}" || "${nprocs}" == "0" ]]; then
    nprocs="$(os_nproc)"
fi
args+=(--num-processes "${nprocs}")

# Extra user args
if ((${#EXTRA_ARGS[@]} > 0)); then
    args+=("${EXTRA_ARGS[@]}")
fi

log "[INFO] Running: ${MESON_BIN} ${args[*]}"
exec "${MESON_BIN}" "${args[@]}"
