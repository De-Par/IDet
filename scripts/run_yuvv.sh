#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd -- "${ROOT_DIR}"

die() { printf '%s\n' "[ERROR] $*" >&2; exit 1; }

path_exists() {
    local p="$1"
    [[ -e "$p" ]] || die "Path does not exist: $p"
}

BUILD_DIR="${BUILD_DIR:-build}"
YUV_VIEWER="${BUILD_DIR}/src/app/yuvv/yuv_viewer"
YUV_SRC="${YUV_SRC:-assets/videos/test.yuv}"

# Check path correctness
path_exists "${YUV_VIEWER}"
path_exists "${YUV_SRC}"

"${YUV_VIEWER}" \
    --file "${YUV_SRC}" \
    --w 1920 \
    --h 1080 \
    --fmt i420
    