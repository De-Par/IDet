#!/bin/bash

set -euo pipefail

path_exists() {
    if ! ls $1 1> /dev/null 2>&1; then
        echo "[ERROR] Path '$1' does not exist. Stop"
        exit 1    
    fi
}

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd $PARENT_DIR

YUV_VIEWER=./build/src/app/yuvv/yuv_viewer
YUV_SRC=./assets/videos/test.yuv

# Check path correctness
path_exists $YUV_VIEWER
path_exists $YUV_SRC

$YUV_VIEWER \
    --file $YUV_SRC \
    --w 1920 \
    --h 1080 \
    --fmt i420