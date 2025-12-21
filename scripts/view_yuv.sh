#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd $PARENT_DIR

YUV_VIEWER=./build/src/app/yuv_viewer

if ! ls $YUV_VIEWER 1> /dev/null 2>&1; then
    echo "Executable do not exists. Stop"
    exit 1    
fi

YUV_SRC=./videos/test.yuv

if ! ls $YUV_SRC 1> /dev/null 2>&1; then
    echo "YUV file do not exists. Stop"
    exit 1    
fi

$YUV_VIEWER \
    --file $YUV_SRC \
    --w 1920 \
    --h 1080 \
    --fmt i420