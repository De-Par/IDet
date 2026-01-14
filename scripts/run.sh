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

MODE="${1:-}"
APP=./build/src/app/tdet/text_det
MODEL=./assets/models/paddleocr/ch_ppocr_v4_det.onnx
IMG=./assets/images/test.png

# Check path correctness
path_exists $APP
path_exists $MODEL
path_exists $IMG

if [ "$MODE" == "tile" ]; then
    $APP \
        --model $MODEL \
        --image $IMG \
        --min_text_size 12 --unclip 1.4 \
        --bin_thresh 0.3 \
        --box_thresh 0.6 \
        --nms_iou 0.3 \
        --bind_io 1 \
        --verbose 1 --is_draw 1 \
        --fixed_wh 256x256 \
        --threads_intra 1 --threads_inter 1 --tile_omp 4 \
        --bench 100 --warmup 20 \
        --tiles 2x2 --tile_overlap 0.05
else
    $APP \
        --model $MODEL \
        --image $IMG \
        --min_text_size 10 --unclip 1.1 \
        --bin_thresh 0.3 \
        --box_thresh 0.6 \
        --nms_iou 0.5 \
        --bind_io 1 \
        --verbose 1 --is_draw 1 \
        --fixed_wh 512x512 \
        --threads_intra 10 --threads_inter 1 --tile_omp 1 \
        --bench 100 --warmup 20
fi