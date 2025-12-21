#!/bin/bash

set -euo pipefail

mode="${1:-}"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd $PARENT_DIR

APP=./build/src/app/text_det

if ! ls $APP 1> /dev/null 2>&1; then
    echo "Executable do not exists. Stop"
    exit 1    
fi

MODEL=./models/paddleocr/ch_ppocr_v2_det.onnx
IMG=./images/paper.png

if [ "$mode" == "tile" ]; then
    $APP \
        --model $MODEL \
        --image $IMG \
        --min_text_size 10 --unclip 1.5 \
        --bin_thresh 0.3 \
        --box_thresh 0.3 \
        --nms_iou 0.5 \
        --bind_io 1 \
        --verbose 0 --is_draw 1 \
        --fixed_wh 160x128 \
        --threads_intra 1 --threads_inter 1 --tile_omp 24 \
        --bench 100 --warmup 20 \
        --tiles 6x4 --tile_overlap 0.1
else
    $APP \
        --model $MODEL \
        --image $IMG \
        --min_text_size 10 --unclip 1.0 \
        --bin_thresh 0.3 \
        --box_thresh 0.3 \
        --nms_iou 0.5 \
        --bind_io 1 \
        --verbose 0 --is_draw 1 \
        --fixed_wh 960x512 \
        --threads_intra 24 --threads_inter 1 --tile_omp 1 \
        --bench 100 --warmup 20
fi