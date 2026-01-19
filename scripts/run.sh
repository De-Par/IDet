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
MODEL=./assets/models/scrfd/scrfd_500m_bnkps.onnx
IMG=./assets/images/face.png

# Check path correctness
path_exists $APP
path_exists $MODEL
path_exists $IMG

if [ "$MODE" == "tile" ]; then
    $APP \
        --mode face \
        --model $MODEL \
        --image $IMG \
        --min_text_size 12 --unclip 1.4 \
        --bin_thresh 0.3 \
        --box_thresh 0.6 \
        --nms_iou 0.4 \
        --bind_io 1 \
        --verbose 0 --is_draw 1 \
        --fixed_wh 480x256 \
        --threads_intra 1 --threads_inter 1 --tile_omp 4 \
        --bench 100 --warmup 20 \
        --tiles 2x2 --tile_overlap 0.05
else
    $APP \
        --mode face \
        --image $IMG \
        --min_text_size 10 --unclip 1.1 \
        --bin_thresh 0.3 \
        --box_thresh 0.6 \
        --nms_iou 0.4 \
        --bind_io 0 \
        --verbose 1 --is_draw 1 \
        --side 320 \
        --threads_intra 10 --threads_inter 1 --tile_omp 1 \
        --bench 200 --warmup 100
fi
