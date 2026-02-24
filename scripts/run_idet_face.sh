#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${ROOT_DIR}"

log() { printf '%s\n' "$*"; }
die() { printf '%s\n' "[ERROR] $*" >&2; exit 1; }

require_tc_active() {
    if [[ "${TC_ACTIVE:-0}" != "1" ]]; then
        die "Toolchain env is not active. Run: source toolchain/activate.sh <profile>"
    fi
    [[ -n "${TC_PROFILE:-}" ]] || die "TC_PROFILE is empty (activate.sh didn't export it?)"
    [[ -n "${BUILD_DIR:-}" ]] || die "BUILD_DIR is empty (activate.sh didn't export it?)"
    [[ -n "${TC_APP_REL:-}" ]] || die "TC_APP_REL is empty (activate.sh didn't export it?)"
}

path_exists() {
    local p="$1"
    [[ -e "$p" ]] || die "Path does not exist: $p"
}

usage() {
    cat <<EOF
Run face detection (requires activated toolchain env)

Usage:
    ./scripts/run_idet_face.sh [tile|t|single|s] [--] [extra idet_app args...]

Example:
    source toolchain/activate.sh
    ./scripts/run_idet_face.sh
    ./scripts/run_idet_face.sh tile
EOF
}

MODE="single"
APP_EXTRA_ARGS=()

# First positional arg may be mode
if [[ $# -gt 0 && "${1:-}" != -* ]]; then
    case "$1" in
        tile|t) MODE="tile"; shift ;;
        single|s) MODE="single"; shift ;;
        -h|--help|help) usage; exit 0 ;;
        *) : ;;
    esac
fi

# Pass-through args after --
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help|help) usage; exit 0 ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do APP_EXTRA_ARGS+=("$1"); shift; done
            ;;
        *)
            APP_EXTRA_ARGS+=("$1"); shift
            ;;
    esac
done

require_tc_active

APP="${BUILD_DIR}/${TC_APP_REL}"
MODEL="${MODEL:-assets/models/scrfd/scrfd_500m_bnkps.onnx}"
IMG="${IMG:-assets/images/face/medium.png}"

path_exists "${APP}"
path_exists "${MODEL}"
path_exists "${IMG}"

log "----------------------------------------------"
log "[INFO] Run face (${MODE})"
log "[INFO] profile    : ${TC_PROFILE}"
log "[INFO] build dir : ${BUILD_DIR}"
log "[INFO] app       : ${APP}"
log "[INFO] model     : ${MODEL}"
log "[INFO] image     : ${IMG}"
if ((${#APP_EXTRA_ARGS[@]} > 0)); then
    printf '[INFO] extra     : ' ; printf '%q ' "${APP_EXTRA_ARGS[@]}" ; echo
fi
log "----------------------------------------------"

common_args=(
    --mode face
    --model "${MODEL}"
    --image "${IMG}"
    --bin_thresh 0.3
    --box_thresh 0.5
    --unclip 1.0
    --nms_iou 0.3
    --use_fast_iou 1
    --min_roi_size_w 10
    --min_roi_size_h 10
    --bind_io 1
    --runtime_policy 1
    --soft_mem_bind 1
    --suppress_opencv 1
    --bench_iters 100
    --warmup_iters 20
    --is_draw 1
    --is_dump 0
    --verbose 1
)

tile_args=(
    --tiles_rc 3x3
    --fixed_hw 160x256
    --tile_overlap 0.1
    --threads_intra 1
    --threads_inter 1
    --tile_omp 9
)

single_args=(
    --fixed_hw 512x768
    --max_img_size 768
    --threads_intra 9
    --threads_inter 1
    --tile_omp 1
)

args=( "${common_args[@]}" )
if [[ "${MODE}" == "tile" ]]; then
    args+=( "${tile_args[@]}" )
else
    args+=( "${single_args[@]}" )
fi
if ((${#APP_EXTRA_ARGS[@]} > 0)); then
    args+=( "${APP_EXTRA_ARGS[@]}" )
fi

# Dump full command
printf '[CMD] '
printf '%q ' "${APP}" "${args[@]}"
echo

# Run final command
exec "${APP}" "${args[@]}"
