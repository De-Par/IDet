#!/bin/bash

# Usage:
#   ./script.sh                # rebuild existing project
#   ./script.sh [ force | f ]  # force mode -> clean build

set -euo pipefail

clean_artefacts() {
    rm -rf ./build
    rm -rf ./subprojects/clipper2
    rm -rf ./subprojects/indicators
    rm -rf ./subprojects/onnxruntime
}

MODE="${1:-}"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd $PARENT_DIR

if [[ "$MODE" == "force" || "$MODE" == "f" ]]; then
    echo "[INFO] Running clean build..."
    clean_artefacts
    meson setup build
else
    echo "[INFO] Running rebuilding..."
    if [ ! -d "./build" ]; then
        echo "[WARNING] Target build directory is not found!"
        echo "[INFO] Running clean build..."
        meson setup build
    else
        meson setup --reconfigure build 
    fi
fi

read -r -p "[INFO] Press ENTER to run compilation or type anything else to exit: " ans || true
if [[ -n "${ans}" ]]; then
    echo "Exit"
    exit 1
fi

echo "[INFO] Running compilation..."
meson compile -C build