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
Generate Doxygen HTML documentation

Usage:
    scripts/generate_doxygen.sh [--clean] [--serve [PORT]] [--check]

Options:
    --clean        Remove the previous docs/doxygen output before generation.
    --serve [P]    Generate docs and serve docs/doxygen/html on localhost.
                   PORT defaults to 8000.
    --check        Generate docs and verify that the main HTML files exist.
    -h, --help     Show this help.

Examples:
    scripts/generate_doxygen.sh --clean --check
    scripts/generate_doxygen.sh --serve 8080
EOF
}

# ------------------------- parse args -------------------------

CLEAN=0
SERVE=0
CHECK=0
PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1; shift
            ;;
        --serve)
            SERVE=1; shift
            if [[ $# -gt 0 && "$1" != --* ]]; then
                PORT="$1"; shift
            fi
            ;;
        --check)
            CHECK=1; shift
            ;;
        -h|--help|help)
            usage; exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

# ------------------------- dependencies -------------------------

DOXYGEN_BIN="${DOXYGEN:-doxygen}"
PYTHON_BIN="${PYTHON:-python3}"

need_cmd "${DOXYGEN_BIN}"

if ! command -v -- dot >/dev/null 2>&1; then
    warn "Graphviz 'dot' was not found. Doxygen will still run, but graph rendering may be incomplete."
fi

# ------------------------- generation -------------------------

OUT_DIR="docs/doxygen"
HTML_DIR="${OUT_DIR}/html"
CONFIG="docs/Doxyfile"

[[ -f "${CONFIG}" ]] || die "Doxygen config not found: ${CONFIG}"

if (( CLEAN == 1 )) && [[ -d "${OUT_DIR}" ]]; then
    rm -rf -- "${OUT_DIR}"
fi

log "[INFO] Doxygen version: $("${DOXYGEN_BIN}" --version)"
log "[INFO] Generating API docs into ${HTML_DIR}"
"${DOXYGEN_BIN}" "${CONFIG}"

if (( CHECK == 1 || SERVE == 1 )); then
    [[ -f "${HTML_DIR}/index.html" ]] || die "Missing generated file: ${HTML_DIR}/index.html"
    [[ -f "${HTML_DIR}/annotated.html" ]] || die "Missing generated file: ${HTML_DIR}/annotated.html"
    [[ -f "${HTML_DIR}/files.html" ]] || die "Missing generated file: ${HTML_DIR}/files.html"
    [[ -f "${HTML_DIR}/topics.html" ]] || die "Missing generated file: ${HTML_DIR}/topics.html"
    log "[INFO] Generated HTML sanity check passed."
fi

if (( SERVE == 1 )); then
    need_cmd "${PYTHON_BIN}"
    log "[INFO] Serving ${HTML_DIR} at http://127.0.0.1:${PORT}/"
    log "[INFO] Press Ctrl+C to stop."
    exec "${PYTHON_BIN}" -m http.server "${PORT}" --bind 127.0.0.1 --directory "${HTML_DIR}"
fi
