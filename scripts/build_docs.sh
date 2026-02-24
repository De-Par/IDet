#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOXY_SCRIPT="${SCRIPT_DIR}/docs/build_doxygen.sh"
JTD_SCRIPT="${SCRIPT_DIR}/docs/build_jtd.sh"

log() { printf '[docs] %s\n' "$*"; }
die() { printf '[docs] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage:
  ./scripts/build_docs.sh [all|doxygen|jtd]

Modes:
  all      Build Doxygen API + Just-the-Docs site (default)
  doxygen  Build only API docs
  jtd      Build only Just-the-Docs site
EOF
}

main() {
    local mode="${1:-all}"
    [[ -x "${DOXY_SCRIPT}" ]] || die "Missing executable script: ${DOXY_SCRIPT}"
    [[ -x "${JTD_SCRIPT}" ]] || die "Missing executable script: ${JTD_SCRIPT}"

    case "${mode}" in
        all)
            "${DOXY_SCRIPT}"
            "${JTD_SCRIPT}"
            ;;
        doxygen)
            "${DOXY_SCRIPT}"
            ;;
        jtd)
            "${JTD_SCRIPT}"
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            die "Unknown mode: ${mode}. Use --help."
            ;;
    esac

    if [[ "${mode}" != "-h" && "${mode}" != "--help" && "${mode}" != "help" ]]; then
        log "Done."
    fi
}

main "$@"
