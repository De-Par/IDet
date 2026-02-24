#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

DOXYFILE="${DOCS_DIR}/config/doxygen/Doxyfile"
DOXY_GROUPS="${DOCS_DIR}/config/doxygen/groups.dox"
API_DIR="${DOCS_DIR}/api"

main() {
    has_cmd doxygen || die "doxygen is not installed. Install it and re-run."
    require_file "${DOXYFILE}"
    require_file "${DOXY_GROUPS}"
    mkdir -p "${API_DIR}"

    log "Building Doxygen API docs -> ${API_DIR}"
    (
        cd "${DOCS_DIR}/config/doxygen"
        doxygen "${DOXYFILE}"
    )
}

main "$@"
