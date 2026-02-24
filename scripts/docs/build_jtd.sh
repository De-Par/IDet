#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

JTD_CONFIG="${DOCS_DIR}/config/jtd/_config.yml"
JTD_GEMFILE="${DOCS_DIR}/config/jtd/Gemfile"
JTD_INDEX="${DOCS_DIR}/jtd/index.md"
JTD_SOURCE="${DOCS_DIR}/jtd"
SITE_DIR="${DOCS_DIR}/site"

main() {
    try_activate_brew_ruby
    has_cmd ruby || die "ruby is not installed. Install Ruby and re-run."
    if [[ "$(ruby_major_version)" -lt 3 ]]; then
        die "Ruby >= 3.0 is required for JTD gems. Install Homebrew ruby and add it to PATH."
    fi
    has_cmd bundle || die "bundler is not installed. Run: gem install bundler"

    require_file "${JTD_CONFIG}"
    require_file "${JTD_GEMFILE}"
    require_file "${JTD_INDEX}"
    mkdir -p "${SITE_DIR}"

    log "Installing Jekyll dependencies (bundle install)"
    (
        cd "${DOCS_DIR}/config/jtd"
        bundle config set --local path "${DOCS_DIR}/vendor/bundle" >/dev/null
        bundle install --quiet
    )

    log "Building Just-the-Docs site -> ${SITE_DIR}"
    (
        cd "${DOCS_DIR}/config/jtd"
        bundle exec jekyll build --source "${JTD_SOURCE}" --destination "${SITE_DIR}" --config "${JTD_CONFIG}" --quiet
    )
}

main "$@"
