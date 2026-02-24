#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCS_DIR="${ROOT_DIR}/docs"

log() { printf '[docs] %s\n' "$*"; }
die() { printf '[docs] ERROR: %s\n' "$*" >&2; exit 1; }
has_cmd() { command -v "$1" >/dev/null 2>&1; }

require_file() {
    local f="$1"
    [[ -f "${f}" ]] || die "Required file not found: ${f}"
}

ruby_major_version() {
    ruby -e 'puts(RUBY_VERSION.split(".").first.to_i)'
}

try_activate_brew_ruby() {
    if ! has_cmd ruby || [[ "$(ruby_major_version)" -ge 3 ]] || ! has_cmd brew; then
        return
    fi
    local brew_ruby_prefix=""
    brew_ruby_prefix="$(brew --prefix ruby 2>/dev/null || true)"
    if [[ -n "${brew_ruby_prefix}" && -x "${brew_ruby_prefix}/bin/ruby" ]]; then
        export PATH="${brew_ruby_prefix}/bin:${PATH}"
        hash -r
        log "Using Homebrew Ruby: $("${brew_ruby_prefix}/bin/ruby" -v | awk '{print $2}')"
    fi
}
