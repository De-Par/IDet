#!/usr/bin/env bash

_act_err() { printf '%s\n' "[ERROR] $*" >&2; }

if [[ -z "${BASH_VERSION:-}" ]]; then
    _act_err "activate.sh must be sourced in bash (not zsh). Run: bash, then source this script"
    return 1 2>/dev/null || exit 1
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    _act_err "Must be sourced: source toolchain/scripts/activate.sh [profile]"
    return 1 2>/dev/null || exit 1
fi

_act_usage() {
    cat <<'EOF'
Usage:
    source toolchain/scripts/activate.sh [profile]
    source toolchain/scripts/activate.sh [-h|--help|help]

Description:
    Activates an IDet toolchain profile in the current shell by exporting
    environment variables (BUILD_DIR, MESON_NATIVE_FILE, tool binaries, etc.)
    loaded via toolchain/scripts/tc.sh.

Behavior:
    - If [profile] is omitted, uses TC_PROFILE from:
        toolchain/env/local.env (if exists) -> toolchain/env/defaults.env
    - On unknown profile, prints an error and available profiles, and does not
        modify the current environment.

Examples:
    source toolchain/scripts/activate.sh gcc-perf
    source toolchain/scripts/activate.sh            # uses default TC_PROFILE
    source toolchain/scripts/activate.sh -h

Tip:
    After activation you can run build/test tools without passing profile,
    because the environment is already configured.
EOF
}

case "${1:-}" in
    -h|--help|help) _act_usage; return 0 ;;
    *) : ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd -- "${ROOT_DIR}"

TC_FILE="${ROOT_DIR}/toolchain/scripts/tc.sh"

source "${TC_FILE}" || {
    _act_err "Failed to source ${TC_FILE}"
    return 1 2>/dev/null || exit 1
}

# If arg missing, tc_load will use TC_PROFILE from defaults/local
tc_load "${1:-}" || return 1

_tc_cleanup_internals "_act_*"

export TC_ACTIVE=1

alias idet-build="${ROOT_DIR}/scripts/build.sh"
alias idet-test="${ROOT_DIR}/scripts/run_tests.sh"
alias idet-face="${ROOT_DIR}/scripts/run_idet_face.sh"
alias idet-text="${ROOT_DIR}/scripts/run_idet_text.sh"
alias idet-csa="${ROOT_DIR}/scripts/clang_static_analyzer.sh"
alias idet-inc-clean="${ROOT_DIR}/scripts/include_cleaner.sh"
alias idet-fmt="${ROOT_DIR}/scripts/format_code.sh"

tc_print
