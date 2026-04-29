#!/usr/bin/env bash
#
# Toolchain entry point.
#
# Supports being sourced from both **bash** and **zsh**.
#
# When sourced from zsh, the bash implementation is re-invoked inside a bash
# subshell; the resulting exported environment variables, the `idet-*` aliases
# and thin zsh-side wrappers for `tc_list` / `tc_print` / `tc_load` are written
# to a temporary file that is then sourced back into the calling zsh.
#
# Usage:
#   source toolchain/activate.sh [profile]
#   source toolchain/activate.sh -h|--help|help
#
# ---------------------------------------------------------------------------
# zsh delegation shim
# ---------------------------------------------------------------------------
if [ -z "${BASH_VERSION:-}" ] && [ -n "${ZSH_VERSION:-}" ]; then
    # Resolve this script's own path in a zsh-native way. %x expands to the
    # file name of the currently sourced / evaluated file.
    __idet_self="${(%):-%x}"

    # Serialize $@ through a temp file so the inner bash receives the argument
    # list verbatim (handles spaces / quotes / empty args).
    __idet_argfile="$(mktemp -t idet-activate-argv.XXXXXX 2>/dev/null || mktemp)"
    : > "$__idet_argfile"
    for __idet_a in "$@"; do
        printf '%s\n' "$__idet_a" >> "$__idet_argfile"
    done

    # The bash subprocess writes its post-activation snapshot (env vars,
    # aliases, zsh shim functions) to this file.
    __idet_capture="$(mktemp -t idet-activate-capture.XXXXXX 2>/dev/null || mktemp)"

    IDET_ZSH_CAPTURE_FILE="$__idet_capture" \
    IDET_ZSH_SELF="$__idet_self" \
    IDET_ZSH_ARGFILE="$__idet_argfile" \
        bash -c '
            set --
            while IFS= read -r __a; do
                set -- "$@" "$__a"
            done < "${IDET_ZSH_ARGFILE}"
            unset IDET_ZSH_ARGFILE
            # shellcheck source=/dev/null
            source "${IDET_ZSH_SELF}" "$@"
        '
    __idet_rc=$?

    if [ "$__idet_rc" = 0 ] && [ -s "$__idet_capture" ]; then
        # shellcheck source=/dev/null
        source "$__idet_capture"
    fi

    rm -f -- "$__idet_capture" "$__idet_argfile"

    # Propagate the rc of the inner bash back to the zsh caller.
    __idet_final_rc=$__idet_rc
    unset __idet_self __idet_capture __idet_argfile __idet_a __idet_rc

    return $__idet_final_rc
fi

# ---------------------------------------------------------------------------
# bash implementation
# ---------------------------------------------------------------------------

_act_err() { printf '%s\n' "[ERROR] $*" >&2; }

if [ -z "${BASH_VERSION:-}" ]; then
    _act_err "activate.sh: unsupported shell (expected bash or zsh)"
    return 1 2>/dev/null || exit 1
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    _act_err "Must be sourced: source toolchain/activate.sh [profile]"
    return 1 2>/dev/null || exit 1
fi

_act_usage() {
    cat <<'EOF'
Usage:
    source toolchain/activate.sh [profile]
    source toolchain/activate.sh [-h|--help|help]

Description:
    Activates an IDet toolchain profile in the current shell by exporting
    environment variables (BUILD_DIR, MESON_NATIVE_FILE, tool binaries, etc.)
    loaded via toolchain/tc.sh.

Behavior:
    - If [profile] is omitted, uses TC_PROFILE from:
        toolchain/env/local.env (if exists) -> toolchain/env/defaults.env
    - On unknown profile, prints an error and available profiles, and does not
        modify the current environment.

Examples:
    source toolchain/activate.sh gcc-perf
    source toolchain/activate.sh            # uses default TC_PROFILE
    source toolchain/activate.sh -h

Supported shells:
    bash, zsh. Under zsh the bash implementation is re-invoked in a subshell
    and its exports / aliases / tc_* functions are imported back into zsh.

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
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd -- "${ROOT_DIR}"

TC_FILE="${ROOT_DIR}/toolchain/tc.sh"

# shellcheck source=/dev/null
source "${TC_FILE}" || {
    _act_err "Failed to source ${TC_FILE}"
    return 1 2>/dev/null || exit 1
}

# If arg missing, tc_load will use TC_PROFILE from defaults/local
tc_load "${1:-}" || return 1

_tc_cleanup_internals "_act_*"

export TC_ACTIVE=1

# build
alias idet-build="${ROOT_DIR}/scripts/build.sh"

# run targets
alias idet-test="${ROOT_DIR}/scripts/run_tests.sh"
alias idet-face="${ROOT_DIR}/scripts/run_idet_face.sh"
alias idet-text="${ROOT_DIR}/scripts/run_idet_text.sh"
alias idet-yuvv="${ROOT_DIR}/scripts/run_yuvv.sh"

# run sanitizers
alias idet-csa="${ROOT_DIR}/scripts/clang_static_analyzer.sh"
alias idet-inc-clean="${ROOT_DIR}/scripts/include_cleaner.sh"
alias idet-fmt="${ROOT_DIR}/scripts/format_code.sh"

tc_print

# ---------------------------------------------------------------------------
# zsh delegation: emit a sourceable snapshot (env vars + aliases + shim fns)
# for the outer zsh caller. No-op when sourced directly from bash.
# ---------------------------------------------------------------------------
if [[ -n "${IDET_ZSH_CAPTURE_FILE:-}" ]]; then
    # Exported environment. This list mirrors the `managed=` array in tc.sh's
    # `tc_load` plus a few CC/CXX shadows set by `_tc_apply_ort_toolchain_env`.
    _act_zsh_vars=(
        TC_ACTIVE TC_PROFILE TC_ROOT_DIR TC_PROFILE_DIR TC_PROFILE_INI
        TC_PROFILE_DESC TC_APP_REL
        BUILD_DIR MESON MESON_OPT_FILE MESON_NATIVE_FILE NINJA PKG_CONFIG JOBS
        LLVM_VER CLANG CLANGXX CLANG_FORMAT CLANG_TIDY RUN_CLANG_TIDY SCAN_BUILD
        LLVM_AR LLVM_STRIP LLVM_RANLIB CLANG_INCLUDE_DIR
        GCC_VER GCC GXX GCC_AR GCC_STRIP GCC_RANLIB GCC_INCLUDE_DIR
        NONINTERACTIVE KEEP_SUBPROJECTS ORT_TOOLCHAIN_FAMILY ORT_CACHE_ROOT
        TARGET_TRIPLE SYSROOT
        CC CXX AR RANLIB STRIP
    )

    {
        _v=""
        for _v in "${_act_zsh_vars[@]}"; do
            if [[ -n "${!_v+x}" ]]; then
                printf 'export %s=%q\n' "$_v" "${!_v}"
            fi
        done

        # idet-* aliases. `alias NAME` (no `=`) prints a portable
        # `alias NAME='value'` line that both bash and zsh can source.
        alias idet-build idet-test idet-face idet-text idet-yuvv \
              idet-csa idet-inc-clean idet-fmt 2>/dev/null || true

        # zsh-side wrappers for the interactive tc_* entry points. They
        # re-invoke the bash implementation each time, so zsh users still get
        # the banner / profile list without having to port the bash body.
        cat <<'__IDET_ZSH_SHIMS__'
tc_list() {
    # shellcheck disable=SC2016
    bash -c 'source "${TC_ROOT_DIR}/toolchain/tc.sh" >/dev/null 2>&1 && tc_list'
}
tc_print() {
    # shellcheck disable=SC2016
    bash -c 'source "${TC_ROOT_DIR}/toolchain/tc.sh" >/dev/null 2>&1 && tc_print'
}
tc_load() {
    local __cap __rc
    __cap="$(mktemp -t idet-activate-tc-load.XXXXXX 2>/dev/null || mktemp)"
    IDET_ZSH_CAPTURE_FILE="$__cap" bash -c '
        # shellcheck source=/dev/null
        source "${TC_ROOT_DIR}/toolchain/activate.sh" "$@"
    ' _ "$@"
    __rc=$?
    if [ "$__rc" = 0 ] && [ -s "$__cap" ]; then
        # shellcheck source=/dev/null
        source "$__cap"
    fi
    rm -f -- "$__cap"
    return $__rc
}
__IDET_ZSH_SHIMS__
    } > "${IDET_ZSH_CAPTURE_FILE}"

    unset _v _act_zsh_vars
fi
