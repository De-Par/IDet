#!/usr/bin/env bash

# Must be sourced in bash. When sourced, $0 is the parent shell; BASH_SOURCE[0] is this file
if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "[TC][ERROR] This toolchain must be sourced in bash (not zsh). Run: bash, then source this script." >&2
    return 1 2>/dev/null || exit 1
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[TC][ERROR] tc.sh must be sourced: source toolchain/scripts/tc.sh" >&2
    exit 1
fi

_tc_is_tty() { [[ -t 2 ]]; }

if _tc_is_tty && [[ -z "${NO_COLOR:-}" ]]; then
    C_RESET=$'\033[0m'
    C_DIM=$'\033[2m'
    C_BOLD=$'\033[1m'
    C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'
else
    C_RESET='' C_DIM='' C_BOLD='' C_RED='' C_GREEN='' C_YELLOW='' C_BLUE=''
fi

_tc_die()  { printf '%s%s[TC][ERROR]%s %s\n' "${C_BOLD}" "${C_RED}" "${C_RESET}" "$*" >&2; return 1; }

_tc_warn() { printf '%s%s[TC][WARN]%s %s\n' "${C_BOLD}" "${C_YELLOW}" "${C_RESET}" "$*" >&2; }

_tc_cmd_exists() { command -v -- "$1" >/dev/null 2>&1; }

_tc_trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

_tc_get_root_dir() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || return 1
    (cd "${script_dir}/../.." && pwd -P) || return 1
}

_tc_source_if_exists() {
    local f="$1"
    [[ -f "$f" ]] || return 0

    # Save current shell options
    local had_u=0 had_e=0 had_pipefail=0
    [[ "$-" == *u* ]] && had_u=1
    [[ "$-" == *e* ]] && had_e=1
    # bash way to query pipefail reliably
    [[ "$(set -o | awk '$1=="pipefail"{print $2}')" == "on" ]] && had_pipefail=1

    # Relax for env files
    set +u +e
    set +o pipefail

    source "$f"
    local rc=$?

    # Restore
    ((had_u)) && set -u || set +u
    ((had_e)) && set -e || set +e
    ((had_pipefail)) && set -o pipefail || set +o pipefail

    return $rc
}

_tc_get_version_num() {
    local bin="${1:-}"
    [[ -z "$bin" ]] && { echo "not found"; return 0; }
    command -v "$bin" >/dev/null 2>&1 || { echo "not found"; return 0; }
    local line ver
    line="$("$bin" --version 2>/dev/null | head -n 1)"
    ver="$(printf '%s\n' "$line" | grep -oE '[0-9]+(\.[0-9]+)+' | head -n 1)"
    [[ -n "$ver" ]] && printf '%s\n' "$ver" || echo "none"
}

_tc_cleanup_internals() {
    local pattern="${1:-_tc_*}"
    local fn
    while IFS= read -r fn; do
        [[ "$fn" == $pattern ]] && unset -f "$fn" 2>/dev/null || true
    done < <(declare -F | awk '{print $3}')
}

# -------- profiles discovery --------

_tc_get_list_profiles() {
    local d="${TC_ROOT_DIR}/toolchain/profiles"
    [[ -d "$d" ]] || return 0

    local p
    for p in "$d"/*; do
        [[ -d "$p" ]] || continue
        basename "$p"
    done | sort || true
}

_tc_profile_exists() {
    local p="$1"
    [[ -n "$p" ]] || return 1
    local d="${TC_ROOT_DIR}/toolchain/profiles/${p}"
    [[ -d "$d" ]] || return 1
    [[ -f "${d}/profile.ini" || -f "${d}/meson.ini" ]] || return 1
    return 0
}

# -------- pretty list + desc --------

_tc_profile_desc_of() {
    local p="$1"
    local ini="${TC_ROOT_DIR}/toolchain/profiles/${p}/profile.ini"
    [[ -f "$ini" ]] || { printf '%s' ""; return 0; }

    local section="" line key val
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        line="${line%%;*}"
        line="$(_tc_trim "$line")"
        [[ -n "$line" ]] || continue

        if [[ "$line" =~ ^\[(.+)\]$ ]]; then
            section="${BASH_REMATCH[1]}"
            continue
        fi

        [[ "$section" == "profile" ]] || continue
        [[ "$line" == *"="* ]] || continue

        key="$(_tc_trim "${line%%=*}")"
        val="$(_tc_trim "${line#*=}")"
        val="${val%\"}"; val="${val#\"}"
        val="${val%\'}"; val="${val#\'}"

        if [[ "$key" == "desc" ]]; then
            printf '%s' "$val"
            return 0
        fi
    done < "$ini"
    printf '%s' ""
}

tc_list() {
    # Cold start: allow printing profiles even when TC_ROOT_DIR is not set yet
    local had_u=0
    [[ "$-" == *u* ]] && had_u=1 && set +u
    if [[ -z "${TC_ROOT_DIR:-}" ]]; then
        TC_ROOT_DIR="$(_tc_get_root_dir 2>/dev/null)" || {
            ((had_u)) && set -u
            printf '%s\n' "  <none>"
            return 0
        }
    fi
    ((had_u)) && set -u

    local profiles=()
    local p
    while IFS= read -r p; do
        [[ -n "$p" ]] && profiles+=("$p")
    done < <(_tc_get_list_profiles)

    if ((${#profiles[@]} == 0)); then
        printf '%s\n' "  <none>"
        return 0
    fi

    local max=0
    for p in "${profiles[@]}"; do
        ((${#p} > max)) && max=${#p}
    done

    local desc mark
    for p in "${profiles[@]}"; do
        mark=" "
        # mark active only when we're in an activated environment (TC_ACTIVE=1)
        if [[ -n "${TC_ACTIVE:-}" && -n "${TC_PROFILE:-}" && "$p" == "${TC_PROFILE}" ]]; then
            mark="*"
        fi

        desc="$(_tc_profile_desc_of "$p")"
        if [[ -n "$desc" ]]; then
            printf '%s  %s%s%-*s %s - %s\n' "$mark" "$C_BOLD" "$C_YELLOW" "$max" "$p" "$C_RESET" "$desc"
        else
            printf '%s  %s%s%-*s%s\n' "$mark" "$C_BOLD" "${C_YELLOW}" "$max" "$p" "${C_RESET}"
        fi
    done
}

# -------- tool resolution --------

_tc_pick_versioned() {
    local base="$1" ver="${2:-}"

    # if base contains '/', treat as explicit path
    if [[ "$base" == */* ]]; then
        _tc_cmd_exists "$base" && { printf '%s\n' "$base"; return 0; }
        return 1
    fi

    if [[ -n "$ver" ]] && _tc_cmd_exists "${base}-${ver}"; then
        printf '%s\n' "${base}-${ver}"
        return 0
    fi

    if _tc_cmd_exists "$base"; then
        printf '%s\n' "$base"
        return 0
    fi

    return 1
}

_tc_pick_into_or_empty() {
    local __var="$1" base="$2" ver="${3:-}" picked tried=""
    # If base is empty, keep empty silently (nothing to resolve)
    if [[ -z "${base:-}" ]]; then
        printf -v "$__var" '%s' ""
        return 1
    fi
    if picked="$(_tc_pick_versioned "$base" "$ver")"; then
        printf -v "$__var" '%s' "$picked"
        return 0
    fi
    # What we tried
    if [[ -n "$ver" ]]; then
        tried="${base}-${ver}, ${base}"
    else
        tried="${base}"
    fi
    # Print warn log
    _tc_warn "Tool '${__var}' not found (tried: ${tried}). Leaving empty"
    printf -v "$__var" '%s' ""
    return 1
}

_tc_norm_toolchain_family() {
    # usage: _tc_norm_toolchain_family "<label>" "<value>"
    # returns: system|gcc|clang
    local label="${1:-toolchain family}"
    local v="${2:-}"
    v="$(_tc_trim "$v")"
    v="${v%\"}"; v="${v#\"}"
    v="${v%\'}"; v="${v#\'}"
    v="$(printf '%s' "$v" | tr '[:upper:]' '[:lower:]')"
    case "$v" in
        ""|"system") printf '%s' "system" ;;
        "gcc"|"clang") printf '%s' "$v" ;;
        *)
            _tc_warn "${label}='$v' is invalid (use: gcc|clang|system). Falling back to system"
            printf '%s' "system" ;;
    esac
}

_tc_apply_ort_toolchain_env() {
    local fam="$(_tc_norm_toolchain_family "ORT_TOOLCHAIN_FAMILY" "${ORT_TOOLCHAIN_FAMILY:-}")"

    # falling to correct
    ORT_TOOLCHAIN_FAMILY="$fam"

    # default: do not override CC/CXX -> system
    if [[ "$fam" == "system" ]]; then
        # optional: unset only if we want to guarantee system
        unset CC CXX AR RANLIB STRIP 2>/dev/null || true
        return 0
    fi

    # clang
    if [[ "$fam" == "clang" ]]; then
        if [[ -z "${CLANG:-}" || -z "${CLANGXX:-}" ]]; then
            _tc_warn "ORT requested clang, but CLANG/CLANGXX are empty. Keeping CC/CXX unchanged (system)"
            return 1
        fi
        export CC="$CLANG"
        export CXX="$CLANGXX"

        # tools
        [[ -n "${LLVM_AR:-}"    ]] && export AR="${LLVM_AR}"
        [[ -n "${LLVM_STRIP:-}" ]] && export STRIP="${LLVM_STRIP}"
        [[ -n "${LLVM_RANLIB:-}" ]] && export RANLIB="${LLVM_RANLIB}"
        return 0
    fi

    # gcc
    if [[ -z "${GCC:-}" || -z "${GXX:-}" ]]; then
        _tc_warn "ORT requested gcc, but GCC/GXX are empty. Keeping CC/CXX unchanged (system)"
        return 1
    fi
    export CC="$GCC"
    export CXX="$GXX"

    # tools
    [[ -n "${GCC_AR:-}"    ]] && export AR="${GCC_AR}"
    [[ -n "${GCC_STRIP:-}" ]] && export STRIP="${GCC_STRIP}"
    [[ -n "${GCC_RANLIB:-}" ]] && export RANLIB="${GCC_RANLIB}"

    return 0
}

_tc_resolve_tools() {
    _tc_pick_into_or_empty MESON      "${MESON:-meson}" ""
    _tc_pick_into_or_empty NINJA      "${NINJA:-ninja}" ""
    _tc_pick_into_or_empty PKG_CONFIG "${PKG_CONFIG:-pkg-config}" ""

    # LLVM tools: prefer base-LLVM_VER
    if [[ -n "${LLVM_VER:-}" ]]; then
        _tc_pick_into_or_empty CLANG          "${CLANG:-clang}"                   "$LLVM_VER"
        _tc_pick_into_or_empty CLANGXX        "${CLANGXX:-clang++}"               "$LLVM_VER"
        _tc_pick_into_or_empty LLVM_AR        "${LLVM_AR:-llvm-ar}"               "$LLVM_VER"
        _tc_pick_into_or_empty LLVM_STRIP     "${LLVM_STRIP:-llvm-strip}"         "$LLVM_VER"
        _tc_pick_into_or_empty LLVM_RANLIB    "${LLVM_RANLIB:-llvm-ranlib}"       "$LLVM_VER"

        _tc_pick_into_or_empty CLANG_FORMAT   "${CLANG_FORMAT:-clang-format}"     "$LLVM_VER"
        _tc_pick_into_or_empty CLANG_TIDY     "${CLANG_TIDY:-clang-tidy}"         "$LLVM_VER"
        _tc_pick_into_or_empty RUN_CLANG_TIDY "${RUN_CLANG_TIDY:-run-clang-tidy}" "$LLVM_VER"
        _tc_pick_into_or_empty SCAN_BUILD     "${SCAN_BUILD:-scan-build}"         "$LLVM_VER"
    else
        # no LLVM_VER: just base
        _tc_pick_into_or_empty CLANG          "${CLANG:-clang}" ""
        _tc_pick_into_or_empty CLANGXX        "${CLANGXX:-clang++}" ""
        _tc_pick_into_or_empty LLVM_AR        "${LLVM_AR:-llvm-ar}" ""
        _tc_pick_into_or_empty LLVM_STRIP     "${LLVM_STRIP:-llvm-strip}" ""
        _tc_pick_into_or_empty LLVM_RANLIB    "${LLVM_RANLIB:-llvm-ranlib}" ""

        _tc_pick_into_or_empty CLANG_FORMAT   "${CLANG_FORMAT:-clang-format}" ""
        _tc_pick_into_or_empty CLANG_TIDY     "${CLANG_TIDY:-clang-tidy}" ""
        _tc_pick_into_or_empty RUN_CLANG_TIDY "${RUN_CLANG_TIDY:-run-clang-tidy}" ""
        _tc_pick_into_or_empty SCAN_BUILD     "${SCAN_BUILD:-scan-build}" ""
    fi

    # GCC tools: prefer base-GCC_VER
    if [[ -n "${GCC_VER:-}" ]]; then
        _tc_pick_into_or_empty GCC        "${GCC:-gcc}"               "$GCC_VER"
        _tc_pick_into_or_empty GXX        "${GXX:-g++}"               "$GCC_VER"
        _tc_pick_into_or_empty GCC_AR     "${GCC_AR:-gcc-ar}"         "$GCC_VER"
        _tc_pick_into_or_empty GCC_STRIP  "${GCC_STRIP:-strip}"       "" # usual `strip` without version
        _tc_pick_into_or_empty GCC_RANLIB "${GCC_RANLIB:-gcc-ranlib}" "$GCC_VER"
    else
        # no GCC_VER: just base
        _tc_pick_into_or_empty GCC        "${GCC:-gcc}" ""
        _tc_pick_into_or_empty GXX        "${GXX:-g++}" ""
        _tc_pick_into_or_empty GCC_AR     "${GCC_AR:-gcc-ar}" ""
        _tc_pick_into_or_empty GCC_STRIP  "${GCC_STRIP:-strip}" ""
        _tc_pick_into_or_empty GCC_RANLIB "${GCC_RANLIB:-gcc-ranlib}" ""
    fi
}

_tc_gen_meson_native_file() {
    local profile="$1" opts_ini="$2" out_ini="$3"

    [[ -f "$opts_ini" ]] || { _tc_die "meson_opt.ini not found: $opts_ini"; return 1; }

    local c="" cpp="" ar="" strip="" ranlib="" pkg="${PKG_CONFIG:-pkg-config}"

    case "$profile" in
        clang-*)
            c="${CLANG:-}"
            cpp="${CLANGXX:-}"
            ar="${LLVM_AR:-}"
            strip="${LLVM_STRIP:-}"
            ranlib="${LLVM_RANLIB:-}"
            ;;
        gcc-*)
            c="${GCC:-}"
            cpp="${GXX:-}"
            ar="${GCC_AR:-}"
            strip="${GCC_STRIP:-}"
            ranlib="${GCC_RANLIB:-}"
            ;;
        *)
            # fallback: if user made custom profile name
            c="${CLANG:-${GCC:-}}"
            cpp="${CLANGXX:-${GXX:-}}"
            ar="${LLVM_AR:-${GCC_AR:-}}"
            strip="${LLVM_STRIP:-${GCC_STRIP:-}}"
            ranlib="${LLVM_RANLIB:-${GCC_RANLIB:-}}"
            ;;
    esac

    # Hard fail if compilers are empty (prevents silent /usr/bin/cc fallback)
    [[ -n "$c"   ]] || { _tc_die "C compiler is empty for profile '$profile' (did tool resolution fail?)"; return 1; }
    [[ -n "$cpp" ]] || { _tc_die "C++ compiler is empty for profile '$profile' (did tool resolution fail?)"; return 1; }

    # ar/strip/ranlib: allow fallback to generic if empty
    [[ -n "$ar"    ]] || ar="ar"
    [[ -n "$strip" ]] || strip="strip"
    [[ -n "$ranlib" ]] || ranlib="ranlib"

    # Optional existence checks (warn, but keep going)
    _tc_cmd_exists "$c"      || _tc_warn "Compiler not found in PATH: c='$c'"
    _tc_cmd_exists "$cpp"    || _tc_warn "Compiler not found in PATH: cpp='$cpp'"
    _tc_cmd_exists "$ar"     || _tc_warn "Tool not found in PATH: ar='$ar'"
    _tc_cmd_exists "$strip"  || _tc_warn "Tool not found in PATH: strip='$strip'"
    _tc_cmd_exists "$ranlib" || _tc_warn "Tool not found in PATH: ranlib='$ranlib'"
    _tc_cmd_exists "$pkg"    || _tc_warn "Tool not found in PATH: pkg-config='$pkg'"

    _tc_meson_quote() {
        # wrap in single quotes; escape internal single quotes
        local s="$1"
        s="${s//\'/\'\\\'\'}"
        printf "'%s'" "$s"
    }

    {
        printf "[binaries]\n"
        printf "c = %s\n"          "$(_tc_meson_quote "$c")"
        printf "cpp = %s\n"        "$(_tc_meson_quote "$cpp")"
        printf "ar = %s\n"         "$(_tc_meson_quote "$ar")"
        printf "strip = %s\n"      "$(_tc_meson_quote "$strip")"
        printf "ranlib = %s\n"     "$(_tc_meson_quote "$ranlib")"
        printf "pkg-config = %s\n" "$(_tc_meson_quote "$pkg")"
        printf "\n"
        cat "$opts_ini"
    } > "$out_ini"
}

# -------- parse profile.ini --------

_tc_ini_apply_profile() {
    local ini="$1"
    local profile_dir="$2"
    [[ -f "$ini" ]] || _tc_die "Missing profile.ini: $ini"

    local section="" line key val
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        line="${line%%;*}"
        line="$(_tc_trim "$line")"
        [[ -n "$line" ]] || continue

        if [[ "$line" =~ ^\[(.+)\]$ ]]; then
            section="${BASH_REMATCH[1]}"
            continue
        fi

        [[ "$line" == *"="* ]] || continue
        key="$(_tc_trim "${line%%=*}")"
        val="$(_tc_trim "${line#*=}")"
        val="${val%\"}"; val="${val#\"}"
        val="${val%\'}"; val="${val#\'}"

        # Skip if nothing, stay empty
        [[ -n "$val" ]] || continue

        case "$section:$key" in
            profile:desc) TC_PROFILE_DESC="$val" ;;

            build:dir) BUILD_DIR="$val" ;;
            build:app_rel) TC_APP_REL="$val" ;;

            tools:llvm_ver) LLVM_VER="$val" ;;
            tools:clang) CLANG="$val" ;;
            tools:clangxx) CLANGXX="$val" ;;
            tools:clang_include_dir) CLANG_INCLUDE_DIR="$val" ;;

            tools:clang_format) CLANG_FORMAT="$val" ;;
            tools:clang_tidy) CLANG_TIDY="$val" ;;
            tools:run_clang_tidy) RUN_CLANG_TIDY="$val" ;;
            tools:scan_build) SCAN_BUILD="$val" ;;

            tools:gcc_ver) GCC_VER="$val" ;;
            tools:gcc) GCC="$val" ;;
            tools:gxx) GXX="$val" ;;
            tools:gcc_include_dir) GCC_INCLUDE_DIR="$val" ;;

            tools:target_triple) TARGET_TRIPLE="$val" ;;
            tools:sysroot) SYSROOT="$val" ;;

            *) : ;;
        esac
    done < "$ini"
}

# -------- atomic loader --------

tc_load() {
    local requested_profile="${1:-}"
    local managed=(
        TC_PROFILE TC_ROOT_DIR TC_PROFILE_DIR TC_PROFILE_INI TC_PROFILE_DESC TC_APP_REL
        BUILD_DIR MESON MESON_OPT_FILE MESON_NATIVE_FILE NINJA PKG_CONFIG JOBS
        LLVM_VER CLANG CLANGXX CLANG_FORMAT CLANG_TIDY RUN_CLANG_TIDY SCAN_BUILD LLVM_AR LLVM_STRIP LLVM_RANLIB CLANG_INCLUDE_DIR
        GCC_VER GCC GXX GCC_AR GCC_STRIP GCC_RANLIB GCC_INCLUDE_DIR
        NONINTERACTIVE KEEP_SUBPROJECTS ORT_TOOLCHAIN_FAMILY ORT_CACHE_ROOT TARGET_TRIPLE SYSROOT
    )

    local _snap_names=() _snap_set=() _snap_vals=()
    local v
    for v in "${managed[@]}"; do
        _snap_names+=("$v")
        if [[ -n "${!v+x}" ]]; then
            _snap_set+=(1)
            _snap_vals+=("${!v}")
        else
            _snap_set+=(0)
            _snap_vals+=("")
        fi
    done

    _tc_restore_snapshot() {
        local i name
        for i in "${!_snap_names[@]}"; do
            name="${_snap_names[i]}"
            if [[ "${_snap_set[i]}" == "1" ]]; then
                printf -v "$name" '%s' "${_snap_vals[i]}"
            else
                unset "$name" 2>/dev/null || true
            fi
        done
    }

    # Resolve root early. Keep a copy so we can list profiles even if snapshot had no TC_ROOT_DIR
    local _root_for_list=""
    TC_ROOT_DIR="$(_tc_get_root_dir)" || { _tc_restore_snapshot; return 1; }
    _root_for_list="${TC_ROOT_DIR}"

    local defaults="${TC_ROOT_DIR}/toolchain/env/defaults.env"
    local localenv="${TC_ROOT_DIR}/toolchain/env/local.env"
    [[ -f "$defaults" ]] || { _tc_die "Missing defaults.env: $defaults"; _tc_restore_snapshot; return 1; }

    # 1) defaults (may set vars, not exported yet)
    _tc_source_if_exists "$defaults"

    # Pick profile + warn if default
    local picked_profile=""
    if [[ -n "$requested_profile" ]]; then
        picked_profile="$requested_profile"
    else
        picked_profile="${TC_PROFILE:-}"
        [[ -n "$picked_profile" ]] || { _tc_die "TC_PROFILE empty (set in defaults/local or pass arg)"; _tc_restore_snapshot; return 1; }
        _tc_warn "Profile not specified, using default: ${picked_profile}"
    fi

    # Validate BEFORE exporting anything
    if ! _tc_profile_exists "$picked_profile"; then
        _tc_die "Unknown profile: '${picked_profile}'"
        printf '%s\n' "[TC] Available profiles:" >&2

        # IMPORTANT: restore snapshot BEFORE printing list (prevents '*' pointing to defaults.env TC_PROFILE)
        _tc_restore_snapshot
        if [[ -z "${TC_ROOT_DIR:-}" && -n "${_root_for_list}" ]]; then
            TC_ROOT_DIR="${_root_for_list}"
        fi
        tc_list >&2 || true
        return 1
    fi

    # Apply profile (safe now)
    TC_PROFILE="$picked_profile"
    TC_PROFILE_DIR="${TC_ROOT_DIR}/toolchain/profiles/${TC_PROFILE}"
    TC_PROFILE_INI="${TC_PROFILE_DIR}/profile.ini"

    TC_PROFILE_DESC=""
    TC_APP_REL="${TC_APP_REL:-src/app/idet/idet_app}"
    MESON_OPT_FILE=""

    if [[ -f "$TC_PROFILE_INI" ]]; then
        _tc_ini_apply_profile "$TC_PROFILE_INI" "$TC_PROFILE_DIR" || { _tc_restore_snapshot; return 1; }
    fi

    if [[ -z "${MESON_OPT_FILE:-}" ]]; then
        MESON_OPT_FILE="${TC_PROFILE_DIR}/meson_opt.ini"
    fi

    if [[ ! -f "${MESON_OPT_FILE}" ]]; then
        _tc_die "MESON_OPT_FILE not found: ${MESON_OPT_FILE}"
        printf '%s\n' "[TC] Available profiles:" >&2

        # same: restore snapshot BEFORE printing list
        _tc_restore_snapshot
        if [[ -z "${TC_ROOT_DIR:-}" && -n "${_root_for_list}" ]]; then
            TC_ROOT_DIR="${_root_for_list}"
        fi
        tc_list >&2 || true
        return 1
    fi

    # 2) local overrides
    _tc_source_if_exists "$localenv"

    # Auto BUILD_DIR if empty or generic "build"
    if [[ -z "${BUILD_DIR:-}" || "${BUILD_DIR}" == "build" ]]; then
        BUILD_DIR="build_${TC_PROFILE//-/_}"
    fi

    # Setup toolchain
    _tc_resolve_tools
    _tc_apply_ort_toolchain_env

    # Generate native file with [binaries] from env/tool resolution + profile meson.ini options
    local meson_native_ini="${TC_PROFILE_DIR}/meson_native_gen.ini"
    _tc_gen_meson_native_file "${TC_PROFILE}" "${MESON_OPT_FILE}" "$meson_native_ini" || { _tc_restore_snapshot; return 1; }
    MESON_NATIVE_FILE="$meson_native_ini"

    # EXPORT ONLY ON SUCCESS
    export TC_PROFILE TC_ROOT_DIR TC_PROFILE_DIR TC_PROFILE_INI TC_PROFILE_DESC TC_APP_REL
    export BUILD_DIR MESON MESON_OPT_FILE MESON_NATIVE_FILE NINJA PKG_CONFIG JOBS
    export LLVM_VER CLANG CLANGXX CLANG_FORMAT CLANG_TIDY RUN_CLANG_TIDY SCAN_BUILD LLVM_AR LLVM_STRIP LLVM_RANLIB CLANG_INCLUDE_DIR
    export GCC_VER GCC GXX GCC_AR GCC_STRIP GCC_RANLIB GCC_INCLUDE_DIR
    export NONINTERACTIVE KEEP_SUBPROJECTS ORT_TOOLCHAIN_FAMILY ORT_CACHE_ROOT TARGET_TRIPLE SYSROOT

    unset -f _tc_restore_snapshot 2>/dev/null || true

    return 0
}

# -------- pretty config log --------

tc_print() {
    local title="[TC]"
    local sep='---------------------------------------------------------------------------'

    # label|varname|kind|default|verflag
    # kind: kv | bool | blank
    # verflag: ver (show version)
    local -a rows=(
        "TC_PROFILE|TC_PROFILE|kv|none|"
        "BUILD_DIR|BUILD_DIR|kv|none|"
        "ROOT_DIR|TC_ROOT_DIR|kv|none|"
        "APP_REL|TC_APP_REL|kv|none|"
        "PROFILE_DIR|TC_PROFILE_DIR|kv|none|"
        "MESON_OPT_FILE|MESON_OPT_FILE|kv|none|"
        "PROFILE_INI|TC_PROFILE_INI|kv|none|"
        "MESON_NATIVE_FILE|MESON_NATIVE_FILE|kv|none|"
        "BLANK||blank||"

        "PKG_CONFIG|PKG_CONFIG|kv|none|ver"
        "MESON|MESON|kv|none|ver"
        "NINJA|NINJA|kv|none|ver"
        "JOBS|JOBS|kv|none|"
        "BLANK||blank||"

        "LLVM_VER|LLVM_VER|kv|none|"
        "LLVM_AR|LLVM_AR|kv|none|ver"
        "LLVM_STRIP|LLVM_STRIP|kv|none|ver"
        "LLVM_RANLIB|LLVM_RANLIB|kv|none|ver"
        "CLANG|CLANG|kv|none|ver"
        "CLANGXX|CLANGXX|kv|none|ver"
        "CLANG_INCLUDE_DIR|CLANG_INCLUDE_DIR|kv|none|"
        "BLANK||blank||"

        "CLANG_FORMAT|CLANG_FORMAT|kv|none|ver"
        "CLANG_TIDY|CLANG_TIDY|kv|none|ver"
        "RUN_CLANG_TIDY|RUN_CLANG_TIDY|kv|none|ver"
        "SCAN_BUILD|SCAN_BUILD|kv|none|ver"
        "BLANK||blank||"

        "GCC_VER|GCC_VER|kv|none|"
        "GCC_AR|GCC_AR|kv|none|ver"
        "GCC_STRIP|GCC_STRIP|kv|none|ver"
        "GCC_RANLIB|GCC_RANLIB|kv|none|ver"
        "GCC|GCC|kv|none|ver"
        "GXX|GXX|kv|none|ver"
        "GCC_INCLUDE_DIR|GCC_INCLUDE_DIR|kv|none|"
        "BLANK||blank||"

        "ORT_TOOLCHAIN|ORT_TOOLCHAIN_FAMILY|kv|system|"
        "ORT_CACHE_ROOT|ORT_CACHE_ROOT|kv|none|"
        "NONINTERACTIVE|NONINTERACTIVE|bool|0|"
        "KEEP_SUBPROJECTS|KEEP_SUBPROJECTS|bool|0|"
        "TARGET_TRIPLE|TARGET_TRIPLE|kv|none|"
        "SYSROOT|SYSROOT|kv|none|"
    )

    # compute max label width
    local max_key=0 row label var kind def verflag
    for row in "${rows[@]}"; do
        IFS='|' read -r label var kind def verflag <<<"$row"
        [[ "$kind" == "blank" ]] && continue
        (( ${#label} > max_key )) && max_key=${#label}
    done

    # -------- formatters --------

    _tc_fmt_val() {
        local v="$1"
        if [[ -z "$v" || "$v" == "none" ]]; then
            printf '%snone%s' "$C_DIM" "$C_RESET"
        else
            printf '%s%s%s' "$C_BOLD" "$v" "$C_RESET"
        fi
    }

    _tc_fmt_ver() {
        local v="$1"
        v="${v//$'\n'/ }"
        v="${v#"${v%%[![:space:]]*}"}"
        v="${v%"${v##*[![:space:]]}"}"
        if [[ -z "$v" || "$v" == "not found" || "$v" == "none" ]]; then
            printf '%s%s%s' "$C_DIM" "=> version not found" "$C_RESET"
        else
            printf '%s%s%s' "$C_GREEN" "=> v$v" "$C_RESET"
        fi
    }

    _tc_bool01() {
        local v="${1:-0}"
        v="$(_tc_trim "$v")"
        v="${v%\"}"; v="${v#\"}"
        v="${v%\'}"; v="${v#\'}"
        case "$v" in
            1|true|TRUE|yes|YES|on|ON)  printf '1' ;;
            0|false|FALSE|no|NO|off|OFF|"") printf '0' ;;
            *) printf '0' ;;
        esac
    }

    _tc_fmt_tf() {
        local b="${1:-0}"
        if [[ "$b" == "1" ]]; then
            printf '%s%strue%s'  "$C_BOLD" "$C_GREEN" "$C_RESET"
        else
            printf '%s%sfalse%s' "$C_BOLD" "$C_RED"   "$C_RESET"
        fi
    }

    _tc_prefix() {
        local key="$1"
        printf '%s%s %s%-*s%s %s:%s ' \
            "$C_BOLD" "$title" "$C_BLUE" "$max_key" "$key" "$C_RESET" \
            "$C_DIM" "$C_RESET"
    }

    # -------- version cache --------

    local -a _ver_cmds=() _ver_vals=()

    _tc_ver_cache_set() {
        local cmd="$1" i
        [[ -z "$cmd" || "$cmd" == "none" ]] && return 0
        for i in "${!_ver_cmds[@]}"; do
            [[ "${_ver_cmds[i]}" == "$cmd" ]] && return 0
        done
        _ver_cmds+=("$cmd")
        _ver_vals+=("$(_tc_get_version_num "$cmd")")
        return 0
    }

    _tc_ver_cache_get() {
        local cmd="$1" i
        [[ -z "$cmd" || "$cmd" == "none" ]] && { printf '%s\n' "not found"; return 0; }
        for i in "${!_ver_cmds[@]}"; do
            if [[ "${_ver_cmds[i]}" == "$cmd" ]]; then
                printf '%s\n' "${_ver_vals[i]}"
                return 0
            fi
        done
        # safety fallback
        printf '%s\n' "$(_tc_get_version_num "$cmd")"
    }

    # precompute versions first
    local val
    for row in "${rows[@]}"; do
        IFS='|' read -r label var kind def verflag <<<"$row"
        [[ "$kind" == "blank" ]] && continue
        [[ "$verflag" == "ver" ]] || continue
        val="${!var-}"
        [[ -z "$val" ]] && val="$def"
        [[ -z "$val" ]] && val="none"
        _tc_ver_cache_set "$val"
    done

    # -------- print --------

    printf '%s%s%s\n' "$C_DIM" "$sep" "$C_RESET"

    for row in "${rows[@]}"; do
        IFS='|' read -r label var kind def verflag <<<"$row"

        if [[ "$kind" == "blank" ]]; then
            printf '\n'
            continue
        fi

        val="${!var-}"
        [[ -z "$val" ]] && val="$def"
        [[ -z "$val" ]] && val="none"

        if [[ "$kind" == "bool" ]]; then
            local b; b="$(_tc_bool01 "$val")"
            _tc_prefix "$label"
            _tc_fmt_tf "$b"
            printf '\n'
            continue
        fi

        _tc_prefix "$label"
        _tc_fmt_val "$val"
        if [[ "$verflag" == "ver" ]]; then
            printf ' '
            _tc_fmt_ver "$(_tc_ver_cache_get "$val")"
        fi
        printf '\n'
    done

    printf '%s%s%s\n' "$C_DIM" "$sep" "$C_RESET"
}
