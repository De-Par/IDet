#!/usr/bin/env bash

set -euo pipefail

# ------------------------- paths -------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${ROOT_DIR}"

# ------------------------- logging -------------------------

log()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "[WARN] $*" >&2; }
die()  { printf '%s\n' "[ERROR] $*" >&2; exit 1; }
need_cmd() { command -v -- "$1" >/dev/null 2>&1 || die "Required command '$1' not found in PATH"; }

usage() {
    cat <<'EOF'
Project builder script (Meson)

Usage:
    ./scripts/build.sh [mode] -- [meson_setup_args...]

Modes:
    (empty)     setup/reconfigure + build
    setup       setup/reconfigure only
    build       build only (requires existing build dir)
    force|f     wipe build dir (and optionally subprojects) + setup/reconfigure + build

Passing Meson args:
    Everything after `--` is forwarded to `meson setup` / `meson setup --reconfigure`

Examples:
    ./scripts/build.sh
    ./scripts/build.sh force
    ./scripts/build.sh setup -- -Dbuild_tests=false -Duse_openmp=true
    ./scripts/build.sh force -- -Donnxruntime_system=true

Typical workflow:
    source ./toolchain/scripts/activate.sh
    ./scripts/build.sh force
EOF
}

# ------------------------- parse args -------------------------

MODE=""
MESON_USER_ARGS=()

# 1) optional mode
if [[ $# -gt 0 ]]; then
    case "${1}" in
        -h|--help|help) usage; exit 0 ;;
        "" ) ;;
        force|f|setup|build) MODE="${1}"; shift ;;
        --) MODE=""; shift ;;
        *) die "Unknown mode '${1}'. Use: force|f|setup|build (or no args). Use --help" ;;
    esac
fi

# 2) args to forward to meson setup/reconfigure
if [[ $# -gt 0 ]]; then
    if [[ "${1}" == "--" ]]; then
        shift
        MESON_USER_ARGS=("$@")
    else
        # Allow passing meson args without '--' (after mode), but recommend using '--'
        MESON_USER_ARGS=("$@")
        warn "Meson args passed without '--'. Recommended: ./scripts/build.sh ${MODE:-<mode>} -- <meson args...>"
    fi
fi

# ------------------------- env (from activate.sh) -------------------------

if [[ "${TC_ACTIVE:-0}" != "1" ]]; then
    warn "No active toolchain detected (TC_ACTIVE!=1)."
    warn "Recommended: source toolchain/scripts/activate.sh <profile>"
fi

BUILD_DIR="${BUILD_DIR:-build}"
MESON_BIN="${MESON:-meson}"
MESON_NATIVE_FILE="${MESON_NATIVE_FILE:-}"
KEEP_SUBPROJECTS="${KEEP_SUBPROJECTS:-1}"
NONINTERACTIVE="${NONINTERACTIVE:-1}"
JOBS="${JOBS:-0}"

need_cmd "${MESON_BIN}"

# ------------------------- portable helpers -------------------------

os_nproc() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu 2>/dev/null || echo 8
    else
        echo 8
    fi
}

resolve_native_file_abs() {
    local nf="${1:-}"
    [[ -n "${nf}" ]] || return 0
    if [[ "${nf}" == /* ]]; then
        printf '%s\n' "${nf}"
    else
        printf '%s\n' "${ROOT_DIR}/${nf}"
    fi
}

# Simple & safe: BUILD_DIR must be a relative path inside repo
safe_rm_rf() {
    local p="${1:-}"
    [[ -n "${p}" ]] || die "safe_rm_rf: empty path"

    case "${p}" in
        /*) die "safe_rm_rf: absolute path is not allowed: '${p}'" ;;
        "."|"./") die "safe_rm_rf: refusing to remove '${p}'" ;;
    esac

    # refuse any '..' path component
    if [[ "${p}" == *"/.."* || "${p}" == ".."* || "${p}" == *"../"* ]]; then
        die "safe_rm_rf: refusing suspicious path with '..': '${p}'"
    fi

    local abs="${ROOT_DIR}/${p}"
    if [[ -e "${abs}" ]]; then
        rm -rf -- "${abs}"
        log "  - Removed ${p}"
    else
        log "  - Skip (not found) ${p}"
    fi
}

# ------------------------- meson native-file args -------------------------

MESON_NATIVE_ARGS=()
native_label="(system default compiler)"
if [[ -n "${MESON_NATIVE_FILE}" ]]; then
    nf_abs="$(resolve_native_file_abs "${MESON_NATIVE_FILE}")"
    [[ -f "${nf_abs}" ]] || die "MESON_NATIVE_FILE not found: '${nf_abs}'"
    MESON_NATIVE_ARGS=(--native-file "${nf_abs}")
    native_label="${nf_abs}"
fi

# ------------------------- ops -------------------------

print_effective_build_config() {
    local jobs="${JOBS:-0}"
    if [[ -z "${jobs}" || "${jobs}" == "0" ]]; then jobs="$(os_nproc)"; fi

    log "----------------------------------------------"
    log "[BUILD] ROOT_DIR          : ${ROOT_DIR}"
    log "[BUILD] TC_ACTIVE         : ${TC_ACTIVE:-0}"
    log "[BUILD] TC_PROFILE        : ${TC_PROFILE:-<none>}"
    log "[BUILD] BUILD_DIR         : ${BUILD_DIR}"
    log "[BUILD] MESON             : ${MESON_BIN}"
    log "[BUILD] MESON_NATIVE_FILE : ${native_label}"
    log "[BUILD] JOBS              : ${jobs}"
    log "[BUILD] NONINTERACTIVE    : ${NONINTERACTIVE}"
    log "[BUILD] KEEP_SUBPROJECTS  : ${KEEP_SUBPROJECTS}"
    if ((${#MESON_USER_ARGS[@]})); then
        log "[BUILD] MESON_USER_ARGS   : ${MESON_USER_ARGS[*]}"
    fi
    log "----------------------------------------------"
    "${MESON_BIN}" --version 2>/dev/null | sed 's/^/[BUILD] MESON_VERSION     : /' || true
    log "----------------------------------------------"
}

apply_packagefiles_overlays() {
    local pf_dir="${ROOT_DIR}/subprojects/packagefiles"
    [[ -d "${pf_dir}" ]] || return 0

    log "[INFO] Applying subprojects/packagefiles overlays..."

    local d name
    shopt -s nullglob
    for d in "${pf_dir}"/*; do
        [[ -d "${d}" ]] || continue
        name="$(basename "${d}")"

        if [[ -d "${ROOT_DIR}/subprojects/${name}" ]]; then
            "${MESON_BIN}" subprojects packagefiles --apply "${name}" || warn "packagefiles apply failed for '${name}' (continuing)"
        else
            log "Skip: ${name} (subprojects/${name} not present)"
        fi
    done
    shopt -u nullglob
}

invalidate_subproject_stamps() {
    local babs="${ROOT_DIR}/${BUILD_DIR}"
    [[ -d "${babs}" ]] || return 0
    # stamps
    rm -f -- "${babs}/subprojects/onnxruntime/onnxruntime_build.stamp" 2>/dev/null || true
}

clean_artifacts() {
    log "[INFO] Clean mode: removing build artifacts..."
    safe_rm_rf "${BUILD_DIR}"

    if [[ "${KEEP_SUBPROJECTS}" != "1" ]]; then
        local ARTEFACTS=(
            "subprojects/stb"
            "subprojects/gtest"
            "subprojects/clipper2"
            "subprojects/indicators"
            "subprojects/onnxruntime"
        )
        local d
        for d in "${ARTEFACTS[@]}"; do safe_rm_rf "$d"; done
    else
        log "  - KEEP_SUBPROJECTS=1 -> keeping subprojects/*"
        apply_packagefiles_overlays
        invalidate_subproject_stamps
    fi
}

meson_setup_or_reconfigure() {
    local -a native_args=("${MESON_NATIVE_ARGS[@]-}")
    local -a user_args=("${MESON_USER_ARGS[@]-}")

    if [[ ! -d "${BUILD_DIR}" ]]; then
        log "[INFO] meson setup: '${BUILD_DIR}'"
        "${MESON_BIN}" setup "${BUILD_DIR}" "${native_args[@]}" "${user_args[@]}"
    else
        log "[INFO] meson reconfigure: '${BUILD_DIR}'"
        "${MESON_BIN}" setup --reconfigure "${BUILD_DIR}" "${native_args[@]}" "${user_args[@]}"
    fi
}

meson_build() {
    local jobs="${JOBS:-0}"
    if [[ -z "${jobs}" || "${jobs}" == "0" ]]; then jobs="$(os_nproc)"; fi
    log "[INFO] meson compile: dir='${BUILD_DIR}' jobs=${jobs}"
    "${MESON_BIN}" compile -C "${BUILD_DIR}" -j "${jobs}"
}

# ------------------------- run -------------------------

case "${MODE}" in
    force|f)
        clean_artifacts
        meson_setup_or_reconfigure
        ;;
    setup)
        meson_setup_or_reconfigure
        exit 0
        ;;
    build)
        [[ -d "${BUILD_DIR}" ]] || die "Build dir '${BUILD_DIR}' not found (run './scripts/build.sh setup' first)"
        ;;
    "")
        meson_setup_or_reconfigure
        ;;
esac

print_effective_build_config

if [[ "${NONINTERACTIVE}" != "1" ]]; then
    read -r -p "[INFO] Press ENTER to run compilation or type anything else to exit: " ans || true
    [[ -z "${ans}" ]] || { log "[INFO] Exit"; exit 1; }
fi

meson_build
