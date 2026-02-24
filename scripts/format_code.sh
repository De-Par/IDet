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

need_cmd() { command -v -- "$1" >/dev/null 2>&1 || die "Command '$1' not found in PATH"; }

usage() {
    cat <<EOF
Formats C/C++ sources in-place (or checks formatting without modifying)

Usage:
    ./scripts/format_code.sh [--check|--help]

Examples:
    source toolchain/activate.sh 
    ./scripts/format_code.sh --check
    ./scripts/format_code.sh 
EOF
}

# ------------------------- args -------------------------

CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK_ONLY=1; shift ;;
        -h|--help|help) usage; exit 0 ;;
        *) die "Unknown argument: '$1' (use --help)" ;;
    esac
done

# ------------------------- require activated env -------------------------

if [[ "${TC_ACTIVE:-0}" != "1" ]]; then
    die "Toolchain environment is not active. Run: source toolchain/activate.sh <profile>"
fi

CLANG_FORMAT_BIN="${CLANG_FORMAT:-}"
[[ -n "${CLANG_FORMAT_BIN}" ]] || die "CLANG_FORMAT is not set in env (activate.sh should set it)"
need_cmd "${CLANG_FORMAT_BIN}"

log "[INFO] clang-format: ${CLANG_FORMAT_BIN}"
"${CLANG_FORMAT_BIN}" --version 2>/dev/null || true

# ------------------------- inputs (dirs/exts) -------------------------

# Defaults apply only if env is not set (we do NOT override env values)
SEARCH_DIRS_STR="${SEARCH_DIRS:-include src tests}"
EXTS_STR="${EXTS:-c h cc cpp cxx hpp hxx}"

# Split env strings into arrays (space-separated)
IFS=' ' read -r -a SEARCH_DIRS_ARR <<< "${SEARCH_DIRS_STR}"
IFS=' ' read -r -a FILE_EXTENSIONS <<< "${EXTS_STR}"

# Keep only existing dirs
REAL_DIRS=()
for d in "${SEARCH_DIRS_ARR[@]-}"; do
    if [[ -d "${d}" ]]; then
        REAL_DIRS+=("${d}")
    else
        warn "Search dir '${d}' not found, skip"
    fi
done

if ((${#REAL_DIRS[@]} == 0)); then
    log "[INFO] No search dirs exist. Nothing to do"
    exit 0
fi

# Build find expression: \( -name '*.c' -o -name '*.cpp' ... \)
find_expr=()
for ext in "${FILE_EXTENSIONS[@]-}"; do
    [[ -n "${ext}" ]] || continue
    if ((${#find_expr[@]} > 0)); then
        find_expr+=(-o)
    fi
    find_expr+=(-name "*.${ext}")
done

FILES=()
while IFS= read -r -d '' f; do
    FILES+=("$f")
done < <(find "${REAL_DIRS[@]-}" -type f \( "${find_expr[@]-}" \) -print0 2>/dev/null)

if ((${#FILES[@]} == 0)); then
    log "[INFO] No files to format. Exit"
    exit 0
fi

[[ "${CHECK_ONLY:-0}" == "1" || "${CHECK_ONLY:-}" == "true" ]] && \
    log "[INFO] Only check  : true" || log "[INFO] Only check  : false"
log "[INFO] Files found : ${#FILES[@]}"

# ------------------------- fast check mode (if supported) -------------------------
# If clang-format supports --dry-run and -Werror, do a fast check and exit early.
if (( CHECK_ONLY == 1 )); then
    if "${CLANG_FORMAT_BIN}" --help 2>/dev/null | grep -q -- '--dry-run' \
        && "${CLANG_FORMAT_BIN}" --help 2>/dev/null | grep -q -- '-Werror'; then
        if "${CLANG_FORMAT_BIN}" --dry-run -Werror -- "${FILES[@]-}" >/dev/null 2>&1; then
            log "✅ No formatting changes"
            exit 0
        fi
    fi
fi

# ------------------------- portable mktemp -------------------------

mktemp_dir() {
    # Linux: mktemp -d works; macOS: mktemp -d needs -t template
    local d=""
    if d="$(mktemp -d 2>/dev/null)"; then
        printf '%s\n' "$d"
        return 0
    fi
    d="$(mktemp -d -t idet_clangfmt 2>/dev/null)" || return 1
    printf '%s\n' "$d"
}

TMPDIR="$(mktemp_dir)" || die "mktemp failed"
trap 'rm -rf -- "${TMPDIR}"' EXIT

# ------------------------- backup originals -------------------------

for f in "${FILES[@]-}"; do
    mkdir -p -- "${TMPDIR}/$(dirname -- "${f}")"
    cp -p -- "${f}" "${TMPDIR}/${f}"
done

# ------------------------- format in-place -------------------------

"${CLANG_FORMAT_BIN}" -i -- "${FILES[@]-}"

# ------------------------- detect changes -------------------------

changed_files=()
for f in "${FILES[@]-}"; do
    if ! cmp -s -- "${f}" "${TMPDIR}/${f}"; then
        changed_files+=("${f}")
    fi
done

# Restore originals in check mode
if (( CHECK_ONLY == 1 )); then
    for f in "${FILES[@]-}"; do
        cp -p -- "${TMPDIR}/${f}" "${f}"
    done
fi

# ------------------------- report -------------------------

if ((${#changed_files[@]} == 0)); then
    log "✅ No formatting changes"
    exit 0
fi

log "[INFO] Formatting would change files:"
for f in "${changed_files[@]-}"; do
    log "🟡 ${f}"
done

if (( CHECK_ONLY == 1 )); then
    warn "Formatting disabled (run without --check to apply)"
    exit 0
fi

log "✅ Done"
