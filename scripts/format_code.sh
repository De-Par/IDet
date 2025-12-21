#!/bin/bash

# Run clang-format for all files with specidied extensions from FILE_EXTENSIONS in SEARCH_DIRS

set -euo pipefail

# Navigate to root dir of project
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd $PARENT_DIR

# -------------------------------
# Search clang-format
# -------------------------------
CLANG_FORMAT_BIN="${CLANG_FORMAT_BIN:-}"

if [[ -n "$CLANG_FORMAT_BIN" ]]; then
    # User set specified bin path
    if ! command -v "$CLANG_FORMAT_BIN" >/dev/null 2>&1; then
        echo "[ERROR] Specified CLANG_FORMAT_BIN='$CLANG_FORMAT_BIN' not find in PATH." >&2
        exit 1
    fi
else
    # Auto-search by names
    CANDIDATES=(
        "clang-format"
        "clang-format-21"
        "clang-format-20"
        "clang-format-19"
        "clang-format-18"
        "clang-format-17"
        "clang-format-16"
        "clang-format-15"
        "clang-format-14"
    )

    for bin in "${CANDIDATES[@]}"; do
        if command -v "$bin" >/dev/null 2>&1; then
            CLANG_FORMAT_BIN="$bin"
            break
        fi
    done

    if [[ -z "$CLANG_FORMAT_BIN" ]]; then
        echo "[ERROR] clang-format not found: install it and check in PATH" >&2
        echo "The following command options are considered: ${CANDIDATES[*]}" >&2
        exit 1
    fi
fi

echo "[INFO] Find clang-format: $CLANG_FORMAT_BIN"
if ! "$CLANG_FORMAT_BIN" --version; then
    echo "⚠️ Enable to get clang-format version" >&2
fi
echo "----------------------------------------------"

# Where search files (dirs)
SEARCH_DIRS=("src" "include")

# Which extensions formatting
FILE_EXTENSIONS=("c" "h" "cc" "cpp" "cxx" "hpp" "hxx")

# Build command for 'find': -name '*.cpp' -o -name '*.hpp' ...
find_expr=()
for ext in "${FILE_EXTENSIONS[@]}"; do
    if ((${#find_expr[@]} > 0)); then
        find_expr+=(-o)
    fi
    find_expr+=(-name "*.${ext}")
done

# Search files, carefully processing spaces in the names (print0 + mapfile)
FILES=()
while IFS= read -r -d '' f; do
    FILES+=("$f")
done < <(find "${SEARCH_DIRS[@]}" -type f \( "${find_expr[@]}" \) -print0)

if ((${#FILES[@]} == 0)); then
    echo "[INFO] No files to format. Exit"
    exit 0
fi

# Temp catalog for old versions of files
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

for f in "${FILES[@]}"; do
    mkdir -p "${tmpdir}/$(dirname "$f")"
    cp "$f" "${tmpdir}/$f"
done

# Formatting on-site
"$CLANG_FORMAT_BIN" -i "${FILES[@]}"

# Check changes
changed_files=()
for f in "${FILES[@]}"; do
    if ! cmp -s "$f" "${tmpdir}/$f"; then
        changed_files+=("$f")
    fi
done

echo "Formatting..."
if ((${#changed_files[@]} == 0)); then
    echo "✅ There are no changes, everything is fine"
else
    echo "Formatted files:"
    for line in "${changed_files[@]}"; do
        echo "🟡 $line"
    done
fi