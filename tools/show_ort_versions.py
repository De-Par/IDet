#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple


REPO_URL = "https://github.com/microsoft/onnxruntime.git"

SEMVER_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

CMAKE_MIN_RE = re.compile(
    r"cmake_minimum_required\s*\(\s*VERSION\s+([0-9]+(?:\.[0-9]+){1,2})",
    re.IGNORECASE,
)

PYTHON_REQUIRES_RE = re.compile(
    r"""
    (?:
        python_requires
        |
        requires-python
        |
        Requires-Python
    )
    \s*
    (?:
        =
        |
        :
    )
    \s*
    ["']?
    ([^"'\n\r]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

PYTHON_MIN_RE = re.compile(r">=\s*([0-9]+(?:\.[0-9]+){0,2})")

CMAKE_CXX_STANDARD_RE = re.compile(
    r"""
    (?:
        set\s*\(\s*CMAKE_CXX_STANDARD\s+([0-9]+)
        |
        CMAKE_CXX_STANDARD\s+([0-9]+)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

CXX_STD_FEATURE_RE = re.compile(r"cxx_std_([0-9]+)", re.IGNORECASE)

CMAKE_MIN_CANDIDATE_PATHS = (
    "cmake/CMakeLists.txt",
    "CMakeLists.txt",
)

PYTHON_MIN_CANDIDATE_PATHS = (
    "pyproject.toml",
    "setup.py",
    "tools/ci_build/build.py",
)

CXX_STANDARD_CANDIDATE_PATHS = (
    "cmake/CMakeLists.txt",
    "CMakeLists.txt",
    "cmake/onnxruntime_common.cmake",
    "cmake/onnxruntime_providers.cmake",
    "cmake/onnxruntime_python.cmake",
)


@dataclass(order=True, frozen=True)
class TagInfo:
    major: int
    minor: int
    patch: int
    name: str

    @property
    def version_tuple(self) -> Tuple[int, int, int]:
        return (self.major, self.minor, self.patch)


@dataclass
class Row:
    tag: TagInfo
    commit: str
    cmake_min: Optional[str]
    python_min: Optional[str]
    cxx_standard: Optional[str]


class ProgressBar:
    def __init__(self, total: int, enabled: bool = True, width: int = 32) -> None:
        self.total = max(total, 1)
        self.enabled = enabled
        self.width = width
        self.last_len = 0

    def update(self, current: int, label: str = "") -> None:
        if not self.enabled:
            return

        current = min(max(current, 0), self.total)

        ratio = current / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = ratio * 100.0

        message = f"\r[{bar}] {current}/{self.total} {percent:6.2f}%"

        if label:
            message += f"  {label}"

        padding = " " * max(0, self.last_len - len(message))

        sys.stderr.write(message + padding)
        sys.stderr.flush()

        self.last_len = len(message)

    def finish(self) -> None:
        if not self.enabled:
            return

        sys.stderr.write("\n")
        sys.stderr.flush()


def run(cmd: List[str], cwd: Optional[str] = None, check: bool = True) -> str:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{p.stdout}\n"
            f"STDERR:\n{p.stderr}"
        )

    return p.stdout


def ensure_git_repo(repo_dir: str, repo_url: str) -> None:
    git_dir = os.path.join(repo_dir, ".git")

    if os.path.isdir(git_dir):
        return

    os.makedirs(repo_dir, exist_ok=True)

    run(["git", "init"], cwd=repo_dir)
    run(["git", "remote", "add", "origin", repo_url], cwd=repo_dir)

    update_tags(repo_dir)


def update_tags(repo_dir: str) -> None:
    run(
        [
            "git",
            "-c",
            "protocol.version=2",
            "fetch",
            "--tags",
            "--filter=blob:none",
            "origin",
        ],
        cwd=repo_dir,
    )


def list_semver_tags(repo_dir: str) -> List[TagInfo]:
    out = run(["git", "tag", "--list", "v*"], cwd=repo_dir)

    tags: List[TagInfo] = []

    for line in out.splitlines():
        tag = line.strip()
        match = SEMVER_TAG_RE.match(tag)

        if not match:
            continue

        tags.append(
            TagInfo(
                major=int(match.group(1)),
                minor=int(match.group(2)),
                patch=int(match.group(3)),
                name=tag,
            )
        )

    tags.sort()
    return tags


def resolve_commit(repo_dir: str, tag_name: str) -> str:
    return run(["git", "rev-list", "-n", "1", tag_name], cwd=repo_dir).strip()


def git_show_file(repo_dir: str, tag_name: str, path: str) -> Optional[str]:
    p = subprocess.run(
        ["git", "show", f"{tag_name}:{path}"],
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if p.returncode != 0:
        return None

    return p.stdout


def detect_cmake_min(text: str) -> Optional[str]:
    match = CMAKE_MIN_RE.search(text)

    if not match:
        return None

    return match.group(1)


def normalize_version_str(value: str) -> str:
    parts = value.split(".")

    if len(parts) == 1:
        return f"{parts[0]}.0"

    return value


def parse_version_str(value: str) -> Tuple[int, int, int]:
    parts = [int(part) for part in value.split(".")]

    while len(parts) < 3:
        parts.append(0)

    return tuple(parts[:3])


def detect_python_min(text: str) -> Optional[str]:
    versions: List[str] = []

    for requires_match in PYTHON_REQUIRES_RE.finditer(text):
        spec = requires_match.group(1)

        for min_match in PYTHON_MIN_RE.finditer(spec):
            versions.append(normalize_version_str(min_match.group(1)))

    if not versions:
        return None

    versions.sort(key=parse_version_str)
    return versions[-1]


def detect_cxx_standard(text: str) -> Optional[str]:
    standards: List[int] = []

    for match in CMAKE_CXX_STANDARD_RE.finditer(text):
        value = match.group(1) or match.group(2)

        if value:
            standards.append(int(value))

    for match in CXX_STD_FEATURE_RE.finditer(text):
        standards.append(int(match.group(1)))

    if not standards:
        return None

    return str(max(standards))


def inspect_first_match(
    repo_dir: str,
    tag_name: str,
    paths: Iterable[str],
    detector: Callable[[str], Optional[str]],
) -> Optional[str]:
    for path in paths:
        content = git_show_file(repo_dir, tag_name, path)

        if content is None:
            continue

        value = detector(content)

        if value:
            return value

    return None


def inspect_tag(repo_dir: str, tag: TagInfo) -> Row:
    commit = resolve_commit(repo_dir, tag.name)

    cmake_min = inspect_first_match(
        repo_dir=repo_dir,
        tag_name=tag.name,
        paths=CMAKE_MIN_CANDIDATE_PATHS,
        detector=detect_cmake_min,
    )

    python_min = inspect_first_match(
        repo_dir=repo_dir,
        tag_name=tag.name,
        paths=PYTHON_MIN_CANDIDATE_PATHS,
        detector=detect_python_min,
    )

    cxx_standard = inspect_first_match(
        repo_dir=repo_dir,
        tag_name=tag.name,
        paths=CXX_STANDARD_CANDIDATE_PATHS,
        detector=detect_cxx_standard,
    )

    return Row(
        tag=tag,
        commit=commit,
        cmake_min=cmake_min,
        python_min=python_min,
        cxx_standard=cxx_standard,
    )


def inspect_tags_with_progress(
    repo_dir: str,
    tags: List[TagInfo],
    show_progress: bool,
) -> List[Row]:
    rows: List[Row] = []
    progress = ProgressBar(total=len(tags), enabled=show_progress)

    progress.update(0, "starting")

    for index, tag in enumerate(tags, start=1):
        rows.append(inspect_tag(repo_dir, tag))
        progress.update(index, tag.name)

    progress.finish()
    return rows


def print_rows(rows: List[Row]) -> None:
    print(
        f"{'Tag':<10} "
        f"{'Commit':<40} "
        f"{'CMake min':<10} "
        f"{'Python min':<10} "
        f"{'C++ std':<8}"
    )
    print("-" * 100)

    for row in rows:
        cmake_min = row.cmake_min if row.cmake_min else "N/A"
        python_min = row.python_min if row.python_min else "N/A"
        cxx_standard = f"C++{row.cxx_standard}" if row.cxx_standard else "N/A"

        print(
            f"{row.tag.name:<10} "
            f"{row.commit:<40} "
            f"{cmake_min:<10} "
            f"{python_min:<10} "
            f"{cxx_standard:<8}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print minimal CMake, Python and C++ standard for ONNX Runtime semver tags"
    )

    parser.add_argument(
        "--repo-dir",
        type=str,
        default="~/.cache/onnxruntime-src",
        help="Path to local cache repo. If omitted, a temporary repo is used.",
    )

    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Do not run git fetch --tags if repo already exists.",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar.",
    )

    args = parser.parse_args()

    temp_dir = None

    try:
        if args.repo_dir:
            repo_dir = os.path.abspath(args.repo_dir)

            ensure_git_repo(repo_dir, REPO_URL)

            if not args.no_update:
                update_tags(repo_dir)
        else:
            temp_dir = tempfile.mkdtemp(prefix="onnxruntime_tags_")
            repo_dir = temp_dir

            ensure_git_repo(repo_dir, REPO_URL)

        tags = list_semver_tags(repo_dir)

        if not tags:
            print("No semver tags found (vX.Y.Z)")
            return 1

        rows = inspect_tags_with_progress(
            repo_dir=repo_dir,
            tags=tags,
            show_progress=not args.no_progress,
        )

        print_rows(rows)

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130

    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
