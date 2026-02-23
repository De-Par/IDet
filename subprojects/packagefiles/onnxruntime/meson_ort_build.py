import os
import re
import sys
import shutil
import hashlib
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


LOG_PREFIX = "[onnxruntime-wrap]"


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    log(" ".join(map(str, cmd)))
    log(f"env CC  = {os.environ.get('CC')}")
    log(f"env CXX = {os.environ.get('CXX')}")
    subprocess.check_call(list(map(str, cmd)), cwd=str(cwd) if cwd else None)


def run_capture(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(list(map(str, cmd)), stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def join_flags(*chunks) -> str:
    """Join compiler flags from lists/strings preserving order."""
    out: List[str] = []
    for c in chunks:
        if not c:
            continue
        if isinstance(c, (list, tuple)):
            out.extend([str(x) for x in c if str(x).strip()])
        else:
            out.extend([x for x in str(c).split() if x.strip()])
    return " ".join(out)


def is_apple_clang(cxx_bin: str) -> bool:
    if not cxx_bin:
        return False
    ver = run_capture([cxx_bin, "--version"])
    return "Apple clang" in ver


def compiler_kind(cxx_bin: str) -> str:
    if not cxx_bin:
        return "unknown"
    ver = run_capture([cxx_bin, "--version"]).lower()
    if "apple clang" in ver:
        return "apple-clang"
    if "clang" in ver:
        return "clang"
    if "gcc" in ver or "g++" in ver:
        return "gcc"
    return "unknown"


def _candidate_dirs_from_search_dirs(cxx_bin: str) -> List[Path]:
    out = run_capture([cxx_bin, "-print-search-dirs"])
    if not out:
        return []
    dirs: List[Path] = []
    for line in out.splitlines():
        if not line.startswith("libraries:"):
            continue
        rhs = line.split("=", 1)[1] if "=" in line else ""
        for part in rhs.split(":"):
            part = part.strip()
            if not part:
                continue
            p = Path(part)
            if p.exists():
                dirs.append(p)
    return dirs


def _candidate_dirs_from_compiler_prefix(cxx_bin: str) -> List[Path]:
    try:
        cxx_path = Path(cxx_bin).resolve()
    except Exception:
        return []

    dirs: List[Path] = []
    if cxx_path.name.startswith("clang") or cxx_path.name in ("g++", "gcc"):
        prefix = cxx_path.parent.parent
        guesses = [
            prefix / "lib",
            prefix / "lib" / "c++",
            prefix / "lib" / "unwind",
        ]
        dirs.extend([p for p in guesses if p.exists()])
    return dirs


def detect_runtime_lib_dir(cxx_bin: str, libnames: List[str]) -> Optional[Path]:
    if not cxx_bin:
        return None

    # 1) Preferred: ask compiler directly.
    for libname in libnames:
        cand = run_capture([cxx_bin, "-print-file-name=" + libname])
        if not cand:
            continue
        p = Path(cand)
        if p.is_absolute() and p.exists():
            return p.parent

    # 2) Search in compiler-reported library dirs.
    dirs = _candidate_dirs_from_search_dirs(cxx_bin) + _candidate_dirs_from_compiler_prefix(cxx_bin)
    seen = set()
    for d in dirs:
        dr = str(d.resolve())
        if dr in seen:
            continue
        seen.add(dr)
        for libname in libnames:
            if (d / libname).exists():
                return d
    return None


def parse_extra_defines(extra: str) -> List[str]:
    """Parse semicolon-separated KEY=VALUE items."""
    extra = (extra or "").strip()
    if not extra:
        return []
    parts = [p.strip() for p in extra.split(";") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad ort_cmake_extra entry (need KEY=VALUE): {p}")
    return parts


def natural_version_key(p: Path):
    nums = re.findall(r"\d+", p.name)
    return tuple(int(x) for x in nums) if nums else ()


def ensure_symlink(link_path: Path, target_dir: Path) -> None:
    """Force link_path to be a symlink to target_dir (directory)."""
    link_path = Path(link_path)
    target_dir = Path(target_dir)

    link_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve current link if present
    if link_path.is_symlink():
        try:
            cur = (link_path.parent / os.readlink(link_path)).resolve()
        except OSError:
            cur = None
        if cur and cur == target_dir.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        if link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()

    link_path.symlink_to(target_dir, target_is_directory=True)
    log(f"Symlink {link_path} -> {target_dir}")


def ensure_soname_symlink(lib_dir: Path) -> None:
    """
    Ensure libonnxruntime.so (or .dylib) exists. ORT sometimes installs only versioned soname.
    """
    lib_dir = Path(lib_dir)

    if sys.platform == "darwin":
        want = lib_dir / "libonnxruntime.dylib"
        patterns = ["libonnxruntime*.dylib"]
    else:
        want = lib_dir / "libonnxruntime.so"
        patterns = ["libonnxruntime.so.*", "libonnxruntime.so"]

    if want.exists() or want.is_symlink():
        return

    candidates: List[Path] = []
    for pat in patterns:
        for p in lib_dir.glob(pat):
            if p.name == want.name:
                continue
            if p.suffix == ".a":
                continue
            if p.is_file() or p.is_symlink():
                candidates.append(p)

    if not candidates:
        listing = "\n  ".join(sorted(x.name for x in lib_dir.glob("*")))
        raise FileNotFoundError(
            f"Shared build requested but no candidates found for {want.name} in {lib_dir}\n"
            f"Directory contents:\n  {listing if listing else '(empty)'}"
        )

    candidates.sort(key=lambda p: (natural_version_key(p), p.name))
    chosen = candidates[-1]

    try:
        want.symlink_to(chosen.name)  # relative symlink (same dir)
        log(f"Created symlink {want.name} -> {chosen.name}")
    except Exception:
        shutil.copy2(chosen, want)
        log(f"Copied {chosen.name} -> {want.name}")


def default_cache_root() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "idet" / "onnxruntime"
    return Path.home() / ".cache" / "idet" / "onnxruntime"


def compute_toolchain_signature(
    ort_version: str,
    tests: str,
    acl: str,
    xnnpack: str,
    extra: str,
) -> str:
    """
    Signature determines ORT build/install dir. If any of these changes -> new build.
    Note: FETCHCONTENT download cache is shared per ORT version, not per signature.
    """
    parts = [
        ort_version,
        os.environ.get("CC", "").strip(),
        os.environ.get("CXX", "").strip(),
        os.environ.get("CFLAGS", "").strip(),
        os.environ.get("CXXFLAGS", "").strip(),
        str(tests),
        str(acl),
        str(xnnpack),
        (extra or "").strip(),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


def cmake_defines(
    install_dir: Path,
    fetchcontent_dir: Path,
    c_flags: str,
    cxx_flags: str,
    linker_flags: str,
    tests: str,
    acl: str,
    xnnpack: str,
) -> Dict[str, str]:

    defs: Dict[str, str] = {
        # CMake basics
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "CMAKE_POLICY_VERSION_MINIMUM": "3.5",
        "CMAKE_INSTALL_PREFIX": str(install_dir),

        # FetchContent cache (persistent!)
        "FETCHCONTENT_BASE_DIR": str(fetchcontent_dir),
        "FETCHCONTENT_UPDATES_DISCONNECTED": "ON",
        "FETCHCONTENT_TRY_FIND_PACKAGE_MODE": "NEVER",

        # Quiet logs (and avoid noisy warnings)
        "CMAKE_SUPPRESS_DEVELOPER_WARNINGS": "ON",
        "CMAKE_WARN_DEPRECATED": "OFF",
        "CMAKE_ERROR_DEPRECATED": "OFF",
        "CMAKE_MESSAGE_LOG_LEVEL": "ERROR",
        "CMAKE_VERBOSE_MAKEFILE": "OFF",

        # Toolchain flags
        "CMAKE_C_FLAGS": c_flags,
        "CMAKE_CXX_FLAGS": cxx_flags,
        "CMAKE_EXE_LINKER_FLAGS": linker_flags,
        "CMAKE_SHARED_LINKER_FLAGS": linker_flags,
        "CMAKE_MODULE_LINKER_FLAGS": linker_flags,

        # ORT toggles
        "onnxruntime_BUILD_SHARED_LIB": "ON",
        "onnxruntime_ENABLE_PYTHON": "OFF",
        "onnxruntime_ENABLE_TRAINING": "OFF",
        "onnxruntime_BUILD_APPLE_FRAMEWORK": "OFF",

        # Tests
        "onnxruntime_BUILD_UNIT_TESTS": "ON" if tests == "1" else "OFF",
        "onnxruntime_RUN_ONNX_TESTS": "ON" if tests == "1" else "OFF",

        # CPU-only EPs
        "onnxruntime_USE_CUDA": "OFF",
        "onnxruntime_USE_ROCM": "OFF",
        "onnxruntime_USE_TENSORRT": "OFF",
        "onnxruntime_USE_OPENVINO": "OFF",
        "onnxruntime_USE_COREML": "OFF",
        "onnxruntime_USE_DML": "OFF",

        # Optional: reduce build time / size
        "onnxruntime_DISABLE_ML_OPS": "ON",
        "onnxruntime_ENABLE_LTO": "ON",
        "onnxruntime_BUILD_FOR_NATIVE_MACHINE": "ON",

        # Optional EPs
        "onnxruntime_USE_ACL": "ON" if acl == "1" else "OFF",
        "onnxruntime_USE_XNNPACK": "ON" if xnnpack == "1" else "OFF",

        # Custom allocator
        "onnxruntime_USE_MIMALLOC": "OFF",
    }

    # Toolchain selection
    cc = os.environ.get("CC", "").strip()
    cxx = os.environ.get("CXX", "").strip()
    if cc:
        defs["CMAKE_C_COMPILER"] = cc
    if cxx:
        defs["CMAKE_CXX_COMPILER"] = cxx

    return defs


def is_built(install_dir: Path) -> bool:
    """Heuristic: ORT is built+installed if lib exists."""
    lib_dir = install_dir / "lib"
    if not lib_dir.exists():
        return False
    if sys.platform == "darwin":
        return any(lib_dir.glob("libonnxruntime*.dylib"))
    return any(lib_dir.glob("libonnxruntime.so*"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmake", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--build", required=True)    # meson-visible link path
    ap.add_argument("--install", required=True)  # meson-visible link path
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--acl", required=True)
    ap.add_argument("--xnnpack", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--cache-root", default="")  # will auto fetch
    ap.add_argument("--ort-ver", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-if-built", action="store_true", help="Skip build/install if ORT already installed")
    args = ap.parse_args()

    ort_version = (args.ort_ver or "unknown").strip()

    cmake_src_dir = Path(args.src) / "cmake"
    if not (cmake_src_dir / "CMakeLists.txt").exists():
        raise FileNotFoundError(f"Expected CMakeLists.txt at: {cmake_src_dir / 'CMakeLists.txt'}")

    cache_root_env = (args.cache_root or os.environ.get("ORT_CACHE_ROOT", "")).strip()
    cache_root = Path(cache_root_env) if cache_root_env else default_cache_root()

    tool_sig = compute_toolchain_signature(
        ort_version=ort_version,
        tests=args.tests,
        acl=args.acl,
        xnnpack=args.xnnpack,
        extra=args.extra,
    )

    # Persistent dirs
    persistent_build_dir = cache_root / "build" / ort_version / tool_sig
    persistent_install_dir = cache_root / "install" / ort_version / tool_sig
    fetchcontent_dir = cache_root / "fetchcontent" / ort_version

    if args.force:
        shutil.rmtree(persistent_build_dir, ignore_errors=True)
        shutil.rmtree(persistent_install_dir, ignore_errors=True)
        log("Force rebuild: wiped cached build/install dirs")

    persistent_build_dir.mkdir(parents=True, exist_ok=True)
    persistent_install_dir.mkdir(parents=True, exist_ok=True)
    fetchcontent_dir.mkdir(parents=True, exist_ok=True)

    # Meson-visible dirs are symlinks to persistent dirs
    meson_build_link = Path(args.build)
    meson_install_link = Path(args.install)
    ensure_symlink(meson_build_link, persistent_build_dir)
    ensure_symlink(meson_install_link, persistent_install_dir)

    # Flags
    suppress_warn = [
        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-deprecated-declarations",
    ]

    suppress_werror = [
        "-Wno-error=unused-parameter",
        "-Wno-error=unused-variable",
        "-Wno-error=deprecated-declarations",
    ]

    common = [
        "-mcpu=native",
        "-mcpu=native",
        "-fno-strict-aliasing",
        "-ffast-math",
        "-fno-math-errno",
        "-fno-math-errno",
        "-ffp-contract=fast",
        "-w",
    ]

    c_flags = join_flags(common, suppress_warn, suppress_werror, os.environ.get("CFLAGS", ""))

    stdlib_compile = []
    stdlib_linker = []
    cxx_bin = os.environ.get("CXX", "").strip()
    if sys.platform == "darwin" and cxx_bin:
        kind = compiler_kind(cxx_bin)

        if kind in ("clang", "apple-clang"):
            libcxx_dir = detect_runtime_lib_dir(cxx_bin, ["libc++.dylib", "libc++.1.dylib", "libc++.1.0.dylib"])
            if libcxx_dir:
                d = str(libcxx_dir)
                stdlib_compile = ["-stdlib=libc++"]
                stdlib_linker = ["-stdlib=libc++", "-L" + d, "-Wl,-rpath," + d]
                log(f"Detected clang++ and libc++ at: {d}")
            else:
                log("Detected clang++ but libc++.dylib path was not resolved")

        elif kind == "gcc":
            libstdcpp_dir = detect_runtime_lib_dir(
                cxx_bin, ["libstdc++.dylib", "libstdc++.6.dylib", "libstdc++.a"]
            )
            if libstdcpp_dir:
                d = str(libstdcpp_dir)
                stdlib_linker = ["-L" + d, "-Wl,-rpath," + d]
                log(f"Detected g++ and libstdc++ at: {d}")
            else:
                log("Detected g++ but libstdc++.dylib path was not resolved")

    cxx_flags = join_flags(common, suppress_warn, suppress_werror, stdlib_compile, os.environ.get("CXXFLAGS", ""))
    linker_flags = join_flags(stdlib_linker, os.environ.get("LDFLAGS", ""))

    defs = cmake_defines(
        install_dir=persistent_install_dir,
        fetchcontent_dir=fetchcontent_dir,
        c_flags=c_flags,
        cxx_flags=cxx_flags,
        linker_flags=linker_flags,
        tests=args.tests,
        acl=args.acl,
        xnnpack=args.xnnpack,
    )

    # CMake configure command
    cmake_cmd: List[str] = [
        args.cmake,
        "-Wno-deprecated",
        "-Wno-dev",
        "-S", str(cmake_src_dir),
        "-B", str(persistent_build_dir),
    ]
    if shutil.which("ninja"):
        cmake_cmd += ["-G", "Ninja"]
    for k, v in defs.items():
        cmake_cmd.append(f"-D{k}={v}")

    for kv in parse_extra_defines(args.extra):
        cmake_cmd.append("-D" + kv)

    # Build/install
    if args.skip_if_built and is_built(persistent_install_dir):
        log(f"Skip: already installed at {persistent_install_dir}")
    else:
        run(cmake_cmd)
        run([
            args.cmake, "--build", str(persistent_build_dir), "--config", "Release",
            "--parallel", str(os.cpu_count() or 4),
        ])
        run([args.cmake, "--install", str(persistent_build_dir), "--config", "Release"])

    lib_dir = persistent_install_dir / "lib"
    if not lib_dir.exists():
        raise FileNotFoundError(f"Install libdir not found: {lib_dir}")

    ensure_soname_symlink(lib_dir)
    Path(args.stamp).touch()

    log(f"ORT={ort_version} sig={tool_sig}")
    log(f"build={persistent_build_dir}")
    log(f"install={persistent_install_dir}")


if __name__ == "__main__":
    main()
    
