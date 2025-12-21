import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd, cwd=None):
    print("[onnxruntime-wrap] " + " ".join(map(str, cmd)), flush=True)
    subprocess.check_call(list(map(str, cmd)), cwd=cwd)


def parse_extra_defs(extra: str):
    extra = (extra or "").strip()
    if not extra:
        return []
    parts = [p.strip() for p in extra.split(";") if p.strip()]
    out = []
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad ort_cmake_extra entry (need KEY=VALUE): {p}")
        out.append(p)
    return out


def ensure_shared_symlink(libdir: Path):
    # -lonnxruntime needs libonnxruntime.so (not only .so.1 / .so.1.23.2)
    so = libdir / "libonnxruntime.so"
    if so.exists():
        return
    cands = sorted(libdir.glob("libonnxruntime.so.*"))
    if not cands:
        raise FileNotFoundError(f"Shared build requested but no libonnxruntime.so* found in {libdir}")
    target = cands[-1].name
    try:
        so.symlink_to(target)
        print(f"[onnxruntime-wrap] Created symlink {so} -> {target}")
    except Exception:
        # fallback: copy
        shutil.copy2(libdir / target, so)
        print(f"[onnxruntime-wrap] Copied {target} -> {so}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmake", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--install", required=True)
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--tests", required=True)    
    ap.add_argument("--acl", required=True)      
    ap.add_argument("--xnnpack", required=True)  
    ap.add_argument("--extra", default="")

    args = ap.parse_args()

    src = Path(args.src)
    bld = Path(args.build)
    inst = Path(args.install)
    bld.mkdir(parents=True, exist_ok=True)
    inst.mkdir(parents=True, exist_ok=True)

    cmake_src = src / "cmake"
    if not (cmake_src / "CMakeLists.txt").exists():
        raise FileNotFoundError(f"Expected CMakeLists.txt at: {cmake_src / 'CMakeLists.txt'}")

    jobs = str(os.cpu_count() or 4)

    ninja = shutil.which("ninja")
    gen = ["-G", "Ninja"] if ninja else []

    fc_dir = bld / "_fc"
    fc_dir.mkdir(parents=True, exist_ok=True)

    extra_defs = parse_extra_defs(args.extra)

    defs = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "CMAKE_INSTALL_PREFIX": str(inst),

        # keep FetchContent inside build tree
        "FETCHCONTENT_BASE_DIR": str(fc_dir),
        "FETCHCONTENT_UPDATES_DISCONNECTED": "ON",

        # quiet logs
        "CMAKE_SUPPRESS_DEVELOPER_WARNINGS": "ON",
        "CMAKE_CXX_FLAGS": "-Wno-unused-parameter -Wno-unused-variable -fno-strict-aliasing -Wno-lto-type-mismatch",
        "CMAKE_C_FLAGS":   "-Wno-unused-parameter -Wno-unused-variable -fno-strict-aliasing -Wno-lto-type-mismatch",

        # tests/benchmarks (prod should be OFF)
        "onnxruntime_BUILD_UNIT_TESTS": "ON" if args.tests == "1" else "OFF",
        "onnxruntime_RUN_ONNX_TESTS":   "ON" if args.tests == "1" else "OFF",

        # core toggles
        "onnxruntime_BUILD_SHARED_LIB":      "ON",
        "onnxruntime_ENABLE_PYTHON":         "OFF",
        "onnxruntime_ENABLE_TRAINING":       "OFF",
        "onnxruntime_BUILD_APPLE_FRAMEWORK": "OFF",

        # CPU-only
        "onnxruntime_USE_CUDA":     "OFF",
        "onnxruntime_USE_ROCM":     "OFF",
        "onnxruntime_USE_TENSORRT": "OFF",
        "onnxruntime_USE_OPENVINO": "OFF",
        "onnxruntime_USE_COREML":   "OFF",
        "onnxruntime_USE_XNNPACK":  "OFF",
        "onnxruntime_USE_DML":      "OFF",

        # optional (reduce memory / time)
        "onnxruntime_DISABLE_ML_OPS":           "ON",
        "onnxruntime_ENABLE_LTO":               "ON",
        "onnxruntime_BUILD_FOR_NATIVE_MACHINE": "ON",

        # optional EPs
        "onnxruntime_USE_ACL":     "ON" if args.acl     == "1" else "OFF",
        "onnxruntime_USE_XNNPACK": "ON" if args.xnnpack == "1" else "OFF",
    }

    cmake_cmd = [args.cmake, "-S", str(cmake_src), "-B", str(bld)] + gen
    for k, v in defs.items():
        cmake_cmd.append(f"-D{k}={v}")
    for kv in extra_defs:
        cmake_cmd.append("-D" + kv)

    run(cmake_cmd)
    run([args.cmake, "--build", str(bld), "--config", "Release", "--parallel", jobs])
    run([args.cmake, "--install", str(bld), "--config", "Release"])

    libdir = inst / "lib"
    if not libdir.exists():
        raise FileNotFoundError(f"Install libdir not found: {libdir}")

    ensure_shared_symlink(libdir)
    Path(args.stamp).touch()


if __name__ == "__main__":
    main()