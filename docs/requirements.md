# Requirements

[Back to README](../README.md) | [Documentation index](index.md) | Next: [Build & Install](build.md)

**IDet** is intentionally a **low-level inference toolkit**, not a full OCR stack or face recognition framework.

## Project Scope

### In scope
- ONNX Runtime-based CPU detector execution
- pre-processing / post-processing for supported detector families
- tiled inference and result stitching
- performance benchmarking and runtime policy control
- library integration + CLI demo tooling

### Out of scope
- OCR text recognition and language models
- ROI recognition / embeddings / tracking
- dataset labeling / training pipelines
- GUI applications


## Host Requirements

| Component | Minimum | Scope | Requirement | Notes |
|:---|:---:|:---:|:---:|:---|
| **C++ toolchain** | **C++17** | Build | 🟢 | Any GCC/Clang that fully supports C++17 |
| **Meson** | — | Build | 🟢 | Primary build system (typically uses Ninja as backend) |
| **pkg-config** | — | Build | 🟢 | Used to discover system dependencies (OpenCV / ORT, etc.) |
| **OpenCV** | **3.0+** | Runtime | 🟢 | Modules: `core`, `imgproc`, `imgcodecs` |
| **ONNX Runtime (CPU / MLAS)** | — | Runtime | 🟢 | Can be provided via system install, manual paths, or Meson wrap |
| **CMake** | **≥ 3.18** | Build | 🟡 | Needed only if ONNX Runtime is built from sources / via wrap (depends on ORT version) |
| **OpenMP runtime** | — | Runtime | 🟡 | Recommended for tiling / parallelism (Linux: often via `libomp-dev` for Clang; MacOS: `libomp`) |
| **NUMA** | — | Runtime | 🔵 | Optional; Linux-only (multi-socket topology / affinity; typically `libnuma-dev`) |
| **Doxygen + Graphviz** | — | Docs | 🔵 | Optional; required only for local API HTML generation |

> **Legend:** required (🟢), recommended / conditional (🟡), optional (🔵)


## Linux (Ubuntu / Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential ninja-build meson pkg-config python3 \
    python3-pip libopencv-dev libnuma-dev libomp-dev git \
    doxygen graphviz
```


## MacOS (Apple Silicon / Intel)

```bash
brew install \
    meson ninja opencv onnxruntime \
    libomp cmake python llvm doxygen graphviz
```

> 💡 **Note:** Create a **virtual environment** for tooling in project root directory (optional but recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```
