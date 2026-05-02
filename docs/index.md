# IDet Documentation

**IDet** is a fast, production-oriented CPU-only C++ library for image detection pipelines. This documentation is split into focused pages so each integration path is easier to read and maintain.

| Page | What it covers |
|:---|:---|
| [Requirements](requirements.md) | Host dependencies for Linux/macOS and project scope |
| [Build & Install](build.md) | Toolchain profiles, ORT setup, Meson options, build/test/install flow |
| [Model Zoo](model-zoo.md) | Supported model families, conversion notes, compatibility rules |
| [Command-line Options](cli.md) | Actual `idet_app` flags, defaults, validation rules |
| [Quick Start](quick-start.md) | Text / face / cloth scripts and direct smoke commands |
| [C++ Integration Guide](integration.md) | Blocking API, hot-loop worker, image lifetime, fixed-shape IOBinding |
| [Performance Guide](performance.md) | Runtime policy, benchmark output, tuning, IOBinding, tiling and NMS |
| [Troubleshooting & FAQ](troubleshooting.md) | Common failures, ORT ABI mismatch, FAQ |
| [Doxygen API Reference](doxygen.md) | Generate and browse local API HTML documentation |
