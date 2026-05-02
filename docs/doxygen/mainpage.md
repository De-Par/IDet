# IDet

@htmlonly
<span class="idet-hero">
  <span class="idet-hero-copy">
    <span class="idet-eyebrow">CPU-only C++ detection runtime</span>
    <span class="idet-hero-title">IDet API Reference</span>
    <span class="idet-lead">
      Fast image detection pipelines on top of ONNX Runtime: text, face, and cloth
      detectors with explicit control over tiling, IOBinding, OpenMP, CPU affinity,
      and image ownership.
    </span>
    <span class="idet-pill-row">
      <span class="idet-pill">C++17</span>
      <span class="idet-pill">ONNX Runtime CPU</span>
      <span class="idet-pill">OpenMP-aware</span>
      <span class="idet-pill">Pipeline-ready</span>
    </span>
  </span>
  <img class="idet-main-logo" src="assets/idet_logo.png" alt="IDet logo" />
</span>
@endhtmlonly

## Start Here

@htmlonly
<span class="idet-card-grid">
  <span class="idet-card">
    <span class="idet-card-title">Build and install</span>
    <span class="idet-card-copy">Toolchain profiles, Meson options, dependency setup, and installation flow.</span>
    <a href="md_docs_2build.html">Open build guide</a>
  </span>
  <span class="idet-card">
    <span class="idet-card-title">Integrate in C++</span>
    <span class="idet-card-copy">Blocking detector calls, hot-loop worker usage, image lifetime, and fixed-shape I/O.</span>
    <a href="md_docs_2integration.html">Open integration guide</a>
  </span>
  <span class="idet-card">
    <span class="idet-card-title">Tune performance</span>
    <span class="idet-card-copy">Runtime policy, thread budgets, OpenMP tiling, IOBinding, and benchmark interpretation.</span>
    <a href="md_docs_2performance.html">Open performance guide</a>
  </span>
  <span class="idet-card">
    <span class="idet-card-title">Browse API</span>
    <span class="idet-card-copy">Public API, engine contracts, algorithms, platform helpers, examples, and tests.</span>
    <a href="topics.html">Open API topics</a>
  </span>
</span>
@endhtmlonly

## Supported Pipelines

| Task | Engine family | Main API path |
|:---|:---|:---|
| Text detection | DBNet / DBNet++ / PP-OCR-style exports | `idet::Detector`, `idet::engine::DBNet` |
| Face detection | SCRFD family | `idet::Detector`, `idet::engine::SCRFD` |
| Cloth detection | YOLO family | `idet::Detector`, `idet::engine::YOLO` |

## Runtime Model

IDet keeps the application boundary explicit:

- pass pixels through `idet::ImageView` / `idet::Image`;
- create a configured `idet::Detector`;
- use `idet::DetectorWorker` for hot-loop pipelines;
- call `idet::setup_runtime_policy` only when process-global CPU/OpenMP/OpenCV policy changes are acceptable.

The generated reference uses Doxygen's default HTML output with a minimal IDet-specific stylesheet. Source files for the Doxygen setup live under `docs/doxygen`; generated HTML is written to `docs/doxygen/html`.
