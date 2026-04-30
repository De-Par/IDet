# Quick Start

[Back to README](../README.md) | [Documentation index](index.md) | Previous: [Command-line Options](cli.md) | Next: [C++ Integration](integration.md)

**IDet** ships a **demo CLI app** (`idet_app`) that supports several detection modes: **Text**, **Face**, **Cloth**. Each mode is launched by its own wrapper shell script, where you can override the default behavior by changing parameters if desired.

> ⚠️ **Warn:** The images below are for **illustration only** and do not reflect detection quality. Actual results depend on many factors, including the chosen model, input resolution, and pre-/post-processing settings.


## Build and Test

```bash
source toolchain/activate.sh
scripts/build.sh force -- -Didet_libtype=shared -Dbuild_tests=true
scripts/run_tests.sh
```


## Text Detection

Basic single-shot detection:

```bash
scripts/run_idet_text.sh
```

<p align="center">
  <img src="assets/single_text_mode.png" alt="single_text_mode" width="70%">
</p>

Detection with tiling:

```bash
scripts/run_idet_text.sh tile
```

<p align="center">
  <img src="assets/tiled_text_mode.png" alt="tiled_text_mode" width="70%">
</p>


## Face Detection

Basic single-shot detection:

```bash
scripts/run_idet_face.sh
```

<p align="center">
  <img src="assets/single_face_mode.png" alt="single_face_mode" width="70%">
</p>

Detection with tiling:

```bash
scripts/run_idet_face.sh tile
```

<p align="center">
  <img src="assets/tiled_face_mode.png" alt="tiled_face_mode" width="70%">
</p>


## Cloth Detection

The bundled Fashionpedia YOLO model is fixed-shape and expects `640x640` input tensors. The wrapper keeps that shape for both single-shot and tiled runs.

Basic single-shot detection:

```bash
scripts/run_idet_cloth.sh
```

Detection with tiling:

```bash
scripts/run_idet_cloth.sh tile
```


## Direct Smoke Commands

These commands are useful for CI-style smoke runs because they disable drawing/dumping and run only one benchmark iteration.

```bash
build_gcc_perf/src/app/idet/idet_app \
    --mode text \
    --model assets/models/paddleocr/ch_ppocr_v2_det.onnx \
    --image assets/images/text/small.png \
    --is_draw 0 --is_dump 0 \
    --bench_iters 1 --warmup_iters 1 \
    --runtime_policy 0 --verbose 0

build_gcc_perf/src/app/idet/idet_app \
    --mode face \
    --model assets/models/scrfd/scrfd_500m_bnkps.onnx \
    --image assets/images/face/small.png \
    --is_draw 0 --is_dump 0 \
    --bench_iters 1 --warmup_iters 1 \
    --runtime_policy 0 --verbose 0

build_gcc_perf/src/app/idet/idet_app \
    --mode cloth \
    --model assets/models/yolo/yolov8n-fashionpedia-1.onnx \
    --image assets/images/cloth/small.png \
    --is_draw 0 --is_dump 0 \
    --bench_iters 1 --warmup_iters 1 \
    --runtime_policy 0 --verbose 0 \
    --bind_io 1 --fixed_hw 640x640
```

🔝 [Back to top](#quick-start)
