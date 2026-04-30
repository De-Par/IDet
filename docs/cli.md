# Command-line Options

[Back to README](../README.md) | [Documentation index](index.md) | Previous: [Model Zoo](model-zoo.md) | Next: [Quick Start](quick-start.md)

These flags belong to the **demo CLI application** (`idet_app`) that links against the **IDet library**. Use the CLI for quick sanity checks, visualization, and benchmarking; in production you typically integrate the library directly via its C++ API.

> 💡 **Note:** The parser accepts both `--flag value` and `--flag=value`.


## Required / Model Input

| Flag | Type | Default | Mode | Description |
|:---|:---:|:---:|:---:|:---|
| `--model` | STR | — | All | ONNX model path. Optional only when the matching embedded model is available |
| `--mode` | STR | — | All | Detector mode: `text` \| `face` \| `cloth` |
| `--image` | STR | — | All | Input image path |


## Generic

| Flag | Type | Default | Mode | Description |
|:---|:---:|:---:|:---:|:---|
| `--is_draw` | 0\|1 | `1` | All | Draw detections on image |
| `--is_dump` | 0\|1 | `1` | All | Write detections to stdout |
| `--output` | STR | `result.png` | All | Output image path (when `--is_draw=1`) |
| `--verbose` | 0\|1 | `0` | All | Verbose logging |


## Inference

| Flag | Type | Default | Mode | Description |
|:---|:---:|:---:|:---:|:---|
| `--bin_thresh` | F | `0.3` | Text | Binarization threshold |
| `--box_thresh` | F | `0.5` | All | Box score / confidence threshold |
| `--unclip` | F | `1.0` | Text | Unclip ratio |
| `--max_img_size` | N | `960` | All | Max side length for non-tiling inference |
| `--min_roi_size_w` | N | `5` | All | Minimal ROI width; `0` disables width filtering |
| `--min_roi_size_h` | N | `5` | All | Minimal ROI height; `0` disables height filtering |
| `--tiles_rc` | RxC | `off` | All | Enable tiling grid (e.g. `2x2`, `3x4`). Disable: `off`\|`no`\|`0` |
| `--tile_overlap` | F | `0.1` | All | Tile overlap fraction; valid range `[0, 1)` |
| `--nms_iou` | F | `0.3` | All | NMS IoU threshold; valid range `[0, 1]` |
| `--use_fast_iou` | 0\|1 | task default | All | Fast AABB IoU option for NMS / overlap checks |
| `--sigmoid` | 0\|1 | `0` | All | Apply sigmoid on output map (useful if model outputs logits) |
| `--bind_io` | 0\|1 | `0` | All | Use ORT I/O binding (buffer reuse) |
| `--fixed_hw` | HxW | `off` | All | Fixed input size (e.g. `480x480`). Disable: `off`\|`no`\|`0` |

> 💡 **Note:** `DetectorConfig::setup()` sets task defaults. In particular, face and cloth use fast AABB IoU by default; text uses polygon IoU by default.


## Runtime

| Flag | Type | Default | Mode | Description |
|:---|:---:|:---:|:---:|:---|
| `--threads_intra` | N | `1` | All | ORT intra-op threads (inside operators) |
| `--threads_inter` | N | `1` | All | ORT inter-op threads (between graph nodes) |
| `--tile_omp` | N | `1` | All | OpenMP threads for tiling |
| `--runtime_policy` | 0\|1 | `1` | All | Setup runtime policy (CPU/mem binding + OpenCV suppression) |
| `--soft_mem_bind` | 0\|1 | `1` | All | Best-effort memory locality (when supported) |
| `--suppress_opencv` | 0\|1 | `1` | All | Limit OpenCV global thread count to 1 |


## Benchmark

| Flag | Type | Default | Mode | Description |
|:---|:---:|:---:|:---:|:---|
| `--bench_iters` | N | `100` | All | Benchmark iterations; must be positive |
| `--warmup_iters` | N | `20` | All | Warmup iterations; may be `0` |


## Help

| Flag | Type | Default | Mode | Description |
|:---|:---:|:---:|:---:|:---|
| `-h`, `--help` | — | — | — | Show usage message |


## Output Coordinates

Each detection is reported as a quadrilateral using four vertices in TL → TR → BR → BL order:

```text
x0,y0 x1,y1 x2,y2 x3,y3
```

🔝 [Back to top](#command-line-options)
