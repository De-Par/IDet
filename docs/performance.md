# Performance Guide

[Back to README](../README.md) | [Documentation index](index.md) | Previous: [C++ Integration](integration.md) | Next: [Troubleshooting & FAQ](troubleshooting.md)

The **demo CLI app** (`idet_app`) can run a warmup + benchmark loop and prints latency percentiles for the selected detector configuration.

Use this guide when you need to compare detector configurations, tune CPU-only inference, or collect reproducible performance numbers for reports.

The most important latency metrics are:

- `p50_ms` — median latency;
- `p90_ms` — 90th percentile latency;
- `p95_ms` — 95th percentile latency;
- `p99_ms` — tail latency and the primary ranking metric for near-real-time use.

For real-time and near-real-time pipelines, prefer stable `p95_ms` / `p99_ms` over a low average latency.


## Runtime Policy

When runtime policy is enabled, the report may include:

- CPU topology:
  - sockets;
  - logical cores;
  - physical cores;
  - available CPU IDs;
- per-socket CPU distribution;
- affinity verification and allowed CPU mask;
- memory locality checks;
- OpenMP runtime information;
- effective OpenMP thread count;
- ONNX Runtime intra/inter-op thread configuration.

<table align="center" width="70%">
  <tr>
    <td align="center">
      <img src="assets/policy_config.png" alt="policy_config" width="100%">
      <br>
      <sub>Terminal screenshot of detected CPU topology and OpenMP runtime configuration, showing socket and core counts, available CPU IDs, libomp runtime details, and active threading settings.</sub>
    </td>
  </tr>
</table>

### Why available CPU IDs matter

On shared servers, containers, CI jobs, or manually pinned runs, the process may not have access to all machine cores.

Example:

```text
Total logical:  96
Available CPU IDs: 48-57 (10)
```

In this case, the benchmark must be interpreted relative to **10 available CPUs**, not 96 system CPUs.

A configuration with `desired_threads=16` is suspicious in this environment, because it requests more runnable work than the current CPU mask can provide.


## NUMA and Memory Locality

NUMA means **Non-Uniform Memory Access**. On multi-socket systems, each socket has local memory, and accessing memory attached to another socket may be slower and less predictable.

This matters for IDet because inference repeatedly touches:

- input image buffers;
- preprocessed tensors;
- ONNX Runtime input/output tensors;
- intermediate runtime allocations;
- post-processing buffers;
- tile buffers and stitched ROI outputs.

### Automatic NUMA balancing

Linux automatic NUMA balancing can sample memory accesses and migrate pages closer to the CPUs that use them. This is useful for general workloads, but it can add noise to controlled latency benchmarks, especially when CPU affinity and memory locality are already managed explicitly.

Check current state:

```bash
cat /proc/sys/kernel/numa_balancing
```

or:

```bash
sysctl kernel.numa_balancing
```

Typical values:

```text
0 = disabled
1 = enabled
```

For controlled benchmark runs, it is often useful to disable it temporarily:

```bash
sudo sysctl kernel.numa_balancing=0
```

After benchmarking, restore the previous value if needed:

```bash
sudo sysctl kernel.numa_balancing=1
```

Always record this value in final benchmark notes:

```text
kernel.numa_balancing = 0|1
```

### Memory locality checks

If the report includes sampled page locality, verify that pages are located on an allowed or selected NUMA node.

A stable run should have a high ratio of pages in the allowed node set, for example:

```text
summary        : valid=4096 in_allowed=4096 out_allowed=0 ratio=1
selected_node  : node=1
```

If many sampled pages are outside the expected node set, benchmark noise may increase and `p95_ms` / `p99_ms` can become less stable.


## Configuration

Effective application and detector configuration:

<table align="center" width="70%">
  <tr>
    <td align="center">
      <img src="assets/detector_config.png" alt="detector_config" width="100%">
      <br>
      <sub>Terminal screenshot of the detector and application configuration, showing the selected task and engine, input/output paths, benchmarking parameters, inference thresholds, tiling settings, and runtime threading options.</sub>
    </td>
  </tr>
</table>

Important fields:

- `task` — selected detector mode: `text`, `face`, or `cloth`;
- `engine` — selected detector backend;
- `model_path` — ONNX model path;
- `fixed_input_dim` — fixed input shape used by the detector;
- `tiles_dim` — tiling grid or disabled tiling;
- `tile_overlap` — tile overlap ratio;
- `bind_io` — whether ONNX Runtime I/O Binding is enabled;
- `ort_intra_threads` — ONNX Runtime intra-op threads;
- `ort_inter_threads` — ONNX Runtime inter-op threads;
- `tile_omp_threads` — OpenMP tile-level threads;
- `runtime_policy` — whether CPU/memory/runtime policy setup is applied.


## Benchmark Protocol

Use separate modes for latency measurement and ROI dump.

### Clean latency benchmark

Use this mode for latency tables:

```bash
--is_draw 0 --is_dump 0 --verbose 0
```

Reason:

- no drawing overhead;
- no dump overhead;
- minimal stdout overhead;
- `p50_ms` / `p90_ms` / `p95_ms` / `p99_ms` come directly from `idet_app`.

### ROI dump pass

Use this mode only when you need detection coordinates:

```bash
--is_draw 0 --is_dump 1 --verbose 0
```

Expected output:

```text
dets_n: 11
Quads:
    1 -> x0,y0 x1,y1 x2,y2 x3,y3
```

Do not mix dump-mode latency with clean benchmark latency.

### Minimum metadata for a benchmark table

Record:

```text
build profile
model path
target mode
image path
fixed_hw
tiles_rc
tile_overlap
threads_intra
threads_inter
tile_omp
desired_threads
available CPU IDs
kernel.numa_balancing
warmup_iters
bench_iters
p50_ms
p90_ms
p95_ms
p99_ms
```


## Results

The benchmark report may include:

- warmup progress;
- benchmark progress;
- latency percentiles;
- iteration count;
- estimated FPS;
- detection count and quads when dump is enabled.

<table align="center" width="70%">
  <tr>
    <td align="center">
      <img src="assets/bench_results.png" alt="bench_results" width="100%">
      <br>
      <sub>Terminal screenshot of benchmark execution and results, showing warmup and measurement progress bars together with latency statistics, percentile metrics, iteration count, estimated FPS, and basic application output.</sub>
    </td>
  </tr>
</table>


## Performance Tuning Guide

### Two levels of parallelism

IDet has two main levels of CPU parallelism:

- **OpenMP outer parallelism**
  - CLI: `--tile_omp`;
  - C++: `RuntimePolicy::tile_omp_threads`;
  - usually used to process tiles in parallel.

- **ONNX Runtime inner parallelism**
  - CLI: `--threads_intra`;
  - C++: `RuntimePolicy::ort_intra_threads`;
  - used inside model operators.

Optional inter-op parallelism:

```bash
--threads_inter N
```

For most detector workloads, start with:

```bash
--threads_inter 1
```

and tune `--threads_intra` / `--tile_omp` first.

### Avoid oversubscription

Oversubscription happens when OpenMP and ONNX Runtime together request more runnable work than the available CPU mask can provide.

A practical estimate:

```text
desired_threads = tile_omp + max(threads_intra, threads_inter)
```

If both ORT intra-op and inter-op parallelism are active, use the more conservative estimate:

```text
desired_threads = tile_omp + threads_intra + threads_inter
```

Rules of thumb:

- for single-shot inference:
  - keep `--tile_omp 1`;
  - tune `--threads_intra`;
- for tiled inference:
  - use more `--tile_omp`;
  - keep `--threads_intra` small, often `1–2`;
- do not exceed the available CPU mask;
- compare by `p99_ms`, not only by `p50_ms`.

Example:

```text
Available CPU IDs: 48-57 (10)
tile_omp = 2
threads_intra = 8
threads_inter = 1

desired_threads = 2 + max(8, 1) = 10
```

This is a reasonable upper bound for a 10-CPU cpuset.


## I/O Binding Deep-Dive

**What it is:** binding ONNX input/output tensors directly to reusable buffers.

**Why it matters:** it reduces per-frame allocation churn and improves latency stability.

Best practice:

- enable I/O Binding:

```bash
--bind_io 1
```

- always combine it with a fixed input shape:

```bash
--fixed_hw HxW
```

- with tiling, prepare enough binding contexts for tile workers;
- do not share one bound buffer concurrently across multiple tile workers;
- with `DetectorWorker`, a single-flight lane usually needs one bound context.

Good:

```bash
--bind_io 1 --fixed_hw 512x960
```

Suspicious:

```bash
--bind_io 1
```

without a fixed input shape.


## Tiling & NMS

`--tiles_rc RxC` splits the image into a grid and runs inference per tile.

Example:

```bash
--tiles_rc 2x2 --tile_overlap 0.1
```

`--tile_overlap` helps avoid cutting objects at tile borders. After stitching, polygon NMS removes duplicate boxes across tile overlaps.

Typical values:

- text / face:
  - `--nms_iou 0.2–0.4`;
- YOLO-style cloth detector:
  - `--nms_iou 0.5`.

### Tiling shape rule

For tiling candidates:

```text
fixed_hw <= tile size
```

Using a per-tile `fixed_hw` larger than the tile itself usually wastes compute because the tile is upscaled beyond the source information available in that tile.


## Thresholds

Typical starting points:

### Text / face

```bash
--bin_thresh 0.3
--box_thresh 0.5
--nms_iou 0.3
```

### Cloth / YOLO-style detector

```bash
--box_thresh 0.01
--nms_iou 0.5
```

For small objects:

- increase `--max_img_size`;
- or use tiling with overlap `0.10–0.20`;
- then validate quality with ROI area metrics, not only `dets_n`.


## ROI Quality vs Latency

A fast detector configuration is not always the best detector configuration. It must preserve enough ROI quality.

Recommended area-based metrics:

```text
area_recall      = intersection_area / reference_area
area_precision   = intersection_area / candidate_area
area_f1          = 2 * recall * precision / (recall + precision)
extra_area_ratio = (candidate_area - intersection_area) / reference_area
```

Interpretation:

- `area_recall` — how much reference ROI area is covered;
- `area_precision` — how much candidate ROI area is useful;
- `area_f1` — balanced area-quality score;
- `extra_area_ratio` — how much unnecessary ROI area is added.

Recommended starting filters for text:

```bash
--min-coverage 0.85
--min-area-precision 0.65
--min-area-f1 0.75
--max-extra-area-ratio 0.50
```

This avoids selecting very fast but overly coarse ROI configurations.


## Suggested Benchmark Workflow

### 1. Build a perf profile

```bash
source toolchain/activate.sh gcc-perf
./scripts/build.sh f
```

Avoid ASan/UBSan builds for final performance measurements.

### 2. Record system state

```bash
lscpu
numactl --hardware || true
cat /proc/sys/kernel/numa_balancing
taskset -pc $$
```

### 3. Optional: disable automatic NUMA balancing

```bash
sudo sysctl kernel.numa_balancing=0
```

### 4. Run grid search

```bash
python tools/grid_search.py \
  --exe build_gcc_perf/src/app/idet/idet_app \
  --model assets/models/paddleocr/ch_ppocr_v2_det.onnx \
  --image assets/images/text/medium.png \
  --target text \
  --out ./grid/result_text.csv \
  --mode both \
  --min-coverage 0.85 \
  --min-area-precision 0.65 \
  --min-area-f1 0.75 \
  --max-extra-area-ratio 0.50
```

### 5. Re-run top candidates with more iterations

```bash
--bench-iters 300 --warmup-iters 50
```

### 6. Restore automatic NUMA balancing if needed

```bash
sudo sysctl kernel.numa_balancing=1
```


## Reporting Template

```text
Hardware:
  CPU:
  architecture:
  sockets:
  available CPU IDs:
  NUMA nodes:
  kernel.numa_balancing:

Software:
  build profile:
  compiler:
  ONNX Runtime:
  OpenMP runtime:

Model:
  target:
  model path:
  fixed_hw:
  tiles_rc:
  tile_overlap:

Runtime:
  bind_io:
  runtime_policy:
  soft_mem_bind:
  cpu_placement:
  suppress_opencv:
  threads_intra:
  threads_inter:
  tile_omp:
  desired_threads:

Benchmark:
  warmup_iters:
  bench_iters:
  p50_ms:
  p90_ms:
  p95_ms:
  p99_ms:

Quality:
  reference mode:
  area_recall:
  area_precision:
  area_f1:
  extra_area_ratio:
```


## Quick Checks

If `p99_ms` is unexpectedly high:

- verify that `--is_draw 0 --is_dump 0 --verbose 0` is used;
- verify that the build profile is not ASan/UBSan;
- check available CPU IDs;
- check `kernel.numa_balancing`;
- reduce `--threads_intra` or `--tile_omp`;
- confirm that `--bind_io 1` is paired with `--fixed_hw HxW`;
- compare single-shot and tiling with the same model and image.
