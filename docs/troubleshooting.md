# Troubleshooting & FAQ

[Back to README](../README.md) | [Documentation index](index.md) | Previous: [Performance Guide](performance.md)


## Troubleshooting

### `onnxruntime_cxx_api.h: No such file or directory`

Make sure ONNX Runtime is installed and headers are visible to Meson. Typical options:

```bash
scripts/build.sh setup -- -Donnxruntime_system=true
scripts/build.sh setup -- -Donnxruntime_inc="/usr/local/include/onnxruntime" -Donnxruntime_lib="/usr/local/lib"
scripts/build.sh setup -- -Donnxruntime_system=false
```


### `Unexpected output shape`

For DBNet-style text maps, IDet supports:

```text
[1,1,H,W]
[1,H,W,1]
[1,H,W]
[H,W]
```

If your model differs, verify the export and final layers.

If outputs are logits rather than values in `[0,1]`, pass:

```bash
--sigmoid 1
```


### Performance flatlines when increasing threads

Likely causes:

- CPU oversubscription;
- ONNX Runtime and OpenMP competing for the same CPUs;
- process restricted to a smaller cpuset than expected;
- memory locality / NUMA effects;
- ASan/UBSan build used for timing.

Check:

```bash
taskset -pc $$
cat /proc/sys/kernel/numa_balancing
```

If the report shows only a few available CPUs, tune for that CPU mask, not for the full machine.

Practical fixes:

- lower `--threads_intra` to `1–2`;
- keep `--threads_inter 1`;
- increase `--tile_omp` only while staying within the available CPU mask;
- compare configurations by `p99_ms`, not only by `p50_ms`;
- use a perf/release build without sanitizers.


### p99 latency is unstable

Common causes:

- automatic NUMA balancing;
- remote NUMA memory access;
- background system load;
- too many OpenMP / ORT threads;
- `--is_dump 1`, `--is_draw 1`, or `--verbose 1` accidentally enabled;
- debug/sanitizer build used by mistake.

Check:

```bash
cat /proc/sys/kernel/numa_balancing
taskset -pc $$
numactl --hardware || true
```

For controlled benchmark runs, try temporarily disabling automatic NUMA balancing:

```bash
sudo sysctl kernel.numa_balancing=0
```

Restore it after benchmarking if needed:

```bash
sudo sysctl kernel.numa_balancing=1
```


### Only a few CPU IDs are available

Example:

```text
Total logical:  96
Available CPU IDs: 48-49 (2)
```

This means the process is restricted by affinity, cgroups, a job scheduler, or manual `taskset`.

Do not interpret such results as full-machine performance.

Options:

- lower `--max-threads`;
- lower `--threads_intra` / `--tile_omp`;
- run outside the restrictive cpuset;
- explicitly document the available CPU IDs in benchmark reports.


### `SPE is not configured`

CCA or other Arm profiling tools may print:

```text
[SPE]: The system is not configured with Statistical Profiling Extension.
```

This is a profiling limitation, not an IDet runtime error.

The application can still be benchmarked with its own latency metrics and available CPU counters.

If `dmesg` contains:

```text
ACPI: SPE must be homogeneous
```

then the kernel did not expose SPE because of platform firmware or topology constraints.


### `Virtual environment check failed: SMBIOS ... permission denied`

Example:

```text
SMBIOS: failed to open stream: open /sys/firmware/dmi/tables/smbios_entry_point: permission denied
```

This is usually emitted by a profiling or system inventory tool. It means the tool could not read SMBIOS metadata due to permissions.

It does not necessarily affect `idet_app` execution.


### Boxes are weak or too many false positives

Tune thresholds:

```bash
--bin_thresh
--box_thresh
--unclip
--nms_iou
```

For text / face, start with:

```bash
--bin_thresh 0.3 --box_thresh 0.5 --nms_iou 0.3
```

For YOLO-style cloth models, start with:

```bash
--box_thresh 0.01 --nms_iou 0.5
```

If objects are small:

- increase `--max_img_size`;
- or use tiling with `--tile_overlap 0.10–0.20`;
- validate quality with ROI area metrics, not only `dets_n`.


### Tiling produces duplicate boxes

This usually happens near tile borders.

Try:

- increase or decrease `--nms_iou`;
- reduce `--tile_overlap`;
- verify coordinate remapping after tile stitching;
- inspect printed `Quads`.

For text / face, typical NMS IoU is around:

```text
0.2–0.4
```

For cloth / YOLO-style models, typical NMS IoU is around:

```text
0.5
```


### `DetectorConfig: tiles_dim must be > 0`

Do not pass disabled tiling as:

```bash
--tiles_rc off
```

Some app versions may parse this as `0x0`.

For single-shot mode, omit `--tiles_rc` entirely.

Good single-shot command style:

```bash
idet_app --mode text --model model.onnx --image image.png --fixed_hw 512x960
```


### `--bind_io 1` does not improve latency

Check that fixed shape is enabled:

```bash
--bind_io 1 --fixed_hw HxW
```

I/O Binding is a fixed-shape contract in IDet. If the shape changes every frame, the binding context must be rebuilt and the benefit may disappear.

Also verify that the workload is not dominated by preprocessing, post-processing, or NMS rather than ORT tensor allocation.


### ORT API version mismatch

Error example:

```text
The request api version [N] is not available, only api versions [1, M] are supported in this build.
```

This usually means ABI mismatch between the ONNX Runtime headers IDet was compiled against and the `libonnxruntime` resolved at runtime.

IDet probes down from the headers' `ORT_API_VERSION` to a known-good minimum and binds the first supported version via `Ort::InitApi`, so a compatible runtime keeps working where possible.

To fix the underlying mismatch:

- ensure exactly one ORT is visible to the loader:
  - `LD_LIBRARY_PATH`;
  - RPATH;
  - system library paths;
- rebuild with the bundled wrap:

```bash
scripts/build.sh force -- -Donnxruntime_system=false
```

- or upgrade the installed runtime so its API version is at least as new as the headers.


### ONNX model contains `com.microsoft.nchwc` ops

If a portable ONNX model contains nodes such as:

```text
com.microsoft.nchwc::Conv
com.microsoft.nchwc::ReorderInput
com.microsoft.nchwc::ReorderOutput
```

then it is likely an ORT-optimized internal graph, not a portable deployment ONNX.

Use a raw or properly exported ONNX model for normal deployment.

Quick check:

```bash
python3 - <<'PY'
import onnx
from collections import Counter

path = "model.onnx"
model = onnx.load(path)
counter = Counter((node.domain, node.op_type) for node in model.graph.node)

bad = [(k, v) for k, v in counter.items() if "nchwc" in k[0].lower()]
print("BAD_NCHWC =", bad)
PY
```


## FAQ

**Q:** Can I speed up by feeding grayscale instead of RGB?

**A:** Not unless the model itself is changed to accept `[1,1,H,W]`. Feeding one channel into `[1,3,H,W]` does not reduce the first convolution's expected input contract. Changing the first conv to 1-channel may reduce a small part of compute, but accuracy can drop and the model must be retrained or carefully adapted.

---

**Q:** How are coordinates printed?

**A:** Each detection is printed as a quadrilateral:

```text
x0,y0 x1,y1 x2,y2 x3,y3
```

Built-in engines normalize points to:

```text
TL -> TR -> BR -> BL
```

---

**Q:** Does the tool support dynamic sizes?

**A:** Yes. The dynamic path uses `--max_img_size`. For best latency and zero re-binding, use:

```bash
--bind_io 1 --fixed_hw HxW
```

Binding is treated as a fixed-shape contract.

---

**Q:** Can OpenCV be used directly in my app?

**A:** Yes. Keep OpenCV at the app boundary and wrap `cv::Mat` as `idet::ImageView`. Library internals keep OpenCV behind adapters where possible. See [C++ Integration Guide](integration.md).

---

**Q:** Should I rank configurations by FPS or p99?

**A:** For interactive and near-real-time pipelines, rank by `p99_ms` first. FPS from median latency can hide tail-latency spikes.

---

**Q:** Should I use debug builds for performance results?

**A:** No. Use `release` or `perf` profiles. Sanitizers and debug instrumentation can heavily distort latency, memory traffic, and profiler hotspots.

---

**Q:** Why does a faster configuration sometimes have worse ROI quality?

**A:** Lower input resolution or aggressive tiling can reduce latency but may produce larger, weaker, or missing ROI boxes. Compare speed with area-based ROI metrics such as `area_recall`, `area_precision`, `area_f1`, and `extra_area_ratio`.
