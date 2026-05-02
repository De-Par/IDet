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

For DBNet-style text maps, IDet supports `[1,1,H,W]`, `[1,H,W,1]`, `[1,H,W]`, and `[H,W]`. If your model differs, verify your export and final layers. If outputs are logits (not in `[0,1]`), pass `--sigmoid 1`.


### Performance flatlines when increasing threads

Likely oversubscription. Lower `--threads_intra` (ORT) to 1–2; increase `--tile_omp`, or use multiple externally scheduled `DetectorWorker` lanes with small per-lane budgets.


### Boxes are weak or too many false positives

Tune `--bin_thresh`, `--box_thresh`, `--unclip`. If model lacks final sigmoid, set `--sigmoid 1`.


### ORT API version mismatch

Error example:

```text
The request api version [N] is not available, only api versions [1, M] are supported in this build.
```

This usually means ABI mismatch between the ONNX Runtime headers IDet was compiled against and the `libonnxruntime` resolved at runtime. IDet probes down from the headers' `ORT_API_VERSION` to a known-good minimum and binds the first supported version via `Ort::InitApi`, so a compatible runtime keeps working where possible.

To fix the underlying mismatch:

- ensure exactly one ORT is visible to the loader (`DYLD_LIBRARY_PATH` / `LD_LIBRARY_PATH` / RPATH),
- rebuild with the bundled wrap via `scripts/build.sh force -- -Donnxruntime_system=false`,
- or upgrade the installed runtime so its API version is at least as new as the headers.


## FAQ

**Q:** Can I speed up by feeding grayscale instead of RGB?

**A:** Not unless the model itself is changed to accept `[1,1,H,W]`. Feeding one channel into `[1,3,H,W]` doesn’t reduce compute. Changing the first conv to 1-channel helps only a little overall; accuracy may drop.

---

**Q:** How are coordinates printed?

**A:** Each detection line on stdout: `x0,y0 x1,y1 x2,y2 x3,y3`. Built-in engines normalize points to TL → TR → BR → BL order.

---

**Q:** Does the tool support dynamic sizes?

**A:** Yes. Dynamic path uses `--max_img_size`. For best latency and zero re-binding, use `--bind_io 1` together with `--fixed_hw HxW`; binding is a fixed-shape contract.

---

**Q:** Can OpenCV be used directly in my app?

**A:** Yes. Keep OpenCV at the app boundary and wrap `cv::Mat` as `idet::ImageView`; library internals keep OpenCV behind adapters where possible. See [C++ Integration Guide](integration.md).
