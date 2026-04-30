# C++ Integration Guide

This guide focuses on embedding IDet into a larger C++ pipeline where the application owns
resource policy, scheduling, image lifetime, and backpressure.

Checked examples live in [`examples/`](../examples):

- [`examples/sync_detector.cpp`](../examples/sync_detector.cpp) - direct blocking inference.
- [`examples/hot_loop_worker.cpp`](../examples/hot_loop_worker.cpp) - single-flight async worker for a hot loop.

Build them with:

```bash
scripts/build.sh force -- -Dbuild_examples=true
```

The exact output directory depends on the active toolchain profile. Run the produced binaries from
the repository root so the default `assets/...` paths resolve.

## Runtime Ownership

IDet separates local detector configuration from process-global runtime setup:

- `DetectorConfig::runtime` is consumed by detectors and engines.
- `setup_runtime_policy()` is explicit and should be called by the application at a controlled
  boundary, typically process startup or worker-pool setup.
- `setup_runtime_policy()` may touch process-global OpenCV/OpenMP state, so `DetectorWorker` never
  calls it implicitly.

Typical policy for a pipeline lane:

```cpp
idet::RuntimePolicy policy{};
policy.ort_intra_threads = 1;
policy.ort_inter_threads = 1;
policy.tile_omp_threads = 1;
policy.suppress_opencv = true;

const idet::Status status = idet::setup_runtime_policy(policy, false);
if (!status.ok()) {
    // log status.message and abort worker setup
}
```

Use small ORT thread counts when the surrounding application already has parallel work. Scale
through multiple lanes, tiles, or app-level scheduling instead of letting every dependency create
its own large thread team.

## Blocking Detector

Use the blocking API when the caller thread is allowed to spend time inside inference:

```cpp
idet::DetectorConfig config =
    idet::DetectorConfig::setup(idet::Task::Text, "assets/models/paddleocr/ch_ppocr_v2_det.onnx");
config.verbose = false;
config.runtime = policy;

auto detector_result = idet::Detector::create(config);
if (!detector_result.ok()) {
    // detector_result.status().message
}
idet::Detector detector = std::move(detector_result).value();

auto image_result = idet::load_image("assets/images/text/small.png", idet::PixelFormat::BGR_U8);
if (!image_result.ok()) {
    // image_result.status().message
}
idet::Image image = std::move(image_result).value();

auto detections = detector.detect(image);
if (!detections.ok()) {
    // detections.status().message
}
```

This is the simplest model and is often enough for batch tools or a dedicated inference thread.

## Hot-Loop Worker

Use `DetectorWorker` when a thread should submit one task, continue application work, poll
completion, then submit the next task. The worker is intentionally single-flight: it does not hide
an unbounded queue inside the library.

```cpp
idet::DetectorWorkerOptions options{};
options.copy_input = true; // safe default for non-owning frames

auto worker_result = idet::DetectorWorker::create(config, options);
if (!worker_result.ok()) {
    // worker_result.status().message
}
idet::DetectorWorker worker = std::move(worker_result).value();

while (running) {
    switch (worker.state()) {
    case idet::DetectorWorkerState::Idle:
        if (has_next_frame()) {
            const idet::Status s = worker.submit(next_frame());
            if (!s.ok()) {
                // backpressure or invalid input
            }
        }
        break;

    case idet::DetectorWorkerState::Running:
        run_other_application_work();
        break;

    case idet::DetectorWorkerState::Ready: {
        auto result = worker.take_result();
        if (!result.ok()) {
            // result.status().message
            break;
        }
        consume_detections(result.value());
        break;
    }
    }
}
```

`state()` is atomic and suitable for frequent polling. `submit()` and `take_result()` are serialized
handoff points; call them from one controlling thread per worker unless you add your own higher-level
synchronization.

For multiple in-flight tasks, create multiple workers and assign a resource budget to each lane.
That keeps queueing, cancellation, and fairness policy in the host pipeline where the rest of the
dependencies are visible.

## Fixed Shape and Bound I/O

For stable low-latency loops, prefer fixed input shapes and binding when the model contract allows
it. `GridSpec` uses rows x cols, so `fixed_input_dim = {640, 640}` means H=640, W=640.

```cpp
config.infer.bind_io = true;
config.infer.fixed_input_dim = {640, 640};

idet::DetectorWorkerOptions options{};
options.use_bound = true;
options.binding_height = 640;
options.binding_width = 640;
options.binding_contexts = 1;
options.binding_context_index = 0;
```

With tiling and direct `Detector::detect_bound()`, prepare one binding context per concurrently used
context. With `DetectorWorker`, a single-flight lane usually needs one context.

## Image Ownership

`idet::Image` can be either owning or non-owning. For asynchronous use, choose one of these:

- Set `DetectorWorkerOptions::copy_input = true` to deep-copy submitted frames at the handoff.
- Keep `copy_input = false` only when the backing memory remains alive and immutable until
  `take_result()` completes.
- Use `Image::wrap()` with a shared owner token if your application already manages frame buffers
  through reference-counted storage.

If your application uses OpenCV directly, keep OpenCV at the app boundary and wrap the `cv::Mat`
as an `idet::Image` view:

```cpp
cv::Mat bgr = ...; // CV_8UC3

idet::Image image = idet::Image::view(idet::ImageView{
    bgr.data,
    bgr.cols,
    bgr.rows,
    static_cast<std::size_t>(bgr.step),
    idet::PixelFormat::BGR_U8,
});
```

For async submission with `copy_input = false`, the `cv::Mat` storage must outlive the worker task.
With `copy_input = true`, the worker copies the frame before returning from `submit()`.

## Resource Pattern

A robust host pipeline usually looks like this:

1. Decide the CPU budget at the application level.
2. Call `setup_runtime_policy()` once at a controlled boundary if global OpenCV/OpenMP changes are acceptable.
3. Create one `DetectorWorker` per inference lane.
4. Use `state()` for cheap polling.
5. Keep queueing and cancellation outside IDet.
6. Tune ORT intra-op threads, tile OpenMP threads, and worker count together to avoid oversubscription.
