# Model Zoo

[Back to README](../README.md) | [Documentation index](index.md) | Previous: [Build & Install](build.md) | Next: [Command-line Options](cli.md)

This project is **model-agnostic** as long as your detector exports one of the supported output contracts. Built-in engines cover text detection, face detection, and cloth detection.


## MMOCR

MMOCR provides many detectors (R50, MobileNet, DCN variants, etc.). You can export them to ONNX and use them directly with this tool. Detailed information about available models you can find there: **[mmocr_models](https://mmocr.readthedocs.io/en/dev-1.x/textdet_models.html)**. Also, take a look on support in ONNX Runtime: **[mmocr_support](https://mmdeploy.readthedocs.io/en/latest/04-supported-codebases/mmocr.html)**.

Export with MMOCR’s `pytorch2onnx.py`:

```bash
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
python3.11 -m venv mvenv
source ./mvenv/bin/activate
pip install -r requirements.txt
pip install onnx onnxsim

python tools/deployment/pytorch2onnx.py <CONFIG.py> \
    --checkpoint <MODEL.pth> \
    --output-file <OUT.onnx> \
    --opset 11 --dynamic-export
```

Optional simplification:

```bash
python -m onnxsim <OUT.onnx> <OUT-sim.onnx>
```

> 💡 **Notes & tips:**
> - Prefer **opset ≥ 11**. For CPU inference, 11–13 is typically safe.
> - If you need dynamic spatial sizes, keep `--dynamic-export`; otherwise static shapes plus `--fixed_hw` may be faster/stabler.
> - Some MMOCR configs already include the final **Sigmoid** in the head. If your output looks like logits, run with `--sigmoid 1`.
> - Keep input channels at 3 unless you change the first conv to 1-channel and re-train/fine-tune (grayscale alone rarely gives a big speedup).


## PaddleOCR

The bundled PaddleOCR text detectors use pre-converted **PP-OCR ONNX** models from the Hugging Face Hub collection **[deepghs/paddleocr](https://huggingface.co/deepghs/paddleocr/tree/main)**. The collection includes multiple PP-OCR detector generations, including lightweight mobile variants and higher-accuracy server variants.

Models available in `assets/models/paddleocr`:

- `ch_ppocr_v2_det.onnx`
- `ch_ppocr_mobile_v2_det.onnx`
- `ch_ppocr_mobile_slim_v2_det.onnx`
- `ch_ppocr_server_v2_det.onnx`
- `ch_ppocr_v3_det.onnx`
- `en_ppocr_v3_det.onnx`
- `ch_ppocr_v4_det.onnx`
- `ch_ppocr_v4_server_det.onnx`

The models are used for text region detection only. They do not perform OCR text recognition.


## DBNet / DBNet++

The bundled DBNet text detectors use ready-to-use **DBNet / DBNet++ ONNX** exports from the Hugging Face Hub repository **[deepghs/text_detection](https://huggingface.co/deepghs/text_detection/tree/main)**.

Models available in `assets/models/dbnet`:

- `dbnet_resnet18_fpnc_1200e_icdar2015.onnx`
- `dbnet_resnet18_fpnc_1200e_totaltext.onnx`

These models are classic scene-text detectors and are used for text region detection only.


## SCRFD

The bundled face detector uses a pre-converted **SCRFD ONNX** model from the Hugging Face Hub repository **[ykk648/face_lib](https://huggingface.co/ykk648/face_lib/tree/main/face_detect/scrfd_onnx)**.

Model available in `assets/models/scrfd`:

- `scrfd_500m_bnkps.onnx`

SCRFD models output face bounding boxes and confidence scores. Variants with `bnkps` in the name also predict 5 facial landmarks, so `scrfd_500m_bnkps.onnx` provides both bounding boxes and keypoints.


## YOLO

The bundled clothing detectors use custom **YOLO26 ONNX** models trained for 7 clothing-related classes and exported with dynamic spatial input. Models are available in `assets/models/yolo` and are grouped by model size and training dataset.

Model families:

| Directory | Description |
|:---|:---|
| `yolo26n/` | Lightweight nano models for faster CPU inference |
| `yolo26s/` | Larger small models with higher accuracy and higher runtime cost |

Dataset variants:

| Directory | Description |
|:---|:---|
| `deepfashion2-7c/` | Trained on the 7-class DeepFashion2-based dataset |
| `df2-and-fp-7c/` | Trained on the merged DeepFashion2 + Fashionpedia 7-class dataset |

Artifacts provided for each dataset variant:

| Prefix | Description |
|:---|:---|
| `*_fp32_dynamic.onnx` | FP32 baseline model |
| `*_int8_qdq_s8s8_dynamic.onnx` | INT8 QDQ model with signed activations and signed weights |
| `*_int8_qdq_u8s8_dynamic.onnx` | INT8 QDQ model with unsigned activations and signed weights |
| `*_int8_qop_u8u8_dynamic.onnx` | INT8 QOperator model with unsigned activations and unsigned weights |

Models available in `assets/models/yolo`:

```text
assets/models/yolo/
├── yolo26n
│   ├── deepfashion2-7c
│   │   ├── yolo26n_df2-7c_fp32_dynamic.onnx
│   │   ├── yolo26n_df2-7c_int8_qdq_s8s8_dynamic.onnx
│   │   ├── yolo26n_df2-7c_int8_qdq_u8s8_dynamic.onnx
│   │   └── yolo26n_df2-7c_int8_qop_u8u8_dynamic.onnx
│   └── df2-and-fp-7c
│       ├── yolo26n_df2-and-fp-7c_fp32_dynamic.onnx
│       ├── yolo26n_df2-and-fp-7c_int8_qdq_s8s8_dynamic.onnx
│       ├── yolo26n_df2-and-fp-7c_int8_qdq_u8s8_dynamic.onnx
│       └── yolo26n_df2-and-fp-7c_int8_qop_u8u8_dynamic.onnx
└── yolo26s
    ├── deepfashion2-7c
    │   ├── yolo26s_df2-7c_fp32_dynamic.onnx
    │   ├── yolo26s_df2-7c_int8_qdq_s8s8_dynamic.onnx
    │   ├── yolo26s_df2-7c_int8_qdq_u8s8_dynamic.onnx
    │   └── yolo26s_df2-7c_int8_qop_u8u8_dynamic.onnx
    └── df2-and-fp-7c
        ├── yolo26s_df2-and-fp-7c_fp32_dynamic.onnx
        ├── yolo26s_df2-and-fp-7c_int8_qdq_s8s8_dynamic.onnx
        ├── yolo26s_df2-and-fp-7c_int8_qdq_u8s8_dynamic.onnx
        └── yolo26s_df2-and-fp-7c_int8_qop_u8u8_dynamic.onnx
```

All models use dynamic spatial input. Input sizes should still be aligned to the YOLO stride. For stable latency and reusable I/O buffers, runtime scripts may use fixed-shape IOBinding profiles even though the exported ONNX graphs support dynamic input shapes.

### Benchmark setup

The numbers below were measured with the following export and runtime profile:

| Parameter | Value |
|:---|:---|
| Input size | 640 |
| Input shape | Dynamic spatial input |
| ONNX opset | 20 |
| Calibration method | Percentile |
| Calibration percentile | 99.98 |
| Runtime | ONNX Runtime CPUExecutionProvider |
| Target | CPU inference |

Metric meanings:

| Metric | Description |
|:---|:---|
| mAP50-95 | COCO-style mean Average Precision averaged over IoU thresholds from 0.50 to 0.95 |
| mAP50 | mean Average Precision at IoU threshold 0.50 |
| Precision | Detection precision |
| Recall | Detection recall |
| Nodes | Number of ONNX graph nodes |
| Mem, MB | ONNX model file size in megabytes |

---

### YOLO26n — FP32

| Metric | DF2-7C | DF2-FP-7C |
|:---|:---:|:---:|
| mAP50-95 | 0,8131 | 0,8202 |
| mAP50 | 0,9343 | 0,9395 |
| Precision | 0,8974 | 0,9319 |
| Recall | 0,8767 | 0,8621 |
| Nodes | 591 | 591 |
| Mem, mb | 9,3 | 9,3 |

### YOLO26n — INT8

| Scenario | Metric | DF2-7C | DF2-FP-7C |
|:---|:---|:---:|:---:|
| `qdq-s8s8` | mAP50-95 | 0,7466 | 0,7421 |
| | mAP50 | 0,8848 | 0,8736 |
|| Precision | 0,8211 | 0,9401 |
|| Recall | 0,9058 | 0,7984 |
|| Nodes | 1190 | 1190 |
|| Mem, mb | 2,9 | 2,9 |
|||||
| `qdq-u8s8` | mAP50-95 | 0,7479 | 0,7436 |
|| mAP50 | 0,8857 | 0,8729 |
|| Precision | 0,8228 | 0,9373 |
|| Recall | 0,9038 | 0,7981 |
|| Nodes | 827 | 827 |
|| Mem, mb | 2,7 | 2,7 |
|||||
| `qoperator-u8u8` | mAP50-95 | 0,7492 | 0,7437 |
|| mAP50 | 0,8863 | 0,8723 |
|| Precision | 0,8261 | 0,9404 |
|| Recall | 0,9040 | 0,7978 |
|| Nodes | 800 | 800 |
|| Mem, mb | 2,8 | 2,8 |

---

### YOLO26s — FP32

| Metric | DF2-7C | DF2-FP-7C |
|:---|:---:|:---:|
| mAP50-95 | 0,8364 | 0,8161 |
| mAP50 | 0,9490 | 0,9225 |
| Precision | 0,9167 | 0,8982 |
| Recall | 0,8940 | 0,8620 |
| Nodes | 591 | 591 |
| Mem, mb | 36,4 | 36,4 |

### YOLO26s — INT8

| Scenario | Metric | DF2-7C | DF2-FP-7C |
|:---|:---|:---:|:---:|
| `qdq-s8s8` | mAP50-95 | 0,8167 | 0,7740 |
|| mAP50 | 0,9405 | 0,9000 |
|| Precision | 0,9256 | 0,9018 |
|| Recall | 0,8818 | 0,8478 |
|| Nodes | 1190 | 1190 |
|| Mem, mb | 9,8 | 9,8 |
|||||
| `qdq-u8s8` | mAP50-95 | 0,8159 | 0,7707 |
|| mAP50 | 0,9414 | 0,8971 |
|| Precision | 0,9249 | 0,8980 |
|| Recall | 0,8826 | 0,8492 |
|| Nodes | 827 | 827 |
|| Mem, mb | 9,6 | 9,6 |
|||||
| `qoperator-u8u8` | mAP50-95 | 0,8158 | 0,7712 |
|| mAP50 | 0,9403 | 0,8968 |
|| Precision | 0,9203 | 0,9002 |
|| Recall | 0,8817 | 0,8465 |
|| Nodes | 800 | 800 |
|| Mem, mb | 9,6 | 9,6 |

---

> ⚠️ **Warn:** Only validated INT8 variants are bundled. Some additional quantization combinations were tested during export experiments but are not included here because they either produced unstable ONNX Runtime execution or did not provide a better accuracy/latency trade-off.


## Compatibility Notes

- **Output often contains logits** → run with `--sigmoid 1`.
- **Normalization differs from ImageNet**: PaddleOCR commonly uses `img = (img/255.0 - 0.5) / 0.5` (i.e., `mean=(0.5,0.5,0.5)`, `std=(0.5,0.5,0.5)`). The current code uses ImageNet stats (`mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`). For best accuracy with Paddle models, adjust normalization in code to Paddle’s scheme or re-export to match ImageNet stats.
- **Input sizes** are typically dynamic with the constraint **H,W % 32 == 0**. Use `--fixed_hw` (e.g., `640x640`) to meet that requirement.
- DBNet-style output supports `[1,1,H,W]`, `[1,H,W,1]`, `[1,H,W]`, and `[H,W]` probability/logit maps.
- SCRFD and YOLO engines validate their own expected output contracts during model probing.

> 💡 **Notes & tips:**
> - If you switch to Paddle normalization, update mean / std in code accordingly.
> - For highest stability in batch/production: combine **IOBinding** (`--bind_io 1`) with a **fixed input size** (`--fixed_hw`) and keep ORT threads small (`--threads_intra 1–2`) while scaling tiles via OpenMP (`--tile_omp`).
