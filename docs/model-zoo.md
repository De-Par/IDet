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

There are pre-converted **PaddleOCR** detectors on the Hugging Face Hub: **[deepghs/paddleocr](https://huggingface.co/deepghs/paddleocr/tree/main)**. The collection includes multiple **PP-OCR** detector generations (v2/v3/v4), including lightweight **mobile** variants and higher-accuracy **server** variants.

Typical model names in `assets/models/paddleocr`:

- `ch_ppocr_v2_det.onnx`
- `ch_ppocr_mobile_v2_det.onnx`
- `ch_ppocr_mobile_slim_v2_det.onnx`
- `ch_ppocr_server_v2_det.onnx`
- `ch_ppocr_v3_det.onnx`
- `en_ppocr_v3_det.onnx`
- `ch_ppocr_v4_det.onnx`
- `ch_ppocr_v4_server_det.onnx`


## DBNet / DBNet++

If you want to test classic **DBNet / DBNet++** models, the Hugging Face Hub repo by **deepghs** provides ready-to-use ONNX exports: **[deepghs/text_detection](https://huggingface.co/deepghs/text_detection/tree/main)**.

Models available in `assets/models/dbnet`:

- `dbnet_resnet18_fpnc_1200e_icdar2015.onnx`
- `dbnet_resnet18_fpnc_1200e_totaltext.onnx`


## SCRFD

There are pre-converted **SCRFD** face detectors on the Hugging Face Hub: **[ykk648/face_lib](https://huggingface.co/ykk648/face_lib/tree/main/face_detect/scrfd_onnx)**. SCRFD models typically output face bounding boxes + confidence scores, and many variants also predict 5 facial landmarks. In model names, `bnkps` commonly indicates **bboxes + keypoints**.

Model available in `assets/models/scrfd`:

- `scrfd_500m_bnkps.onnx`


## YOLO

The bundled cloth detector is based on the **Kesimeg YOLOv8 clothing detection** model and is exported to ONNX with dynamic spatial input.

Model available in `assets/models/yolo`:

- `yolov8n-kesimeg.onnx`

The model accepts dynamic input sizes aligned to the YOLO stride. For stable latency and reusable I/O buffers, the scripts still run it with fixed-shape IOBinding defaults that match the text/face launch profiles.


## Compatibility Notes

- **Output often contains logits** → run with `--sigmoid 1`.
- **Normalization differs from ImageNet**: PaddleOCR commonly uses `img = (img/255.0 - 0.5) / 0.5` (i.e., `mean=(0.5,0.5,0.5)`, `std=(0.5,0.5,0.5)`). The current code uses ImageNet stats (`mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`). For best accuracy with Paddle models, adjust normalization in code to Paddle’s scheme or re-export to match ImageNet stats.
- **Input sizes** are typically dynamic with the constraint **H,W % 32 == 0**. Use `--fixed_hw` (e.g., `640x640`) to meet that requirement.
- DBNet-style output supports `[1,1,H,W]`, `[1,H,W,1]`, `[1,H,W]`, and `[H,W]` probability/logit maps.
- SCRFD and YOLO engines validate their own expected output contracts during model probing.

> 💡 **Notes & tips:**
> - If you switch to Paddle normalization, update mean / std in code accordingly.
> - For highest stability in batch/production: combine **IOBinding** (`--bind_io 1`) with a **fixed input size** (`--fixed_hw`) and keep ORT threads small (`--threads_intra 1–2`) while scaling tiles via OpenMP (`--tile_omp`).

🔝 [Back to top](#model-zoo)
