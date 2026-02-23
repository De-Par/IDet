#!/usr/bin/env python3

import argparse
import os

from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--output", default="quantized.onnx")
    args = ap.parse_args()

    onnx_path = args.onnx
    out_path = args.output

    if os.path.exists(onnx_path) and onnx_path.endswith(".onnx"):
        quantize_dynamic(
            model_input=onnx_path,
            model_output=out_path,
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=["Conv", "MatMul"],
        )
    else:
        raise FileNotFoundError("ONNX model not found")


if __name__ == "__main__":
    main()
    