#!/usr/bin/env python3

from onnxruntime.quantization import quantize_dynamic, QuantType 
import argparse 
import os


def main(): 
    ap = argparse.ArgumentParser() 
    ap.add_argument("--onnx", required=True) 
    ap.add_argument("--output", default="quantized.onnx", required=False) 
    args = ap.parse_args()

    onnx_path = args.onnx 
    out_path = args.output

    if os.path.exists(onnx_path) and '.onnx' in onnx_path: 
        quantize_dynamic(
            model_input = onnx_path, 
            model_output = out_path, 
            weight_type = QuantType.QUInt8,            # or QuantType.QUInt8 
            op_types_to_quantize = ['Conv', 'MatMul']  # focus on heavy ops
        ) 
    else: 
        raise FileNotFoundError('ONNX model not found')


if __name__ == "__main__": 
    main()
    