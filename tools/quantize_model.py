import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    ap = argparse.ArgumentParser(description="Dynamic weight quantization for ONNX (ONNX Runtime).")
    ap.add_argument("--onnx", required=True, help="Path to input .onnx model")
    ap.add_argument("--output", default="quantized.onnx", help="Path to output .onnx model")
    ap.add_argument(
        "--weight-type",
        choices=["quint8", "qint8"],
        default="quint8",
        help="Weight quantization type (default: quint8)",
    )
    ap.add_argument(
        "--ops",
        default="Conv,MatMul",
        help="Comma-separated list of op types to quantize (default: Conv,MatMul)",
    )
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    out_path = Path(args.output)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if onnx_path.suffix.lower() != ".onnx":
        raise ValueError(f"Expected .onnx input, got: {onnx_path.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    weight_type = QuantType.QUInt8 if args.weight_type == "quint8" else QuantType.QInt8
    op_types = [s.strip() for s in args.ops.split(",") if s.strip()]

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(out_path),
        weight_type=weight_type,
        op_types_to_quantize=op_types,
    )

    print(f"Saved quantized model to: {out_path}")


if __name__ == "__main__":
    main()
