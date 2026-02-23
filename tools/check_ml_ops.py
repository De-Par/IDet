#!/usr/bin/env python3

import argparse
import os
import onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to .onnx model")
    args = ap.parse_args()

    onnx_path = args.onnx

    if not (os.path.isfile(onnx_path) and onnx_path.lower().endswith(".onnx")):
        raise FileNotFoundError(f"ONNX model not found or not an .onnx file: {onnx_path}")

    m = onnx.load(onnx_path)

    nodes = list(m.graph.node)
    domains = sorted({(n.domain or "ai.onnx") for n in nodes})
    ops = sorted({((n.domain or "ai.onnx"), n.op_type) for n in nodes})

    print("Model:", onnx_path)
    print("IR version:", getattr(m, "ir_version", "unknown"))
    print("Opset imports:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])

    print("\nDomains:", domains)
    print("Has ai.onnx.ml:", "ai.onnx.ml" in domains)

    print("\nUnique ops (domain, op_type):", len(ops))
    for d, op in ops:
        print(f"  {d}:{op}")


if __name__ == "__main__":
    main()
    