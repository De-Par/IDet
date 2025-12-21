import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import onnx
from onnx import TensorProto


def _domain_norm(d: str) -> str:
    # In ONNX, empty domain usually means "ai.onnx"
    return d if d else "ai.onnx"


def _dtype_name(elem_type: int) -> str:
    try:
        return TensorProto.DataType.Name(elem_type)
    except Exception:
        return str(elem_type)


def _shape_str(vtype) -> str:
    # vtype is ValueInfoProto.type (may be unset / partial)
    if not vtype or not vtype.HasField("tensor_type"):
        return "<?>"
    tt = vtype.tensor_type
    if not tt.HasField("shape"):
        return "<?>"
    dims = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return "[" + ", ".join(dims) + "]"


def _valueinfo_line(vi) -> str:
    # vi: ValueInfoProto
    if not vi or not vi.HasField("type") or not vi.type.HasField("tensor_type"):
        return f"{vi.name}: <non-tensor or unknown>"
    tt = vi.type.tensor_type
    dtype = _dtype_name(tt.elem_type) if tt.HasField("elem_type") else "<?>"
    shape = _shape_str(vi.type)
    return f"{vi.name}: {dtype} {shape}"


def _collect_graph_ios(model: onnx.ModelProto) -> tuple[list, list]:
    g = model.graph

    # Some graph inputs are initializers (weights); filter them out
    init_names = {init.name for init in g.initializer}
    real_inputs = [i for i in g.input if i.name not in init_names]
    outputs = list(g.output)
    return real_inputs, outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect ONNX model: domains, opsets, IO, op stats.")
    ap.add_argument("--onnx", required=True, help="Path to .onnx model file")
    ap.add_argument(
        "--no-check",
        action="store_true",
        help="Skip onnx.checker.check_model() validation",
    )
    ap.add_argument(
        "--show-ops",
        action="store_true",
        help="Print unique (domain, op_type) list",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=30,
        help="Top-N ops by frequency to print (default: 30)",
    )
    args = ap.parse_args(argv)

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if onnx_path.suffix.lower() != ".onnx":
        raise ValueError(f"Expected a .onnx file, got: {onnx_path.name}")

    model = onnx.load(str(onnx_path))

    if not args.no_check:
        onnx.checker.check_model(model)

    # ---- Basic metadata
    print(f"Model: {onnx_path}")
    print(f"IR version: {getattr(model, 'ir_version', '<?>')}")
    print(f"Producer: {model.producer_name or '<unknown>'} {model.producer_version or ''}".rstrip())
    if model.model_version:
        print(f"Model version: {model.model_version}")

    # ---- Opset imports
    print("Opset imports:")
    if model.opset_import:
        for imp in model.opset_import:
            dom = _domain_norm(imp.domain)
            print(f"  {dom}: {imp.version}")
    else:
        print("  <none>")

    # ---- Domains / ops
    domains = sorted({_domain_norm(n.domain) for n in model.graph.node})
    print("Domains:", domains)
    print("Has ai.onnx.ml:", "ai.onnx.ml" in domains)

    # ---- Inputs/Outputs
    inputs, outputs = _collect_graph_ios(model)
    print("Inputs:")
    if inputs:
        for vi in inputs:
            print("  " + _valueinfo_line(vi))
    else:
        print("  <none>")

    print("Outputs:")
    if outputs:
        for vi in outputs:
            print("  " + _valueinfo_line(vi))
    else:
        print("  <none>")

    # ---- Op statistics
    domop_counter = Counter((_domain_norm(n.domain), n.op_type) for n in model.graph.node)
    op_counter = Counter(n.op_type for n in model.graph.node)

    print(f"Total nodes: {len(model.graph.node)}")
    print(f"Unique ops (domain, op_type): {len(domop_counter)}")
    print(f"Unique op_types: {len(op_counter)}")

    topn = max(0, args.top)
    if topn:
        print(f"Top {topn} op_types:")
        for op, cnt in op_counter.most_common(topn):
            print(f"  {op}: {cnt}")

    if args.show_ops:
        print("Unique ops (domain :: op_type):")
        for dom, op in sorted(domop_counter.keys()):
            print(f"  {_domain_norm(dom)} :: {op}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
