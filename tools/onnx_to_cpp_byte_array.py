#!/usr/bin/env python3

import sys
import argparse

from pathlib import Path


BYTES_PER_LINE = 64


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an ONNX file into a C++ byte array definition")
    parser.add_argument("--name", required=True, help="C++ symbol base name")
    parser.add_argument("--input", required=True, help="Path to input ONNX file")
    parser.add_argument("--output", required=True, help="Path to generated output file")
    args = parser.parse_args()

    data = Path(args.input).read_bytes()
    name = args.name

    out = []
    out.append("#include <cstddef>\n")
    out.append("namespace idet::internal {\n")
    out.append(f"alignas(64) extern const unsigned char {name}[] = {{\n")

    for i in range(0, len(data), BYTES_PER_LINE):
        chunk = data[i : i + BYTES_PER_LINE]
        out.append("  " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",\n")

    out.append("};\n")
    out.append(f"extern const std::size_t {name}_len = {len(data)};\n")
    out.append("} // namespace idet::internal\n")

    Path(args.output).write_text("".join(out), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
