#!/usr/bin/env python3

import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = Path(args.input).read_bytes()
    name = args.name

    out = []
    out.append('#include <cstddef>\n')
    out.append('namespace idet::internal {\n')
    out.append(f'alignas(64) extern const unsigned char {name}[] = {{\n')

    # 12 bytes per line
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        out.append('  ' + ', '.join(f'0x{b:02x}' for b in chunk) + ',\n')

    out.append('};\n')
    out.append(f'extern const std::size_t {name}_len = {len(data)};\n')
    out.append('} // namespace idet::internal\n')

    Path(args.output).write_text(''.join(out))

if __name__ == "__main__":
    main()
    