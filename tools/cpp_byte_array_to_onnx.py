#!/usr/bin/env python3

import argparse
import re
import sys

from pathlib import Path


def extract_array_block(text: str, symbol_name: str | None) -> tuple[str, str]:
    if symbol_name is not None:
        pattern = (
            rf'(?:alignas\(\d+\)\s+)?'
            rf'(?:extern\s+)?const\s+unsigned\s+char\s+{re.escape(symbol_name)}\[\]\s*=\s*\{{(.*?)\}};'
        )
        match = re.search(pattern, text, flags=re.DOTALL)
        if match is None:
            raise ValueError(f'Byte array "{symbol_name}" not found')

        return symbol_name, match.group(1)

    pattern = (
        r'(?:alignas\(\d+\)\s+)?'
        r'(?:extern\s+)?const\s+unsigned\s+char\s+([A-Za-z_]\w*)\[\]\s*=\s*\{(.*?)\};'
    )
    matches = re.findall(pattern, text, flags=re.DOTALL)

    if not matches:
        raise ValueError('No byte arrays found in input file')

    if len(matches) > 1:
        names = ', '.join(name for name, _ in matches)
        raise ValueError(
            'Multiple byte arrays found. '
            f'Please specify --name. Available arrays: {names}'
        )

    return matches[0]


def extract_declared_length(text: str, symbol_name: str) -> int | None:
    pattern = (
        rf'(?:extern\s+)?const\s+std::size_t\s+'
        rf'{re.escape(symbol_name)}_len\s*=\s*(\d+)\s*;'
    )
    match = re.search(pattern, text)
    if match is None:
        return None

    return int(match.group(1))


def parse_hex_bytes(array_block: str) -> bytes:
    hex_bytes = re.findall(r'0x([0-9a-fA-F]{2})', array_block)
    if not hex_bytes:
        raise ValueError('No hex bytes found inside the array block')

    return bytes(int(value, 16) for value in hex_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description='Restore an ONNX file from a C++ byte array source')
    parser.add_argument('--input', required=True, help='Path to input .cpp/.h file containing the embedded byte array')
    parser.add_argument('--output', required=True, help='Path to restored output .onnx file')
    parser.add_argument('--name', help='C++ byte array symbol name. Required if the file contains multiple arrays')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding='utf-8', errors='ignore')

    symbol_name, array_block = extract_array_block(text, args.name)
    data = parse_hex_bytes(array_block)

    declared_length = extract_declared_length(text, symbol_name)
    if declared_length is not None and declared_length != len(data):
        raise ValueError(
            f'Length mismatch for "{symbol_name}": '
            f'declared {declared_length}, parsed {len(data)}'
        )

    output_path.write_bytes(data)

    print(f'Restored ONNX from symbol: {symbol_name}')
    print(f'Output path: {output_path}')
    print(f'Size: {len(data)} bytes')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
