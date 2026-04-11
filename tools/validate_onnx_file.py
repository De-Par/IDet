#!/usr/bin/env python3

import argparse
import hashlib
import sys

from pathlib import Path


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_onnx_file(path: Path) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            'ONNX validation requires the "onnx" package. '
            'Install it with: pip install onnx'
        ) from exc

    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate an ONNX file and optionally verify size and SHA256')
    parser.add_argument('--input', required=True, help='Path to ONNX file')
    parser.add_argument('--expected-size', type=int, help='Expected file size in bytes')
    parser.add_argument('--expected-sha256', help='Expected SHA256 checksum in hex')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    actual_size = input_path.stat().st_size
    actual_sha256 = sha256_file(input_path)

    if args.expected_size is not None and actual_size != args.expected_size:
        raise ValueError(
            f'Size mismatch: expected {args.expected_size}, got {actual_size}'
        )

    if args.expected_sha256 is not None:
        expected_sha256 = args.expected_sha256.lower()
        if actual_sha256.lower() != expected_sha256:
            raise ValueError(
                f'SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}'
            )

    validate_onnx_file(input_path)

    print(f'Input path: {input_path}')
    print(f'Size: {actual_size} bytes')
    print(f'SHA256: {actual_sha256}')
    print('Integrity check: OK')
    print('ONNX validation: OK')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
