#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import lzma
import sys

from pathlib import Path


BASE91_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    "!#$%&()*+,./:;<=>?@[]^_`{|}~\""
)
assert len(BASE91_ALPHABET) == 91

BASE91_DECODE = {ch: i for i, ch in enumerate(BASE91_ALPHABET)}

MAGIC = "B91XZ1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def write_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def base91_encode(data: bytes) -> str:
    b = 0
    n = 0
    out: list[str] = []

    for byte in data:
        b |= byte << n
        n += 8

        if n > 13:
            v = b & 8191
            if v > 88:
                b >>= 13
                n -= 13
            else:
                v = b & 16383
                b >>= 14
                n -= 14

            out.append(BASE91_ALPHABET[v % 91])
            out.append(BASE91_ALPHABET[v // 91])

    if n:
        out.append(BASE91_ALPHABET[b % 91])
        if n > 7 or b > 90:
            out.append(BASE91_ALPHABET[b // 91])

    return "".join(out)


def base91_decode(text: str) -> bytes:
    v = -1
    b = 0
    n = 0
    out = bytearray()

    for ch in text:
        if ch.isspace():
            continue

        c = BASE91_DECODE.get(ch)
        if c is None:
            raise ValueError(f"Invalid basE91 character: {ch!r}")

        if v < 0:
            v = c
        else:
            v += c * 91
            b |= v << n

            if (v & 8191) > 88:
                n += 13
            else:
                n += 14

            while n > 7:
                out.append(b & 255)
                b >>= 8
                n -= 8

            v = -1

    if v >= 0:
        out.append((b | (v << n)) & 255)

    return bytes(out)


def compress_xz(data: bytes) -> bytes:
    return lzma.compress(
        data,
        format=lzma.FORMAT_XZ,
        preset=9 | lzma.PRESET_EXTREME,
    )


def decompress_xz(data: bytes) -> bytes:
    return lzma.decompress(data, format=lzma.FORMAT_XZ)


def wrap_text(text: str, width: int) -> str:
    if width <= 0:
        return text
    return "\n".join(text[i:i + width] for i in range(0, len(text), width))


def build_text_package(raw: bytes, wrap: int = 0) -> str:
    packed = compress_xz(raw)
    encoded = base91_encode(packed)
    encoded = wrap_text(encoded, wrap)
    digest = sha256_bytes(raw)
    return f"{MAGIC}:{digest}\n{encoded}\n"


def parse_text_package(text: str) -> tuple[str, str]:
    first_newline = text.find("\n")
    if first_newline == -1:
        raise ValueError("Invalid package: missing header line")

    header = text[:first_newline].strip()
    payload = text[first_newline + 1:]

    prefix = f"{MAGIC}:"
    if not header.startswith(prefix):
        raise ValueError(f"Invalid package: expected header starting with {prefix!r}")

    digest = header[len(prefix):].strip()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("Invalid package: bad SHA-256 in header")

    return digest, payload


def pack_file(input_path: Path, output_path: Path, wrap: int) -> None:
    raw = read_bytes(input_path)
    package = build_text_package(raw, wrap=wrap)
    output_path.write_text(package, encoding="utf-8")

    packed = compress_xz(raw)
    encoded = base91_encode(packed)

    print(f"input file        : {input_path}")
    print(f"output text       : {output_path}")
    print(f"original bytes    : {len(raw)}")
    print(f"xz bytes          : {len(packed)}")
    print(f"basE91 chars      : {len(encoded)}")
    print(f"text/original     : {len(package.encode('utf-8')) / len(raw):.4f}")
    print(f"sha256            : {sha256_bytes(raw)}")


def unpack_file(input_path: Path, output_path: Path) -> None:
    text = input_path.read_text(encoding="utf-8")
    expected_digest, payload = parse_text_package(text)

    packed = base91_decode(payload)
    raw = decompress_xz(packed)
    actual_digest = sha256_bytes(raw)

    if actual_digest != expected_digest:
        raise ValueError(
            "Integrity check failed: SHA-256 mismatch\n"
            f"expected: {expected_digest}\n"
            f"actual  : {actual_digest}"
        )

    write_bytes(output_path, raw)

    print(f"input text        : {input_path}")
    print(f"restored file      : {output_path}")
    print(f"restored bytes    : {len(raw)}")
    print(f"sha256            : {actual_digest}")
    print("integrity         : OK")


def verify_roundtrip(input_path: Path) -> None:
    raw = read_bytes(input_path)
    package = build_text_package(raw, wrap=0)
    expected_digest, payload = parse_text_package(package)
    restored = decompress_xz(base91_decode(payload))
    actual_digest = sha256_bytes(restored)

    ok = raw == restored and expected_digest == actual_digest
    print(f"file               : {input_path}")
    print(f"original bytes    : {len(raw)}")
    print(f"sha256            : {actual_digest}")
    print(f"roundtrip         : {'OK' if ok else 'FAILED'}")

    if not ok:
        raise ValueError("Roundtrip verification failed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Lossless ONNX <-> text converter using XZ + basE91")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack", help="Pack binary file into compact text")
    pack_parser.add_argument("input", type=Path, help="Input .onnx file")
    pack_parser.add_argument("output", type=Path, help="Output .txt file")
    pack_parser.add_argument("--wrap", type=int, default=0, help="Wrap payload every N chars, 0 = no wrapping for minimal size")

    unpack_parser = subparsers.add_parser("unpack", help="Restore binary file from text")
    unpack_parser.add_argument("input", type=Path, help="Input .txt file")
    unpack_parser.add_argument("output", type=Path, help="Restored .onnx file")

    verify_parser = subparsers.add_parser("verify", help="In-memory roundtrip check")
    verify_parser.add_argument("input", type=Path, help="Input .onnx file")

    args = parser.parse_args()

    try:
        if args.command == "pack":
            pack_file(args.input, args.output, wrap=args.wrap)
        elif args.command == "unpack":
            unpack_file(args.input, args.output)
        elif args.command == "verify":
            verify_roundtrip(args.input)
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
