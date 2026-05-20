#!/usr/bin/env python3
"""Grid-search runner for IDet ROI detector configurations.

Pipeline:
  1. Build one full-image single-shot reference with fixed_hw ~= image HxW.
     The same full-image single-shot reference boxes are used for both single and tiling candidates.
  2. Benchmark candidate configs using p50/p90/p95/p99 from idet_app stdout.
  3. Drop candidates with p99_ms > --max-p99-ms before box extraction.
  4. Run cheap dump pass for remaining candidates to extract Quads.
  5. Compute area quality:
       area_recall     = intersection_area / reference_area
       area_precision  = intersection_area / candidate_area
       area_f1         = harmonic mean of recall and precision
       extra_area_ratio = extra candidate area / reference_area
  6. Write CSV sorted by p99_ms ascending.

Expected idet_app latency output:
  p50_ms: 20.6417
  p90_ms: 20.7104
  p95_ms: 20.7293
  p99_ms: 20.7751

Expected dump output:
  dets_n: 11
  Quads:
      1 -> x0,y0 x1,y1 x2,y2 x3,y3
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

METRICS_RE: dict[str, re.Pattern[str]] = {
    "p50_ms": re.compile(rf"\bp50_ms\s*:\s*({FLOAT_RE})"),
    "p90_ms": re.compile(rf"\bp90_ms\s*:\s*({FLOAT_RE})"),
    "p95_ms": re.compile(rf"\bp95_ms\s*:\s*({FLOAT_RE})"),
    "p99_ms": re.compile(rf"\bp99_ms\s*:\s*({FLOAT_RE})"),
}

DETECTIONS_RE: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdets_n\s*:\s*(\d+)\b"),
    re.compile(r"\bnum\s+detection\s+quads\s*:\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bnum\s+detections\s*:\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bdetections\s*:\s*(\d+)\b", re.IGNORECASE),
)

INDEXED_QUAD_LINE_RE = re.compile(
    rf"^\s*\d+\s*->\s*"
    rf"({FLOAT_RE})\s*,\s*({FLOAT_RE})\s+"
    rf"({FLOAT_RE})\s*,\s*({FLOAT_RE})\s+"
    rf"({FLOAT_RE})\s*,\s*({FLOAT_RE})\s+"
    rf"({FLOAT_RE})\s*,\s*({FLOAT_RE})\s*$",
    re.MULTILINE,
)

DEFAULT_REF_HW = "auto"
DEFAULT_TILE_OVERLAP = 0.1

MIN_FIXED_SIDE = 64
FIXED_MULTIPLE = 32
EPS = 1e-9


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self) -> "Box":
        x1, x2 = sorted((float(self.x1), float(self.x2)))
        y1, y2 = sorted((float(self.y1), float(self.y2)))
        return Box(x1=x1, y1=y1, x2=x2, y2=y2)

    @property
    def area(self) -> float:
        box = self.normalized()
        return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


@dataclass(frozen=True)
class TargetDefaults:
    bin_thresh: float
    box_thresh: float
    unclip: float
    nms_iou: float
    use_fast_iou: int
    min_roi_size_w: int
    min_roi_size_h: int
    apply_sigmoid: int = 0


TARGET_DEFAULTS: dict[str, TargetDefaults] = {
    "text": TargetDefaults(
        bin_thresh=0.3,
        box_thresh=0.5,
        unclip=1.0,
        nms_iou=0.3,
        use_fast_iou=0,
        min_roi_size_w=10,
        min_roi_size_h=10,
    ),
    "face": TargetDefaults(
        bin_thresh=0.3,
        box_thresh=0.5,
        unclip=1.0,
        nms_iou=0.4,
        use_fast_iou=1,
        min_roi_size_w=8,
        min_roi_size_h=8,
    ),
    "cloth": TargetDefaults(
        bin_thresh=0.3,
        box_thresh=0.01,
        unclip=1.0,
        nms_iou=0.5,
        use_fast_iou=1,
        min_roi_size_w=8,
        min_roi_size_h=8,
    ),
}


@dataclass(frozen=True)
class Candidate:
    family: str
    tiles_rc: str
    fixed_hw: str
    max_img_size: int | None
    threads_intra: int
    threads_inter: int
    tile_omp: int
    tile_overlap: float

    def to_cli_kv(self) -> dict[str, Any]:
        kv: dict[str, Any] = {
            "fixed_hw": self.fixed_hw,
            "threads_intra": self.threads_intra,
            "threads_inter": self.threads_inter,
            "tile_omp": self.tile_omp,
        }

        # Single mode: do not pass --tiles_rc at all.
        # Some app versions parse "off/no/0" as 0x0 and fail:
        # DetectorConfig: tiles_dim must be > 0.
        if self.family == "tiling":
            kv["tiles_rc"] = self.tiles_rc
            kv["tile_overlap"] = self.tile_overlap

        if self.max_img_size is not None:
            kv["max_img_size"] = self.max_img_size

        return kv


@dataclass
class AreaQuality:
    area_recall: float | None
    area_precision: float | None
    area_f1: float | None
    intersection_area: float
    reference_area: float
    candidate_area: float
    extra_area: float | None
    extra_area_ratio: float | None


@dataclass
class RunResult:
    candidate: Candidate
    index: int
    command: list[str]
    box_command: list[str]
    status: str
    returncode: int | None
    elapsed_s_python: float
    metrics: dict[str, float | None]
    dets_n: int | None
    boxes_n: int
    quality: AreaQuality | None
    desired_threads: int
    skip_reason: str = ""
    stdout_path: str = ""
    boxes_stdout_path: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "ok" and not self.skip_reason


@dataclass
class SearchStats:
    total_candidates: int = 0
    skipped_by_threads: int = 0
    executed: int = 0
    kept: int = 0
    failed: int = 0
    timed_out: int = 0
    skipped_by_latency: int = 0
    skipped_no_metrics: int = 0
    skipped_no_boxes: int = 0
    skipped_by_quality: int = 0
    started_at: float = field(default_factory=time.time)


CSV_HEADER = [
    "rank",
    "target",
    "family",
    "status",
    "skip_reason",
    "p99_ms",
    "p95_ms",
    "p90_ms",
    "p50_ms",
    "area_recall",
    "area_precision",
    "area_f1",
    "intersection_area",
    "reference_area",
    "candidate_area",
    "extra_area",
    "extra_area_ratio",
    "boxes_n",
    "ref_boxes_n",
    "dets_n",
    "tiles_rc",
    "fixed_hw",
    "max_img_size",
    "threads_intra",
    "threads_inter",
    "tile_omp",
    "desired_threads",
    "tile_overlap",
    "elapsed_s_python",
    "command",
    "box_command",
    "stdout_path",
    "boxes_stdout_path",
]


def log(message: str) -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr, flush=True)


def die(message: str, code: int = 1) -> None:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def shell_join(argv: Sequence[str]) -> str:
    return shlex.join(list(argv))


def parse_hxw(value: str, *, name: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"{name} must have HxW format, got: {value!r}"
        )

    height = int(parts[0])
    width = int(parts[1])

    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError(
            f"{name} dimensions must be positive, got: {value!r}"
        )

    return height, width


def parse_tiles_rc(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"bad tiles_rc {value!r}, expected RxC")

    rows = int(parts[0])
    cols = int(parts[1])

    if rows <= 0 or cols <= 0:
        raise ValueError(f"bad tiles_rc {value!r}: rows/cols must be positive")

    return rows, cols


def floor_to_multiple(value: int, multiple: int = FIXED_MULTIPLE) -> int:
    if multiple <= 0:
        return value
    return (value // multiple) * multiple


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_image_hw(path: Path) -> tuple[int, int]:
    """Return image size as HxW for PNG/JPEG without OpenCV/Pillow."""
    png_signature = bytes.fromhex("89504E470D0A1A0A")
    jpeg_signature = bytes.fromhex("FFD8")
    marker_prefix = bytes.fromhex("FF")

    with path.open("rb") as file:
        header = file.read(32)

        if header.startswith(png_signature):
            if header[12:16] != b"IHDR":
                raise ValueError(f"bad PNG header: {path}")

            width = int.from_bytes(header[16:20], "big")
            height = int.from_bytes(header[20:24], "big")
            return height, width

        if header.startswith(jpeg_signature):
            file.seek(2)

            while True:
                marker_start = file.read(1)
                if not marker_start:
                    break

                if marker_start != marker_prefix:
                    continue

                marker = file.read(1)
                while marker == marker_prefix:
                    marker = file.read(1)

                if not marker:
                    break

                marker_code = marker[0]
                if marker_code in (0xD8, 0xD9):
                    continue

                length_bytes = file.read(2)
                if len(length_bytes) != 2:
                    break

                segment_length = int.from_bytes(length_bytes, "big")
                if segment_length < 2:
                    break

                if marker_code in {
                    0xC0,
                    0xC1,
                    0xC2,
                    0xC3,
                    0xC5,
                    0xC6,
                    0xC7,
                    0xC9,
                    0xCA,
                    0xCB,
                    0xCD,
                    0xCE,
                    0xCF,
                }:
                    data = file.read(segment_length - 2)
                    if len(data) < 5:
                        break

                    height = int.from_bytes(data[1:3], "big")
                    width = int.from_bytes(data[3:5], "big")
                    return height, width

                file.seek(segment_length - 2, os.SEEK_CUR)

    raise ValueError(f"cannot infer image size from file header: {path}")


def resolve_ref_hw(value: str, image_path: Path) -> tuple[int, int]:
    if value.lower() == "auto":
        return read_image_hw(image_path)
    return parse_hxw(value, name="ref_hw")


def dedupe_keep_order(items: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    output: list[Any] = []

    for item in items:
        if item in seen:
            continue

        seen.add(item)
        output.append(item)

    return output


def desired_threads(
    threads_inter: int,
    threads_intra: int,
    tile_omp: int,
) -> int:
    if threads_inter > 1 and threads_intra > 1:
        ort_peak = threads_inter + threads_intra
    else:
        ort_peak = max(threads_inter, threads_intra)

    return tile_omp + ort_peak


def candidate_desired_threads(candidate: Candidate) -> int:
    return desired_threads(
        threads_inter=candidate.threads_inter,
        threads_intra=candidate.threads_intra,
        tile_omp=candidate.tile_omp,
    )


def set_cli_kv(argv: list[str], key: str, value: Any) -> None:
    """Remove all existing '--key value' pairs and append the final pair.

    This avoids duplicate CLI keys and makes generated commands deterministic.
    """
    index = 0

    while index < len(argv):
        if argv[index] == key:
            del argv[index : min(index + 2, len(argv))]
            continue
        index += 1

    argv.extend([key, str(value)])


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_metrics(text: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {key: None for key in METRICS_RE}

    for key, regex in METRICS_RE.items():
        match = regex.search(text)
        if not match:
            continue

        try:
            metrics[key] = float(match.group(1))
        except ValueError:
            metrics[key] = None

    return metrics


def parse_detections(text: str) -> int | None:
    for regex in DETECTIONS_RE:
        match = regex.search(text)
        if not match:
            continue

        try:
            return int(match.group(1))
        except ValueError:
            return None

    return None


def metrics_have_required_latency(metrics: dict[str, float | None]) -> bool:
    return metrics.get("p50_ms") is not None and metrics.get("p99_ms") is not None


def quad_to_box(values: Sequence[float]) -> Box | None:
    if len(values) != 8:
        return None

    xs = [float(values[index]) for index in range(0, 8, 2)]
    ys = [float(values[index]) for index in range(1, 8, 2)]

    box = Box(min(xs), min(ys), max(xs), max(ys))

    if box.area <= EPS:
        return None

    return box


def parse_boxes_from_text(text: str) -> list[Box]:
    boxes: list[Box] = []

    for match in INDEXED_QUAD_LINE_RE.finditer(text):
        nums = [float(match.group(index)) for index in range(1, 9)]
        box = quad_to_box(nums)

        if box:
            boxes.append(box.normalized())

    deduped: list[Box] = []
    seen: set[tuple[int, int, int, int]] = set()

    for box in boxes:
        normalized = box.normalized()
        key = (
            round(normalized.x1),
            round(normalized.y1),
            round(normalized.x2),
            round(normalized.y2),
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(normalized)

    return deduped


def intersection(a: Box, b: Box) -> Box | None:
    """Return real rectangle intersection.

    Important: do NOT normalize after max/min.
    If x2 < x1 or y2 < y1, boxes do not intersect.
    Normalizing at that point would create fake intersections.
    """
    aa = a.normalized()
    bb = b.normalized()

    x1 = max(aa.x1, bb.x1)
    y1 = max(aa.y1, bb.y1)
    x2 = min(aa.x2, bb.x2)
    y2 = min(aa.y2, bb.y2)

    if x2 <= x1 + EPS or y2 <= y1 + EPS:
        return None

    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def union_area_rectangles(rects: Sequence[Box]) -> float:
    boxes = [box.normalized() for box in rects if box.area > EPS]

    if not boxes:
        return 0.0

    xs = sorted({coord for box in boxes for coord in (box.x1, box.x2)})
    area = 0.0

    for left, right in zip(xs, xs[1:]):
        width = right - left

        if width <= EPS:
            continue

        mid_x = (left + right) * 0.5
        intervals = [
            (box.y1, box.y2)
            for box in boxes
            if box.x1 - EPS <= mid_x <= box.x2 + EPS
        ]

        if not intervals:
            continue

        intervals.sort()
        cur_start, cur_end = intervals[0]
        merged_len = 0.0

        for start, end in intervals[1:]:
            if start <= cur_end + EPS:
                cur_end = max(cur_end, end)
            else:
                merged_len += max(0.0, cur_end - cur_start)
                cur_start, cur_end = start, end

        merged_len += max(0.0, cur_end - cur_start)
        area += width * merged_len

    return area


def area_quality(
    reference_boxes: Sequence[Box],
    candidate_boxes: Sequence[Box],
) -> AreaQuality:
    reference_area = union_area_rectangles(reference_boxes)
    candidate_area = union_area_rectangles(candidate_boxes)

    if reference_area <= EPS:
        return AreaQuality(
            area_recall=None,
            area_precision=None,
            area_f1=None,
            intersection_area=0.0,
            reference_area=reference_area,
            candidate_area=candidate_area,
            extra_area=None,
            extra_area_ratio=None,
        )

    intersections: list[Box] = []

    for reference_box in reference_boxes:
        for candidate_box in candidate_boxes:
            inter = intersection(reference_box, candidate_box)
            if inter is not None:
                intersections.append(inter)

    intersection_area = union_area_rectangles(intersections)

    area_recall = intersection_area / reference_area
    area_recall = max(0.0, min(1.0, area_recall))

    if candidate_area > EPS:
        area_precision = intersection_area / candidate_area
    else:
        area_precision = 0.0

    area_precision = max(0.0, min(1.0, area_precision))

    if area_recall + area_precision > EPS:
        area_f1 = 2.0 * area_recall * area_precision / (
            area_recall + area_precision
        )
    else:
        area_f1 = 0.0

    area_f1 = max(0.0, min(1.0, area_f1))

    extra_area = max(0.0, candidate_area - intersection_area)
    extra_area_ratio = extra_area / reference_area

    return AreaQuality(
        area_recall=area_recall,
        area_precision=area_precision,
        area_f1=area_f1,
        intersection_area=intersection_area,
        reference_area=reference_area,
        candidate_area=candidate_area,
        extra_area=extra_area,
        extra_area_ratio=extra_area_ratio,
    )


def run_command(
    cmd: Sequence[str],
    *,
    timeout_s: float,
    env: dict[str, str] | None = None,
) -> tuple[str, str, int | None, float]:
    started = time.perf_counter()

    try:
        proc = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
            env=env,
        )

        elapsed = time.perf_counter() - started
        output = proc.stdout or ""
        status = "ok" if proc.returncode == 0 else "failed"

        return output, status, proc.returncode, elapsed

    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        output = exc.stdout or ""
        error = exc.stderr or ""

        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")

        if isinstance(error, bytes):
            error = error.decode("utf-8", errors="replace")

        if error:
            output += "\n" + error

        return output, "timeout", None, elapsed


def build_command(base_cmd: Sequence[str], candidate: Candidate) -> list[str]:
    cmd = list(base_cmd)

    for key, value in candidate.to_cli_kv().items():
        set_cli_kv(cmd, f"--{key}", value)

    return cmd


def build_box_command(
    base_cmd: Sequence[str],
    candidate: Candidate,
    args: argparse.Namespace,
) -> list[str]:
    cmd = build_command(base_cmd, candidate)

    set_cli_kv(cmd, "--warmup_iters", args.box_warmup_iters)
    set_cli_kv(cmd, "--bench_iters", args.box_bench_iters)
    set_cli_kv(cmd, "--is_draw", int(args.box_is_draw))
    set_cli_kv(cmd, "--is_dump", int(args.box_is_dump))
    set_cli_kv(cmd, "--verbose", int(args.box_verbose))

    return cmd


def max_tile_hw(ref_h: int, ref_w: int, tiles_rc: str) -> tuple[int, int]:
    rows, cols = parse_tiles_rc(tiles_rc)
    return floor_to_multiple(ref_h // rows), floor_to_multiple(ref_w // cols)


def generate_single_candidates(
    *,
    ref_h: int,
    ref_w: int,
    fixed_scales: Sequence[float],
    fixed_hw: Sequence[str],
    max_img_sizes: Sequence[int],
    threads_intra: Sequence[int],
    threads_inter: Sequence[int],
    tile_overlap: float,
) -> list[Candidate]:
    generated_fixed_hw: list[str] = []

    for scale in fixed_scales:
        height = floor_to_multiple(int(ref_h * scale))
        width = floor_to_multiple(int(ref_w * scale))

        if height >= MIN_FIXED_SIDE and width >= MIN_FIXED_SIDE:
            generated_fixed_hw.append(f"{height}x{width}")

    all_fixed_hw = dedupe_keep_order([*fixed_hw, *generated_fixed_hw])
    candidates: list[Candidate] = []

    for hw, max_img_size, ti, te in itertools.product(
        all_fixed_hw,
        max_img_sizes,
        threads_intra,
        threads_inter,
    ):
        height, width = parse_hxw(hw, name="fixed_hw")
        effective_max_img_size = max(max_img_size, height, width)

        candidates.append(
            Candidate(
                family="single",
                tiles_rc="",
                fixed_hw=hw,
                max_img_size=effective_max_img_size,
                threads_intra=ti,
                threads_inter=te,
                tile_omp=1,
                tile_overlap=tile_overlap,
            )
        )

    return candidates


def generate_tiling_candidates(
    *,
    ref_h: int,
    ref_w: int,
    tiles_rc_values: Sequence[str],
    fixed_scales: Sequence[float],
    threads_intra: Sequence[int],
    threads_inter: Sequence[int],
    tile_omp_values: Sequence[int] | None,
    tile_overlap: float,
) -> list[Candidate]:
    candidates: list[Candidate] = []

    for tiles_rc in tiles_rc_values:
        rows, cols = parse_tiles_rc(tiles_rc)
        tile_count = rows * cols
        max_h, max_w = max_tile_hw(ref_h, ref_w, tiles_rc)

        if max_h < MIN_FIXED_SIDE or max_w < MIN_FIXED_SIDE:
            continue

        fixed_hw_values: list[str] = []

        for scale in fixed_scales:
            height = floor_to_multiple(int(max_h * scale))
            width = floor_to_multiple(int(max_w * scale))

            if height < MIN_FIXED_SIDE or width < MIN_FIXED_SIDE:
                continue

            # Important: no point using per-tile fixed_hw larger than tile size.
            if height > max_h or width > max_w:
                continue

            fixed_hw_values.append(f"{height}x{width}")

        fixed_hw_values = dedupe_keep_order(
            sorted(
                fixed_hw_values,
                key=lambda value: parse_hxw(value, name="fixed_hw"),
            )
        )

        omp_values = list(tile_omp_values) if tile_omp_values else [tile_count]

        for hw, ti, te, tile_omp in itertools.product(
            fixed_hw_values,
            threads_intra,
            threads_inter,
            omp_values,
        ):
            height, width = parse_hxw(hw, name="fixed_hw")

            if height > max_h or width > max_w:
                continue

            candidates.append(
                Candidate(
                    family="tiling",
                    tiles_rc=tiles_rc,
                    fixed_hw=hw,
                    max_img_size=None,
                    threads_intra=ti,
                    threads_inter=te,
                    tile_omp=tile_omp,
                    tile_overlap=tile_overlap,
                )
            )

    return candidates


def default_tiles_rc(max_threads: int) -> list[str]:
    values: list[str] = []

    for rows in range(1, 7):
        for cols in range(1, 7):
            tile_count = rows * cols

            if tile_count <= 1:
                continue

            if tile_count + 1 > max_threads:
                continue

            values.append(f"{rows}x{cols}")

    return sorted(
        values,
        key=lambda value: (parse_tiles_rc(value)[0] * parse_tiles_rc(value)[1], value),
    )


def pick_reference_single(
    *,
    ref_h: int,
    ref_w: int,
    max_img_sizes: Sequence[int],
    tile_overlap: float,
) -> Candidate:
    height = floor_to_multiple(ref_h)
    width = floor_to_multiple(ref_w)

    if height < MIN_FIXED_SIDE or width < MIN_FIXED_SIDE:
        die(f"reference single fixed_hw is too small: {height}x{width}")

    max_img_size = (
        max([max(height, width), *max_img_sizes])
        if max_img_sizes
        else max(height, width)
    )

    return Candidate(
        family="single",
        tiles_rc="",
        fixed_hw=f"{height}x{width}",
        max_img_size=max_img_size,
        threads_intra=1,
        threads_inter=1,
        tile_omp=1,
        tile_overlap=tile_overlap,
    )


def compute_reference_boxes(
    *,
    label: str,
    base_cmd: Sequence[str],
    candidate: Candidate,
    args: argparse.Namespace,
    timeout_s: float,
    log_dir: Path,
    env: dict[str, str],
) -> list[Box]:
    cmd = build_box_command(base_cmd, candidate, args)
    log(f"[REF] {label}: {shell_join(cmd)}")

    output, status, returncode, elapsed = run_command(
        cmd,
        timeout_s=timeout_s,
        env=env,
    )

    write_text(log_dir / f"reference_{label}.log", output)

    if status != "ok":
        die(
            f"reference {label} failed: "
            f"status={status}, returncode={returncode}, elapsed_s={elapsed:.3f}"
        )

    boxes = parse_boxes_from_text(output)

    if not boxes:
        die(
            f"reference {label} produced no parseable boxes. "
            "Expected lines like: 1 -> x0,y0 x1,y1 x2,y2 x3,y3"
        )

    area = union_area_rectangles(boxes)
    log(
        f"[REF] {label}: "
        f"boxes={len(boxes)}, union_area={area:.3f}, elapsed_s={elapsed:.3f}"
    )

    return boxes


def metric_cell(metrics: dict[str, float | None], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return ""
    return f"{value:.6f}"


def float_cell(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def sort_key(result: RunResult) -> tuple[float, float, float, int]:
    p99 = result.metrics.get("p99_ms")
    p95 = result.metrics.get("p95_ms")
    p50 = result.metrics.get("p50_ms")

    return (
        math.inf if p99 is None else float(p99),
        math.inf if p95 is None else float(p95),
        math.inf if p50 is None else float(p50),
        result.index,
    )


def write_csv(
    *,
    path: Path,
    target: str,
    rows: Sequence[RunResult],
    reference_boxes_by_family: dict[str, list[Box]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)

        for rank, result in enumerate(sorted(rows, key=sort_key), start=1):
            candidate = result.candidate
            ref_boxes = reference_boxes_by_family.get(candidate.family, [])
            quality = result.quality

            writer.writerow(
                [
                    rank,
                    target,
                    candidate.family,
                    result.status,
                    result.skip_reason,
                    metric_cell(result.metrics, "p99_ms"),
                    metric_cell(result.metrics, "p95_ms"),
                    metric_cell(result.metrics, "p90_ms"),
                    metric_cell(result.metrics, "p50_ms"),
                    float_cell(None if quality is None else quality.area_recall),
                    float_cell(None if quality is None else quality.area_precision),
                    float_cell(None if quality is None else quality.area_f1),
                    float_cell(None if quality is None else quality.intersection_area),
                    float_cell(None if quality is None else quality.reference_area),
                    float_cell(None if quality is None else quality.candidate_area),
                    float_cell(None if quality is None else quality.extra_area),
                    float_cell(None if quality is None else quality.extra_area_ratio),
                    result.boxes_n,
                    len(ref_boxes),
                    "" if result.dets_n is None else result.dets_n,
                    candidate.tiles_rc or "single",
                    candidate.fixed_hw,
                    "" if candidate.max_img_size is None else candidate.max_img_size,
                    candidate.threads_intra,
                    candidate.threads_inter,
                    candidate.tile_omp,
                    result.desired_threads,
                    f"{candidate.tile_overlap:.6f}",
                    f"{result.elapsed_s_python:.6f}",
                    shell_join(result.command),
                    shell_join(result.box_command),
                    result.stdout_path,
                    result.boxes_stdout_path,
                ]
            )


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def make_base_cmd(args: argparse.Namespace) -> list[str]:
    defaults = TARGET_DEFAULTS[args.target]

    cmd = [
        args.exe,
        "--mode",
        args.target,
        "--model",
        args.model,
        "--image",
        args.image,
        "--bind_io",
        str(int(args.bind_io)),
        "--runtime_policy",
        str(int(args.runtime_policy)),
        "--soft_mem_bind",
        str(int(args.soft_mem_bind)),
        "--cpu_placement",
        args.cpu_placement,
        "--suppress_opencv",
        str(int(args.suppress_opencv)),
        "--bench_iters",
        str(args.bench_iters),
        "--warmup_iters",
        str(args.warmup_iters),
        "--is_draw",
        str(int(args.is_draw)),
        "--is_dump",
        str(int(args.is_dump)),
        "--verbose",
        str(int(args.verbose)),
        "--bin_thresh",
        str(defaults.bin_thresh),
        "--box_thresh",
        str(defaults.box_thresh),
        "--unclip",
        str(defaults.unclip),
        "--nms_iou",
        str(defaults.nms_iou),
        "--use_fast_iou",
        str(defaults.use_fast_iou),
        "--min_roi_size_w",
        str(defaults.min_roi_size_w),
        "--min_roi_size_h",
        str(defaults.min_roi_size_h),
        "--sigmoid",
        str(defaults.apply_sigmoid),
    ]

    if args.extra:
        cmd.extend(args.extra)

    return cmd


def make_candidates(args: argparse.Namespace) -> tuple[list[Candidate], int]:
    ref_h, ref_w = resolve_ref_hw(args.ref_hw, Path(args.image))
    candidates: list[Candidate] = []

    if args.mode in ("single", "both"):
        candidates.extend(
            generate_single_candidates(
                ref_h=ref_h,
                ref_w=ref_w,
                fixed_scales=args.single_scales,
                fixed_hw=args.single_fixed_hw,
                max_img_sizes=args.single_max_img_size,
                threads_intra=args.single_threads_intra,
                threads_inter=args.single_threads_inter,
                tile_overlap=args.tile_overlap,
            )
        )

    if args.mode in ("tiling", "both"):
        tiles_rc_values = args.tiles_rc if args.tiles_rc else default_tiles_rc(
            args.max_threads
        )

        candidates.extend(
            generate_tiling_candidates(
                ref_h=ref_h,
                ref_w=ref_w,
                tiles_rc_values=tiles_rc_values,
                fixed_scales=args.tiling_scales,
                threads_intra=args.tiling_threads_intra,
                threads_inter=args.tiling_threads_inter,
                tile_omp_values=args.tile_omp,
                tile_overlap=args.tile_overlap,
            )
        )

    filtered: list[Candidate] = []
    seen: set[tuple[Any, ...]] = set()
    skipped_threads = 0

    for candidate in candidates:
        key = (
            candidate.family,
            candidate.tiles_rc,
            candidate.fixed_hw,
            candidate.max_img_size,
            candidate.threads_intra,
            candidate.threads_inter,
            candidate.tile_omp,
            candidate.tile_overlap,
        )

        if key in seen:
            continue

        seen.add(key)

        if candidate_desired_threads(candidate) > args.max_threads:
            skipped_threads += 1
            continue

        filtered.append(candidate)

    return filtered, skipped_threads


def latency_skip_reason(
    metrics: dict[str, float | None],
    max_p99_ms: float | None,
) -> str:
    if not metrics_have_required_latency(metrics):
        return "required latency metrics missing"

    if max_p99_ms is None:
        return ""

    p99 = metrics.get("p99_ms")

    if p99 is not None and p99 > max_p99_ms:
        return f"p99_ms={p99:.6f} > max_p99_ms={max_p99_ms:.6f}"

    return ""


def quality_skip_reason(quality: AreaQuality, args: argparse.Namespace) -> str:
    if quality.area_recall is None:
        return "reference area is zero"

    if quality.area_recall < args.min_coverage:
        return (
            f"area_recall={quality.area_recall:.6f} "
            f"< min_coverage={args.min_coverage:.6f}"
        )

    if quality.area_precision is None:
        return "area_precision is None"

    if quality.area_precision < args.min_area_precision:
        return (
            f"area_precision={quality.area_precision:.6f} "
            f"< min_area_precision={args.min_area_precision:.6f}"
        )

    if quality.area_f1 is None:
        return "area_f1 is None"

    if quality.area_f1 < args.min_area_f1:
        return (
            f"area_f1={quality.area_f1:.6f} "
            f"< min_area_f1={args.min_area_f1:.6f}"
        )

    if quality.extra_area_ratio is None:
        return "extra_area_ratio is None"

    if quality.extra_area_ratio > args.max_extra_area_ratio:
        return (
            f"extra_area_ratio={quality.extra_area_ratio:.6f} "
            f"> max_extra_area_ratio={args.max_extra_area_ratio:.6f}"
        )

    return ""


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()

    if args.disable_lsan:
        existing = env.get("ASAN_OPTIONS", "")
        env["ASAN_OPTIONS"] = (
            f"{existing}:detect_leaks=0" if existing else "detect_leaks=0"
        )

    return env


def run_search(args: argparse.Namespace) -> None:
    for path, name in (
        (Path(args.exe), "exe"),
        (Path(args.model), "model"),
        (Path(args.image), "image"),
    ):
        if not path.exists():
            die(f"{name} not found: {path}")

    out_path = Path(args.out)
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else out_path.with_suffix("").with_name(out_path.stem + "_logs")
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_path.with_suffix(".jsonl")

    if jsonl_path.exists() and not args.append_jsonl:
        jsonl_path.unlink()

    env = build_env(args)
    base_cmd = make_base_cmd(args)
    candidates, skipped_threads = make_candidates(args)

    stats = SearchStats(
        total_candidates=len(candidates),
        skipped_by_threads=skipped_threads,
    )

    if args.max_runs > 0:
        candidates = candidates[: args.max_runs]

    if args.dry_run:
        for candidate in candidates:
            log(shell_join(build_command(base_cmd, candidate)))

        log(
            f"[DRY-RUN] candidates={len(candidates)}, "
            f"skipped_by_threads={skipped_threads}"
        )
        return

    ref_h, ref_w = resolve_ref_hw(args.ref_hw, Path(args.image))
    log(f"[INFO] Reference HW: {ref_h}x{ref_w}")

    reference_boxes_by_family: dict[str, list[Box]] = {}

    # One reference policy for all candidate families:
    # full-image single-shot with fixed_hw ~= original image HxW.
    #
    # This makes single and tiling candidates comparable against the same
    # high-quality reference instead of comparing tiling against a separate
    # tiled reference. In particular, tiling is now evaluated as an approximation
    # of the full-frame detector output.
    full_image_reference_boxes = compute_reference_boxes(
        label="single_shot_full",
        base_cmd=base_cmd,
        candidate=pick_reference_single(
            ref_h=ref_h,
            ref_w=ref_w,
            max_img_sizes=args.single_max_img_size,
            tile_overlap=args.tile_overlap,
        ),
        args=args,
        timeout_s=args.ref_timeout,
        log_dir=log_dir,
        env=env,
    )

    if args.mode in ("single", "both"):
        reference_boxes_by_family["single"] = full_image_reference_boxes

    if args.mode in ("tiling", "both"):
        reference_boxes_by_family["tiling"] = full_image_reference_boxes

    kept_results: list[RunResult] = []
    total = len(candidates)

    log(
        "[INFO] Search started: "
        f"target={args.target}, mode={args.mode}, candidates={total}, "
        f"min_coverage={args.min_coverage}, "
        f"min_area_precision={args.min_area_precision}, "
        f"min_area_f1={args.min_area_f1}, "
        f"max_extra_area_ratio={args.max_extra_area_ratio}, "
        f"max_p99_ms={args.max_p99_ms}, "
        f"skipped_by_threads={skipped_threads}, out={out_path}"
    )

    for index, candidate in enumerate(candidates, start=1):
        desired = candidate_desired_threads(candidate)
        command = build_command(base_cmd, candidate)

        output, status, returncode, elapsed = run_command(
            command,
            timeout_s=args.timeout,
            env=env,
        )

        stats.executed += 1

        if status == "timeout":
            stats.timed_out += 1
        elif status != "ok":
            stats.failed += 1

        metrics = parse_metrics(output)

        bench_log = log_dir / f"run_{index:05d}_{candidate.family}_bench.log"
        stdout_path = ""

        if args.keep_logs or status != "ok":
            write_text(bench_log, output)
            stdout_path = str(bench_log)

        skip_reason = status if status != "ok" else latency_skip_reason(
            metrics,
            args.max_p99_ms,
        )

        if skip_reason:
            if "max_p99_ms" in skip_reason:
                stats.skipped_by_latency += 1
            elif skip_reason == "required latency metrics missing":
                stats.skipped_no_metrics += 1

        dets_n: int | None = None
        boxes: list[Box] = []
        quality: AreaQuality | None = None
        box_command: list[str] = []
        boxes_stdout_path = ""
        box_status = "not_run"
        box_returncode: int | None = None
        box_elapsed = 0.0

        # If latency failed, discard immediately and do not run dump pass.
        if not skip_reason:
            box_command = build_box_command(base_cmd, candidate, args)
            box_output, box_status, box_returncode, box_elapsed = run_command(
                box_command,
                timeout_s=args.box_timeout,
                env=env,
            )

            box_log = log_dir / f"run_{index:05d}_{candidate.family}_boxes.log"

            if args.keep_logs or box_status != "ok":
                write_text(box_log, box_output)
                boxes_stdout_path = str(box_log)

            if box_status != "ok":
                skip_reason = f"box_run_{box_status}"
                stats.failed += 1
            else:
                dets_n = parse_detections(box_output)
                boxes = parse_boxes_from_text(box_output)

                if not boxes:
                    skip_reason = "candidate boxes missing"
                    stats.skipped_no_boxes += 1
                else:
                    quality = area_quality(
                        reference_boxes_by_family.get(candidate.family, []),
                        boxes,
                    )
                    skip_reason = quality_skip_reason(quality, args)

                    if skip_reason:
                        stats.skipped_by_quality += 1

        result = RunResult(
            candidate=candidate,
            index=index,
            command=command,
            box_command=box_command,
            status=status,
            returncode=returncode,
            elapsed_s_python=elapsed,
            metrics=metrics,
            dets_n=dets_n,
            boxes_n=len(boxes),
            quality=quality,
            desired_threads=desired,
            skip_reason=skip_reason,
            stdout_path=stdout_path,
            boxes_stdout_path=boxes_stdout_path,
        )

        if result.passed:
            kept_results.append(result)
            stats.kept += 1

        p99 = metrics.get("p99_ms")
        p99_text = "none" if p99 is None else f"{p99:.4f}"

        if quality is None or quality.area_recall is None:
            recall_text = "none"
            precision_text = "none"
            f1_text = "none"
        else:
            recall_text = f"{quality.area_recall:.4f}"
            precision_text = (
                "none" if quality.area_precision is None else f"{quality.area_precision:.4f}"
            )
            f1_text = "none" if quality.area_f1 is None else f"{quality.area_f1:.4f}"

        status_text = "KEEP" if result.passed else f"SKIP: {skip_reason}"

        log(
            f"[RUN] {index}/{total} {candidate.family} "
            f"tiles={candidate.tiles_rc or 'single'} fixed={candidate.fixed_hw} "
            f"ti={candidate.threads_intra} te={candidate.threads_inter} "
            f"omp={candidate.tile_omp} desired={desired} "
            f"p99={p99_text} boxes={len(boxes)} "
            f"recall={recall_text} precision={precision_text} f1={f1_text} "
            f"-> {status_text}"
        )

        append_jsonl(
            jsonl_path,
            {
                "index": index,
                "target": args.target,
                "family": candidate.family,
                "candidate": candidate.to_cli_kv(),
                "desired_threads": desired,
                "status": status,
                "returncode": returncode,
                "elapsed_s_python": elapsed,
                "metrics": metrics,
                "dets_n": dets_n,
                "boxes_n": len(boxes),
                "quality": None if quality is None else quality.__dict__,
                "skip_reason": skip_reason,
                "command": command,
                "box_command": box_command,
                "box_status": box_status,
                "box_returncode": box_returncode,
                "box_elapsed_s_python": box_elapsed,
                "stdout_path": stdout_path,
                "boxes_stdout_path": boxes_stdout_path,
            },
        )

        write_csv(
            path=out_path,
            target=args.target,
            rows=kept_results,
            reference_boxes_by_family=reference_boxes_by_family,
        )

    write_csv(
        path=out_path,
        target=args.target,
        rows=kept_results,
        reference_boxes_by_family=reference_boxes_by_family,
    )

    elapsed_total = time.time() - stats.started_at

    if kept_results:
        best = sorted(kept_results, key=sort_key)[0]
        best_p99 = best.metrics.get("p99_ms")
        best_p99_text = "none" if best_p99 is None else f"{best_p99:.4f}"

        best_quality = best.quality
        best_f1 = (
            "none"
            if best_quality is None or best_quality.area_f1 is None
            else f"{best_quality.area_f1:.4f}"
        )

        log(
            "[BEST] "
            f"p99_ms={best_p99_text}, area_f1={best_f1}, "
            f"family={best.candidate.family}, "
            f"tiles_rc={best.candidate.tiles_rc or 'single'}, "
            f"fixed_hw={best.candidate.fixed_hw}, "
            f"threads_intra={best.candidate.threads_intra}, "
            f"threads_inter={best.candidate.threads_inter}, "
            f"tile_omp={best.candidate.tile_omp}"
        )
        log(f"[BEST_CMD] {shell_join(best.command)}")
    else:
        warn("no configurations passed filters")

    log(
        f"[OK] Saved sorted CSV: {out_path.resolve()}\n"
        f"[OK] Saved JSONL trace: {jsonl_path.resolve()}\n"
        f"[SUMMARY] executed={stats.executed}, kept={stats.kept}, "
        f"failed={stats.failed}, timeout={stats.timed_out}, "
        f"skipped_latency={stats.skipped_by_latency}, "
        f"skipped_quality={stats.skipped_by_quality}, "
        f"skipped_no_boxes={stats.skipped_no_boxes}, "
        f"skipped_no_metrics={stats.skipped_no_metrics}, "
        f"skipped_threads={stats.skipped_by_threads}, "
        f"elapsed_s={elapsed_total:.2f}"
    )

    if len(kept_results) >= 2:
        p99_values = [
            item.metrics["p99_ms"]
            for item in kept_results
            if item.metrics.get("p99_ms") is not None
        ]
        f1_values = [
            item.quality.area_f1
            for item in kept_results
            if item.quality is not None and item.quality.area_f1 is not None
        ]

        if p99_values:
            log(
                "[STATS] kept p99_ms: "
                f"min={min(p99_values):.4f}, "
                f"median={statistics.median(p99_values):.4f}, "
                f"max={max(p99_values):.4f}"
            )

        if f1_values:
            log(
                "[STATS] kept area_f1: "
                f"min={min(f1_values):.4f}, "
                f"median={statistics.median(f1_values):.4f}, "
                f"max={max(f1_values):.4f}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search best IDet text/face/cloth detector configuration. "
            "The script benchmarks candidate single/tiling settings, extracts Quads "
            "from a cheap dump pass, computes area-quality against reference boxes, "
            "and writes a CSV sorted by p99_ms ascending."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------------------------------------------------------------------------
    # Required app inputs
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--exe",
        required=True,
        help=(
            "Path to idet_app executable. Example: "
            "build_gcc_perf/src/app/idet/idet_app"
        ),
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_DEFAULTS),
        required=True,
        help=(
            "Detector target / ROI type. This value is passed to idet_app as "
            "--mode. Also selects default thresholds for text, face, or cloth."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to ONNX model used by idet_app.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help=(
            "Path to input image. If --ref-hw auto is used, the script reads "
            "image HxW from this PNG/JPEG file header."
        ),
    )
    parser.add_argument(
        "--out",
        default="result.csv",
        help=(
            "Output CSV path. The file is rewritten incrementally after each "
            "accepted candidate and is always sorted by p99_ms ascending."
        ),
    )
    parser.add_argument(
        "--log-dir",
        default="",
        help=(
            "Directory for reference logs and optional per-run logs. If empty, "
            "the directory is derived from --out, e.g. result.csv -> result_logs/."
        ),
    )

    # -------------------------------------------------------------------------
    # Search scope
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--mode",
        choices=["single", "tiling", "both"],
        default="both",
        help=(
            "Which candidate family to search. single means no --tiles_rc is "
            "passed to idet_app. tiling means candidates use --tiles_rc and "
            "--tile_overlap. both runs both families."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print generated benchmark commands and exit without running idet_app. "
            "Useful for checking the candidate grid."
        ),
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help=(
            "Limit number of generated candidates after filtering by --max-threads. "
            "0 means no limit. Useful for smoke tests."
        ),
    )

    # -------------------------------------------------------------------------
    # Reference geometry
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--ref-hw",
        default=DEFAULT_REF_HW,
        help=(
            "Reference image size as HxW, e.g. 1080x1920. Use 'auto' to read "
            "HxW from the input PNG/JPEG header. This size is used for the "
            "full-image single-shot reference, single fixed_hw generation, and "
            "tiling tile bounds."
        ),
    )
    
    # -------------------------------------------------------------------------
    # Runtime limits and timeouts
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--max-threads",
        type=int,
        default=32,
        help=(
            "Maximum allowed desired thread count for a candidate. The script "
            "uses a conservative estimate: tile_omp + max(inter, intra), or "
            "tile_omp + inter + intra when both inter and intra are > 1. "
            "Candidates above this limit are skipped before running."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds for the main benchmark run of one candidate.",
    )
    parser.add_argument(
        "--ref-timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds for each reference box extraction run.",
    )
    parser.add_argument(
        "--box-timeout",
        type=float,
        default=60.0,
        help=(
            "Timeout in seconds for the cheap candidate dump pass used to "
            "extract Quads after the candidate has passed the latency filter."
        ),
    )

    # -------------------------------------------------------------------------
    # Quality filters
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.8,
        help=(
            "Minimum area recall. Formula: intersection_area / reference_area. "
            "This checks how much of the reference ROI area is covered by the "
            "candidate boxes."
        ),
    )
    parser.add_argument(
        "--min-area-precision",
        type=float,
        default=0.6,
        help=(
            "Minimum area precision. Formula: intersection_area / candidate_area. "
            "This penalizes oversized candidate boxes that cover the reference "
            "but also add too much extra area."
        ),
    )
    parser.add_argument(
        "--min-area-f1",
        type=float,
        default=0.7,
        help=(
            "Minimum area F1 score. Harmonic mean of area recall and area "
            "precision. Useful as a single balanced quality score."
        ),
    )
    parser.add_argument(
        "--max-extra-area-ratio",
        type=float,
        default=0.5,
        help=(
            "Maximum allowed extra candidate area relative to reference area. "
            "Formula: (candidate_area - intersection_area) / reference_area. "
            "Lower values are stricter."
        ),
    )
    parser.add_argument(
        "--max-p99-ms",
        type=float,
        default=20.0,
        help=(
            "Hard latency filter. If candidate p99_ms from idet_app stdout is "
            "greater than this value, the candidate is dropped immediately and "
            "the expensive/extra dump pass is not run. None disables this filter."
        ),
    )

    # -------------------------------------------------------------------------
    # Main benchmark pass
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=30,
        help=(
            "Number of measured iterations for the main benchmark pass. "
            "p50/p90/p95/p99 are parsed from idet_app stdout."
        ),
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Number of warmup iterations for the main benchmark pass.",
    )

    # -------------------------------------------------------------------------
    # Cheap dump pass for Quads / coverage
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--box-bench-iters",
        type=int,
        default=1,
        help=(
            "Measured iterations for the cheap dump pass. This pass is not used "
            "for ranking latency; it only extracts dets_n and Quads."
        ),
    )
    parser.add_argument(
        "--box-warmup-iters",
        type=int,
        default=0,
        help="Warmup iterations for the cheap dump pass.",
    )
    parser.add_argument(
        "--box-verbose",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Verbose flag for the cheap dump pass. Keep 0 if idet_app prints "
            "Quads with --is_dump 1."
        ),
    )
    parser.add_argument(
        "--box-is-draw",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "is_draw flag for the cheap dump pass. Usually 0 because drawing "
            "is not needed for coverage computation."
        ),
    )
    parser.add_argument(
        "--box-is-dump",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "is_dump flag for the cheap dump pass. For the current idet_app, "
            "this must be 1 because Quads are printed when dump is enabled."
        ),
    )

    # -------------------------------------------------------------------------
    # Single-mode candidate grid
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--single-scales",
        type=parse_csv_floats,
        default=parse_csv_floats(
            "0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,"
            "0.70,0.75,0.80,0.85,0.90,0.95,1.00"
        ),
        help=(
            "Comma-separated scales used to generate single fixed_hw values "
            "from reference HxW. Example: 0.5 creates fixed_hw around half "
            "the reference height and width, rounded down to a multiple of 32."
        ),
    )
    parser.add_argument(
        "--single-fixed-hw",
        type=parse_csv_strings,
        default=[],
        help=(
            "Comma-separated explicit fixed_hw values for single mode, e.g. "
            "512x960,640x1152. These are added in addition to --single-scales."
        ),
    )
    parser.add_argument(
        "--single-max-img-size",
        type=parse_csv_ints,
        default=[960],
        help=(
            "Comma-separated max_img_size values for single mode. The script "
            "automatically raises max_img_size to at least max(fixed_h, fixed_w) "
            "to avoid invalid fixed_hw/max_img_size combinations."
        ),
    )
    parser.add_argument(
        "--single-threads-intra",
        type=parse_csv_ints,
        default=parse_csv_ints("1,3,5,7,9"),
        help="Comma-separated ORT intra-op thread counts for single mode.",
    )
    parser.add_argument(
        "--single-threads-inter",
        type=parse_csv_ints,
        default=[1],
        help="Comma-separated ORT inter-op thread counts for single mode.",
    )

    # -------------------------------------------------------------------------
    # Tiling-mode candidate grid
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--tiles-rc",
        type=parse_csv_strings,
        default=[],
        help=(
            "Comma-separated candidate tiling grids, e.g. 2x2,3x3,2x4. "
            "If empty, the script auto-generates practical grids up to 6x6 "
            "under --max-threads."
        ),
    )
    parser.add_argument(
        "--tiling-scales",
        type=parse_csv_floats,
        default=parse_csv_floats("1.00,0.85,0.70,0.60,0.50,0.40,0.30"),
        help=(
            "Comma-separated scales for per-tile fixed_hw. Scale 1.0 means "
            "fixed_hw is the tile size rounded down to a multiple of 32. "
            "The script rejects tiling fixed_hw values larger than the tile size."
        ),
    )
    parser.add_argument(
        "--tiling-threads-intra",
        type=parse_csv_ints,
        default=parse_csv_ints("1,2,4,8"),
        help="Comma-separated ORT intra-op thread counts for tiling mode.",
    )
    parser.add_argument(
        "--tiling-threads-inter",
        type=parse_csv_ints,
        default=[1],
        help="Comma-separated ORT inter-op thread counts for tiling mode.",
    )
    parser.add_argument(
        "--tile-omp",
        type=parse_csv_ints,
        default=[],
        help=(
            "Comma-separated OpenMP thread counts for tiling. If empty, the "
            "script uses rows * cols for each tiles_rc candidate."
        ),
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=DEFAULT_TILE_OVERLAP,
        help="Tile overlap used for generated tiling candidates.",
    )

    # -------------------------------------------------------------------------
    # Common idet_app runtime flags
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--bind-io",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Passes --bind_io to idet_app. Requires fixed_hw. Usually 1 for "
            "benchmarking fixed input shapes."
        ),
    )
    parser.add_argument(
        "--runtime-policy",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Passes --runtime_policy to idet_app. Enables runtime policy setup "
            "such as CPU placement, memory policy, and OpenCV thread suppression."
        ),
    )
    parser.add_argument(
        "--soft-mem-bind",
        type=int,
        choices=[0, 1],
        default=1,
        help="Passes --soft_mem_bind to idet_app for best-effort memory locality.",
    )
    parser.add_argument(
        "--cpu-placement",
        choices=["latency", "throughput"],
        default="latency",
        help=(
            "Passes --cpu_placement to idet_app. latency usually keeps execution "
            "compact; throughput may spread work more broadly depending on app logic."
        ),
    )
    parser.add_argument(
        "--suppress-opencv",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Passes --suppress_opencv to idet_app. Usually 1 to prevent OpenCV "
            "from using its own thread pool during benchmark runs."
        ),
    )

    # -------------------------------------------------------------------------
    # Main benchmark output flags
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--is-draw",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Main benchmark pass --is_draw. Keep 0 for clean latency measurement."
        ),
    )
    parser.add_argument(
        "--is-dump",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Main benchmark pass --is_dump. Keep 0 so p50/p90/p95/p99 are "
            "measured without dump overhead."
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Main benchmark pass --verbose. Keep 0 for clean output and lower "
            "stdout overhead."
        ),
    )

    # -------------------------------------------------------------------------
    # Diagnostics / logs
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--keep-logs",
        action="store_true",
        help=(
            "Keep stdout logs for every benchmark and dump run. By default, "
            "logs are kept only for failed runs and references."
        ),
    )
    parser.add_argument(
        "--append-jsonl",
        action="store_true",
        help=(
            "Append to existing JSONL trace instead of deleting it at startup. "
            "Useful for manual continuation, but can mix old and new runs."
        ),
    )
    parser.add_argument(
        "--disable-lsan",
        action="store_true",
        help=(
            "Set ASAN_OPTIONS=detect_leaks=0 for child idet_app processes. "
            "Useful only for sanitizer builds; perf builds should not use ASan."
        ),
    )

    # -------------------------------------------------------------------------
    # Raw idet_app overrides
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help=(
            "Extra raw idet_app arguments appended to every command. Put this "
            "option last. Example: --extra --box_thresh 0.01 --nms_iou 0.5. "
            "If an extra key duplicates an earlier key, later set_cli_kv calls "
            "for candidate-specific options may replace it."
        ),
    )

    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.max_threads <= 0:
        die("--max-threads must be > 0")

    if args.bench_iters <= 0 or args.box_bench_iters <= 0:
        die("bench iters must be > 0")

    if args.warmup_iters < 0 or args.box_warmup_iters < 0:
        die("warmup iters must be >= 0")

    for name in (
        "min_coverage",
        "min_area_precision",
        "min_area_f1",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            die(f"--{name.replace('_', '-')} must be in [0, 1]")

    if args.max_extra_area_ratio < 0.0:
        die("--max-extra-area-ratio must be >= 0")

    if args.tile_overlap < 0.0:
        die("--tile-overlap must be >= 0")

    if args.timeout <= 0 or args.ref_timeout <= 0 or args.box_timeout <= 0:
        die("timeouts must be > 0")

    if args.ref_hw.lower() != "auto":
        parse_hxw(args.ref_hw, name="ref_hw")

    for hw in args.single_fixed_hw:
        parse_hxw(hw, name="single_fixed_hw")

    for tiles_rc in args.tiles_rc:
        parse_tiles_rc(tiles_rc)

    args.exe = str(Path(args.exe))
    args.model = str(Path(args.model))
    args.image = str(Path(args.image))

    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]

    return args


def main() -> None:
    args = normalize_args(build_arg_parser().parse_args())
    run_search(args)


if __name__ == "__main__":
    main()
