#!/usr/bin/env python3

import argparse
import csv
import itertools
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


METRICS_RE = {
    "p50_ms": re.compile(r"\bp50_ms\s*:\s*([0-9]+(?:\.[0-9]+)?)"),
    "p90_ms": re.compile(r"\bp90_ms\s*:\s*([0-9]+(?:\.[0-9]+)?)"),
    "p95_ms": re.compile(r"\bp95_ms\s*:\s*([0-9]+(?:\.[0-9]+)?)"),
    "p99_ms": re.compile(r"\bp99_ms\s*:\s*([0-9]+(?:\.[0-9]+)?)"),
    "dets_n": re.compile(r"\bdets_n\s*:\s*([0-9]+)\b"),
}

MAX_DETECTION_TIME_MS = 20.0


def calc_desired_threads(inter: int, intra: int, omp: int) -> int:
    ort_peak = 1
    if inter > 1 and intra > 1:
        ort_peak = inter + intra
    else:
        ort_peak = max(inter, intra)
    return omp + ort_peak


def passes_max_threads(kv: Dict[str, Any], max_threads: int) -> Tuple[bool, int]:
    ti = int(kv.get("threads_intra", 0) or 0)
    te = int(kv.get("threads_inter", 0) or 0)
    omp = int(kv.get("tile_omp", 0) or 0)
    desired = calc_desired_threads(te, ti, omp)
    return (desired <= int(max_threads)), desired


def parse_metrics(text: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in METRICS_RE.keys()}
    for k, rx in METRICS_RE.items():
        m = rx.search(text)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                out[k] = None
    return out


def dets_n_int(metrics: Dict[str, Optional[float]]) -> Optional[int]:
    v = metrics.get("dets_n")
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def floor_to_multiple(x: int, m: int) -> int:
    if m <= 0:
        return x
    return (x // m) * m


def parse_tiles_rc(s: str) -> Tuple[int, int]:
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Bad tiles_rc '{s}', expected like '3x3'")
    r = int(parts[0])
    c = int(parts[1])
    if r <= 0 or c <= 0:
        raise ValueError(f"Bad tiles_rc '{s}', rows/cols must be >0")
    return r, c


def shell_join(argv: List[str]) -> str:
    if hasattr(shlex, "join"):
        return shlex.join(argv)  # py3.8+
    return " ".join(shlex.quote(a) for a in argv)


def run_one(cmd: List[str], timeout_s: float) -> Tuple[Dict[str, Optional[float]], str]:
    """
    Returns: (metrics, status)
    status: ok | timeout | no_metrics | failed
    """

    def _to_text(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        return str(x)

    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        out = p.stdout or ""
        metrics = parse_metrics(out)

        if p.returncode != 0:
            return metrics, "failed"

        if all(metrics[k] is None for k in metrics.keys()):
            return metrics, "no_metrics"

        return metrics, "ok"

    except subprocess.TimeoutExpired as e:
        out = _to_text(e.stdout)
        err = _to_text(getattr(e, "stderr", None))
        if err:
            out = out + "\n" + err
        metrics = parse_metrics(out)
        return metrics, "timeout"


def build_cmd(base_cmd: List[str], kv: Dict[str, Any]) -> List[str]:
    cmd = list(base_cmd)
    for k, v in kv.items():
        if v is None:
            continue
        cmd += [f"--{k}", str(v)]
    return cmd


def gen_single_shot(
    fixed_hw_list: List[str],
    max_img_size_list: List[int],
    threads_intra_list: List[int],
    threads_inter_list: List[int],
) -> Iterable[Dict[str, Any]]:
    for fixed_hw, max_img_size, ti, te in itertools.product(
        fixed_hw_list, max_img_size_list, threads_intra_list, threads_inter_list
    ):
        yield {
            "tiles_rc": "1x1",
            "tile_omp": 1,
            "fixed_hw": fixed_hw,
            "max_img_size": max_img_size,
            "threads_intra": ti,
            "threads_inter": te,
        }


def gen_tiling(
    tiles_rc_list: List[str],
    threads_intra_list: List[int],
    threads_inter_list: List[int],
    fhd_w: int,
    fhd_h: int,
    fixed_scales: List[float],
) -> Iterable[Dict[str, Any]]:
    for tiles_rc in tiles_rc_list:
        r, c = parse_tiles_rc(tiles_rc)

        max_h = floor_to_multiple((fhd_h // r), 32)
        max_w = floor_to_multiple((fhd_w // c), 32)
        if max_h <= 64 or max_w <= 64:
            continue

        fixed_set = set()
        for s in fixed_scales:
            hh = floor_to_multiple(int(max_h * s), 32)
            ww = floor_to_multiple(int(max_w * s), 32)
            if hh >= 64 and ww >= 64:
                fixed_set.add((hh, ww))

        fixed_hw_candidates = sorted(fixed_set, key=lambda x: (x[0], x[1]))
        tile_omp_list = [r * c]

        for (hh, ww), ti, te, to in itertools.product(
            fixed_hw_candidates, threads_intra_list, threads_inter_list, tile_omp_list
        ):
            yield {
                "tiles_rc": tiles_rc,
                "fixed_hw": f"{hh}x{ww}",
                "threads_intra": ti,
                "threads_inter": te,
                "tile_omp": to,
                "max_img_size": None,
            }


def fmt_cell(v: Optional[float], status: str) -> str:
    if status != "ok":
        return status
    if v is None:
        return "none"
    return f"{v:.4f}"


def extract_fields_from_kv(kv: Dict[str, Any]) -> Tuple[str, str, int, int, int, int]:
    """
    Returns:
      tiles_rc, fixed_hw, threads_intra, threads_inter, omp_threads, total_threads (= omp + max(intra, inter))
    """
    tiles_rc = str(kv.get("tiles_rc", "none"))
    fixed_hw = str(kv.get("fixed_hw", "none"))

    ti = int(kv.get("threads_intra", 0) or 0)
    te = int(kv.get("threads_inter", 0) or 0)

    omp_threads = int(kv.get("tile_omp", 0) or 0)
    total_threads = omp_threads + max(ti, te)

    return tiles_rc, fixed_hw, ti, te, omp_threads, total_threads


def max_hw_for_tiles_rc(fhd_h: int, fhd_w: int, tiles_rc: str) -> Tuple[int, int]:
    r, c = parse_tiles_rc(tiles_rc)
    max_h = floor_to_multiple((fhd_h // r), 32)
    max_w = floor_to_multiple((fhd_w // c), 32)
    return max_h, max_w


def pick_best_tiling_rc_by_tile_area(
    fhd_h: int, fhd_w: int, tiling_tiles_rc: List[str], max_threads: int
) -> Optional[str]:
    best_rc = None
    best_area = -1
    best_rc_prod = None

    for rc in tiling_tiles_rc:
        try:
            r, c = parse_tiles_rc(rc)
            omp = r * c
            # reference uses inter=1, intra=1 => ort_peak=1 => desired = omp + 1
            if calc_desired_threads(1, 1, omp) > max_threads:
                continue

            max_h, max_w = max_hw_for_tiles_rc(fhd_h, fhd_w, rc)
            if max_h <= 64 or max_w <= 64:
                continue

            area = max_h * max_w
            prod = omp

            if area > best_area or (area == best_area and (best_rc_prod is None or prod < best_rc_prod)):
                best_area = area
                best_rc = rc
                best_rc_prod = prod
        except Exception:
            continue
    return best_rc


def set_cli_kv(argv: List[str], key: str, value: Any) -> List[str]:
    """Set/replace '--key value' in argv; append if missing"""
    key = str(key)
    val = str(value)
    for i in range(len(argv) - 1):
        if argv[i] == key:
            argv[i + 1] = val
            return argv
    argv += [key, val]
    return argv


def compute_reference_dets(
    kind: str,
    base_cmd: List[str],
    timeout_s: float,
    fhd_h: int,
    fhd_w: int,
    single_max_img_size: List[int],
    tiling_tiles_rc: List[str],
    max_threads: int,
) -> Optional[int]:
    # IMPORTANT: reference run must be cheap
    ref_base_cmd = list(base_cmd)
    set_cli_kv(ref_base_cmd, "--warmup_iters", 1)
    set_cli_kv(ref_base_cmd, "--bench_iters", 1)
    set_cli_kv(ref_base_cmd, "--is_draw", 0)
    set_cli_kv(ref_base_cmd, "--is_dump", 0)
    set_cli_kv(ref_base_cmd, "--verbose", 0)

    if kind == "single":
        max_h = floor_to_multiple(fhd_h, 32)
        max_w = floor_to_multiple(fhd_w, 32)
        kv = {
            "tiles_rc": "1x1",
            "tile_omp": 1,
            "fixed_hw": f"{max_h}x{max_w}",
            "max_img_size": max(single_max_img_size) if single_max_img_size else None,
            "threads_intra": 1,
            "threads_inter": 1,
        }

        ok, desired = passes_max_threads(kv, max_threads)
        if not ok:
            print(f"[REF][WARN] single baseline exceeds max_threads: desired={desired} > max_threads={max_threads}")
            return None

        cmd = build_cmd(ref_base_cmd, kv)
        cmd_str = shell_join(cmd)
        print(f"[REF] single-shot baseline run: fixed_hw={kv['fixed_hw']} desired_threads={desired} cmd={cmd_str}")

        metrics, status = run_one(cmd, timeout_s=timeout_s)
        if status != "ok":
            print(f"[REF][WARN] single baseline failed: status={status}")
            return None
        d = dets_n_int(metrics)
        print(f"[REF] single-shot dets_n={d}")
        return d

    if kind == "tiling":
        best_rc = pick_best_tiling_rc_by_tile_area(fhd_h, fhd_w, tiling_tiles_rc, max_threads)
        if not best_rc:
            print("[REF][WARN] tiling baseline: cannot pick tiles_rc under max_threads cap")
            return None

        r, c = parse_tiles_rc(best_rc)
        max_h, max_w = max_hw_for_tiles_rc(fhd_h, fhd_w, best_rc)

        kv = {
            "tiles_rc": best_rc,
            "tile_omp": r * c,
            "fixed_hw": f"{max_h}x{max_w}",
            "max_img_size": None,
            "threads_intra": 1,
            "threads_inter": 1,
        }

        ok, desired = passes_max_threads(kv, max_threads)
        if not ok:
            print(f"[REF][WARN] tiling baseline exceeds max_threads: desired={desired} > max_threads={max_threads}")
            return None

        cmd = build_cmd(ref_base_cmd, kv)
        cmd_str = shell_join(cmd)
        print(
            f"[REF] tiling baseline run: tiles_rc={best_rc} fixed_hw={kv['fixed_hw']} desired_threads={desired} cmd={cmd_str}"
        )

        metrics, status = run_one(cmd, timeout_s=timeout_s)
        if status != "ok":
            print(f"[REF][WARN] tiling baseline failed: status={status}")
            return None
        d = dets_n_int(metrics)
        print(f"[REF] tiling dets_n={d} (tiles_rc={best_rc})")
        return d

    return None


def main():
    ap = argparse.ArgumentParser(
        description="Generate idet_app commands, parse p50/p90/p95/p99 + dets_n, "
        "filter by dets_n baseline, filter by max_threads cap, write CSV"
    )
    ap.add_argument("--exe", required=True, help="Path to idet_app executable")
    ap.add_argument("--model", required=True, help="Path to ONNX model")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--mode", required=True, help="Detection mode: text | face")
    ap.add_argument("--out", type=str, default="result.csv", help="Output csv path (default: result.csv)")

    ap.add_argument("--gen", choices=["single", "tiling", "both"], default="both", help="Which command family to generate")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute")
    ap.add_argument("--ref-hw", dest="ref_hw", default="1080x1920", help="Reference image size HxW for tiling bounds (default: 1080x1920)")

    ap.add_argument("--timeout", type=float, default=3.0, help="Per-run exe timeout in seconds")
    ap.add_argument("--max-runs", type=int, default=0, help="Limit number of runs (0 = no limit)")
    ap.add_argument("--max-threads", type=int, default=16, help="Max allowed desired_threads (default: 16)")
    ap.add_argument("--scale-dets", type=float, default=0.8, help="Filter: keep run only if dets_n >= ref_dets_n * scale_dets (default: 0.8)")
    ap.add_argument("--extra", type=str, default="", help="Extra args appended to command (quoted string)")

    args = ap.parse_args()

    fhd_h, fhd_w = (int(args.ref_hw.split("x")[0]), int(args.ref_hw.split("x")[1]))

    base_cmd = [
        args.exe,
        "--mode", args.mode,
        "--model", args.model,
        "--image", args.image,

        "--bind_io", "1",
        "--runtime_policy", "1",
        "--soft_mem_bind", "1",
        "--suppress_opencv", "1",

        "--bench_iters", "30",
        "--warmup_iters", "5",

        "--is_draw", "0",
        "--is_dump", "0",
        "--verbose", "0",

        "--tile_overlap", "0.1",
        "--bin_thresh", "0.3",
        "--box_thresh", "0.5",
        "--unclip", "1.0",
        "--nms_iou", "0.3",
        "--min_roi_size_w", "10",
        "--min_roi_size_h", "10",
    ]

    if args.extra.strip():
        base_cmd += shlex.split(args.extra)

    # Single mode
    single_fixed_scales = [0.3 + dt * 0.05 for dt in range(15)]
    single_fixed_hw = [
        f"{floor_to_multiple(int(fhd_h * scale), 32)}x{floor_to_multiple(int(fhd_w * scale), 32)}"
        for scale in single_fixed_scales
    ]
    single_fixed_hw = sorted({s for s in single_fixed_hw if not s.startswith("0x")})
    single_max_img_size = [960]
    single_threads_intra = [i for i in range(1, args.max_threads + 1, 2)]
    single_threads_inter = [1]

    # Tiling mode
    tiling_tiles_rc = [f"{i}x{j}" for i in range(1, 100) for j in range(1, 100) if 1 < i * j <= int(args.max_threads)]
    tiling_threads_intra = [1, 2, 4, 8, 16, 24, 32]
    tiling_threads_inter = [1]
    tiling_fixed_scales = [1.0, 0.7, 0.6, 0.5, 0.4, 0.3]

    # --------------------------
    # Reference dets_n baselines
    # --------------------------
    ref_single: Optional[int] = None
    ref_tiling: Optional[int] = None

    if not args.dry_run:
        if args.gen in ("single", "both"):
            ref_single = compute_reference_dets(
                kind="single",
                base_cmd=base_cmd,
                timeout_s=180,
                fhd_h=fhd_h,
                fhd_w=fhd_w,
                single_max_img_size=single_max_img_size,
                tiling_tiles_rc=tiling_tiles_rc,
                max_threads=int(args.max_threads),
            )

        if args.gen in ("tiling", "both"):
            ref_tiling = compute_reference_dets(
                kind="tiling",
                base_cmd=base_cmd,
                timeout_s=180,
                fhd_h=fhd_h,
                fhd_w=fhd_w,
                single_max_img_size=single_max_img_size,
                tiling_tiles_rc=tiling_tiles_rc,
                max_threads=int(args.max_threads),
            )

    def pass_dets_filter(kind: str, dets: Optional[int]) -> Tuple[bool, str]:
        if dets is None:
            return False, "dets not found"

        ref = ref_single if kind == "single" else ref_tiling
        if ref is None:
            return False, "ref_dets is None"

        thr = float(ref) * float(args.scale_dets)
        if dets < thr:
            return False, f"dets={dets} < ref_dets*{args.scale_dets}={thr:.2f}"
        return True, ""

    # --------------------------
    # Generate runs with max_threads filter
    # --------------------------
    runs: List[Tuple[str, Dict[str, Any]]] = []
    skipped_threads = 0

    if args.gen in ("single", "both"):
        for kv in gen_single_shot(single_fixed_hw, single_max_img_size, single_threads_intra, single_threads_inter):
            ok, _desired = passes_max_threads(kv, int(args.max_threads))
            if not ok:
                skipped_threads += 1
                continue
            runs.append(("single", kv))

    if args.gen in ("tiling", "both"):
        for kv in gen_tiling(tiling_tiles_rc, tiling_threads_intra, tiling_threads_inter, fhd_w, fhd_h, tiling_fixed_scales):
            ok, _desired = passes_max_threads(kv, int(args.max_threads))
            if not ok:
                skipped_threads += 1
                continue
            runs.append(("tiling", kv))

    out_path = Path(args.out)
    if out_path.parent and str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "p99_ms", "p95_ms", "p90_ms", "p50_ms",
        "dets_n",
        "tiles_rc", "fixed_hw",
        "threads_intra", "threads_inter",
        "omp_threads",
        "desired_threads",
        "command",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        done = 0
        all_runs = len(runs)

        for kind, kv in runs:
            if args.max_runs and done >= args.max_runs:
                break

            ok, desired = passes_max_threads(kv, int(args.max_threads))
            done += 1

            if not ok:
                print(f"[INFO] Progress: {done}/{all_runs} -> desired_threads={desired} > max_threads={args.max_threads} (skip)")
                continue

            cmd = build_cmd(base_cmd, kv)
            cmd_str = shell_join(cmd)

            if args.dry_run:
                print(cmd_str)
                continue

            metrics, status = run_one(cmd, timeout_s=args.timeout)
            if status != "ok":
                print(f"[INFO] Progress: {done}/{all_runs} -> status={status} (skip)")
                continue

            d = dets_n_int(metrics)
            dets_passed, filter_msg = pass_dets_filter(kind, d)
            if not dets_passed:
                print(f"[INFO] Progress: {done}/{all_runs} -> {filter_msg} (skip)")
                continue

            p90 = metrics.get("p90_ms")
            if p90 is None:
                print(f"[INFO] Progress: {done}/{all_runs} -> p90_ms missing (skip)")
                continue

            if p90 > MAX_DETECTION_TIME_MS:
                print(f"[INFO] Progress: {done}/{all_runs} -> p90_ms={p90:.4f} > {MAX_DETECTION_TIME_MS} (skip)")
                continue

            print(f"[INFO] Progress: {done}/{all_runs} -> p90_ms={p90:.4f}, dets_n={d}, desired_threads={desired}")

            tiles_rc, fixed_hw, ti, te, omp_th, _total_th = extract_fields_from_kv(kv)

            w.writerow([
                fmt_cell(metrics.get("p99_ms"), status),
                fmt_cell(metrics.get("p95_ms"), status),
                fmt_cell(metrics.get("p90_ms"), status),
                fmt_cell(metrics.get("p50_ms"), status),

                "none" if d is None else str(d),

                tiles_rc,
                fixed_hw,
                ti,
                te,
                omp_th,
                desired,

                cmd_str,
            ])
            f.flush()

    print(f"[OK] Saved: {out_path.resolve()}")
    print(f"[OK] Candidate combos: {len(runs)} (skipped_by_max_threads={skipped_threads})")


if __name__ == "__main__":
    main()
    