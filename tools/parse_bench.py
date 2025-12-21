from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


WH_KEYS = {"image_wh", "grid_wh", "tile_wh", "in_wh"}


def parse_wh(value: str) -> dict[str, int] | str:
    parts = value.lower().split("x")
    if len(parts) != 2:
        return value
    try:
        return {"w": int(parts[0]), "h": int(parts[1])}
    except ValueError:
        return value


def cast_value(key: str, value: str) -> Any:
    v = value.strip()

    if key == "tiles":
        vl = v.lower()
        if vl in ("on", "true", "1"):
            return True
        if vl in ("off", "false", "0"):
            return False
        return v

    if key in WH_KEYS:
        vl = v.lower()
        if vl in ("none", "auto"):
            return v
        return parse_wh(v)

    if v.lower().endswith("ms"):
        v_num = v[:-2].strip()
        try:
            return int(v_num)
        except ValueError:
            pass
        try:
            return float(v_num)
        except ValueError:
            return v

    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass

    return v


def parse_bench_line(line: str) -> tuple[str | None, dict[str, Any]]:
    line = line.strip()
    if not line:
        return None, {}

    if "[BENCH][CONFIG]" in line:
        section = "config"
    elif "[BENCH][ INFR ]" in line or "[BENCH][INFR]" in line:
        section = "inference"
    else:
        return None, {}

    parts = [p.strip() for p in line.split("|")]
    values: dict[str, Any] = {}

    for token in parts[1:]:
        if not token or "=" not in token:
            continue

        key, val = token.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue

        cv = cast_value(key, val)

        # If inference values came in "ms", also store them as *_ms.
        if section == "inference" and isinstance(cv, (int, float)) and val.lower().endswith("ms"):
            key = f"{key}_ms"

        values[key] = cv

    return section, values


def run_command(cmd_list: list[str]) -> tuple[str, subprocess.CompletedProcess[str]]:
    cmd_str = " ".join(cmd_list)
    result = subprocess.run(
        cmd_list,
        shell=False,
        capture_output=True,
        text=True,
    )
    return cmd_str, result


def parse_bench_from_text(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config: dict[str, Any] = {}
    inference: dict[str, Any] = {}

    for line in text.splitlines():
        section, values = parse_bench_line(line)
        if not section or not values:
            continue
        if section == "config":
            config.update(values)
        elif section == "inference":
            inference.update(values)

    return config, inference


def extract_det_count_from_text(text: str) -> int | None:
    patterns = ("dets=", "detections=", "num_dets=", "det_count=")

    for line in text.splitlines():
        for pat in patterns:
            if pat in line:
                tail = line.split(pat, 1)[1].strip()
                token = re.split(r"[|,\s]", tail)[0]
                try:
                    return int(token)
                except ValueError:
                    continue

    return None


def get_mean_ms(inference: dict[str, Any]) -> float | None:
    val = inference.get("mean_ms", inference.get("mean"))
    return float(val) if isinstance(val, (int, float)) else None


def save_config_json(
    config: dict[str, Any],
    inference: dict[str, Any],
    image_path: str,
    threads_line: str,
    det_count: int | None,
    raw_meta: dict[str, Any],
    out_root: str = ".",
) -> str:
    if not image_path:
        raise ValueError("image_path is required to save config")

    img_path = Path(image_path)
    out_dir = Path(out_root) / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_val = get_mean_ms(inference)
    mean_str = f"{mean_val:.3f}" if isinstance(mean_val, (int, float)) else "unknown_mean"
    det_str = "unknown_det" if det_count is None else str(det_count)

    file_name = f"{threads_line}___{mean_str}_{det_str}.json".replace(" ", "_")
    out_path = out_dir / file_name

    data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "image": str(image_path),
        "detections": det_count,
        "config": config,
        "inference": inference,
        "raw": raw_meta,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(out_path)


def probe_config(
    cmd: list[str],
) -> tuple[int | None, float | None, dict[str, Any], dict[str, Any], dict[str, Any]]:
    cmd_str, result = run_command(cmd)
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    text = stdout + "\n" + stderr

    config, inference = parse_bench_from_text(text)

    det_count: int | None
    if isinstance(inference.get("dets"), int):
        det_count = inference["dets"]
    else:
        det_count = extract_det_count_from_text(text)

    mean_ms = get_mean_ms(inference)

    raw_meta = {
        "command": cmd_str,
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }
    return det_count, mean_ms, config, inference, raw_meta


def run_full_bench(cmd: list[str], threads_line: str, out_dir: str, img_path: str) -> None:
    det_count, mean_ms, config, inference, raw_meta = probe_config(cmd)

    if raw_meta.get("returncode", 0) != 0:
        print(f"[bench_to_json][WARN] returncode={raw_meta['returncode']} for: {raw_meta['command']}")
        return

    saved_path = save_config_json(
        config=config,
        inference=inference,
        image_path=img_path,
        det_count=det_count,
        threads_line=threads_line,
        raw_meta=raw_meta,
        out_root=out_dir,
    )
    print(f"[bench_to_json] Saved: {saved_path} (mean_ms={mean_ms}, dets={det_count})")


def _iter_tiles(max_th: int) -> list[tuple[str, int, tuple[int, int]]]:
    # (tiles_str, omp_threads, tile_dims(W,H)) using assumed source dims 1920x1080
    out: list[tuple[str, int, tuple[int, int]]] = []
    candidates = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    for w, h in product(candidates, repeat=2):
        omp = w * h
        if omp > max_th:
            continue
        tile_dims = (1920 // w, 1080 // h)
        out.append((f"{w}x{h}", omp, tile_dims))
    return out


def _build_common_args(app_path: str, model_path: str, image_path: str) -> list[str]:
    return [
        app_path,
        "--model",
        model_path,
        "--image",
        image_path,
        "--bin_thresh",
        "0.3",
        "--box_thresh",
        "0.3",
        "--bind_io",
        "1",
        "--tile_overlap",
        "0.01",
        "--threads_inter",
        "1",
        "--verbose",
        "0",
        "--no_draw",
    ]


def _check_paths(app_path: str, model_paths: Iterable[str], image_paths: Iterable[str]) -> None:
    app = Path(app_path)
    if not app.exists():
        raise FileNotFoundError(f"App not found: {app}")
    if not app.is_file():
        raise ValueError(f"App path is not a file: {app}")

    for p in model_paths:
        mp = Path(p)
        if not mp.exists():
            raise FileNotFoundError(f"Model not found: {mp}")

    for p in image_paths:
        ip = Path(p)
        if not ip.exists():
            raise FileNotFoundError(f"Image not found: {ip}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run BENCH app, parse BENCH output and save JSON configs.\n"
            "Uses a baseline det-count and skips bad side/tiles combos."
        )
    )
    parser.add_argument("--root", default="bench_out", help="Root directory for saving configs (default: bench_out)")
    parser.add_argument(
        "--max-time",
        type=float,
        default=10.0,
        help="Max allowed mean time (ms) for side/tiles probe; above this skip all its configs",
    )

    parser.add_argument("--app", required=True, help="Path to text_det executable")
    parser.add_argument("--models", nargs="+", required=True, help="One or more ONNX model paths")
    parser.add_argument("--images", nargs="+", required=True, help="One or more image paths")

    parser.add_argument("--max-th", type=int, default=96, help="Thread budget cap (default: 96)")
    parser.add_argument("--side-min", type=int, default=64, help="Min side (default: 64)")
    parser.add_argument("--side-max", type=int, default=960, help="Max side (default: 960)")
    parser.add_argument("--side-step", type=int, default=32, help="Side step (default: 32)")

    parser.add_argument("--bench-iters", type=int, default=100, help="--bench value for full bench (default: 100)")
    parser.add_argument("--warmup", type=int, default=20, help="--warmup for full bench (default: 20)")
    parser.add_argument("--probe-warmup", type=int, default=1, help="--warmup for probe (default: 1)")

    args = parser.parse_args(argv)

    _check_paths(args.app, args.models, args.images)

    max_th: int = args.max_th
    sides = list(range(args.side_min, args.side_max + 1, args.side_step))
    tiles_omp = _iter_tiles(max_th)

    for model_path in args.models:
        for image_path in args.images:
            output_dir = Path(args.root) / Path(model_path).stem

            # ==== 1) BASELINE RUN ====
            baseline_side = max(sides)
            baseline_cmd = _build_common_args(args.app, model_path, image_path) + [
                "--side",
                str(baseline_side),
                "--tiles",
                "2x2",
                "--tile_omp",
                "4",
                "--threads_intra",
                "1",
                "--bench",
                "1",
                "--warmup",
                "10",
            ]

            print("[baseline] Probing baseline config...")
            baseline_dets, baseline_mean, _, _, base_raw = probe_config(baseline_cmd)
            if base_raw.get("returncode", 0) != 0:
                print(f"[baseline] ERROR: returncode={base_raw['returncode']}. aborting this (model,image).")
                continue

            if baseline_dets is None or baseline_dets <= 0:
                print(f"[baseline] ERROR: baseline det-count is {baseline_dets}, aborting this (model,image).")
                continue

            best_dets = baseline_dets
            print(f"[baseline] baseline_dets = {baseline_dets}, mean_ms = {baseline_mean}")

            # ==== 2) MAIN GRID SEARCH ====
            for side in sides:
                for tiles_str, omp, tile_dims in tiles_omp:
                    mx_intra_th = max(1, max_th // omp)

                    # Basic geometry sanity filter (kept from your logic)
                    mtd = max(tile_dims)
                    if mtd < side or side < mtd * 0.5:
                        continue

                    # ---- 2.1 PROBE THIS side / tiles COMBO ----
                    probe_cmd = _build_common_args(args.app, model_path, image_path) + [
                        "--side",
                        str(side),
                        "--threads_intra",
                        "1",
                        "--tile_omp",
                        str(omp),
                        "--tiles",
                        tiles_str,
                        "--bench",
                        "1",
                        "--warmup",
                        str(args.probe_warmup),
                    ]

                    print(f"[probe] side={side}, tiles={tiles_str}, omp={omp}: probing...")
                    dets_probe, mean_probe, _, _, raw_probe = probe_config(probe_cmd)

                    if raw_probe.get("returncode", 0) != 0:
                        print(f"[skip] returncode={raw_probe['returncode']} -> skip this side/tiles")
                        continue

                    if dets_probe is None:
                        print("[skip] dets=None -> skip this side/tiles")
                        continue

                    # Update best_dets if we see better detections.
                    if dets_probe > best_dets:
                        best_dets = dets_probe

                    if dets_probe < best_dets * 0.5:
                        print(f"[skip] dets={dets_probe} < best_dets={best_dets} * 0.5 -> skip this side/tiles")
                        continue

                    if mean_probe is None or mean_probe > args.max_time:
                        print(f"[skip] mean={mean_probe}ms > {args.max_time} -> skip this side/tiles")
                        continue

                    # ---- 2.2 FULL BENCH FOR THIS side / tiles COMBO ----
                    for intra_th in range(1, mx_intra_th + 1):
                        cmd = _build_common_args(args.app, model_path, image_path) + [
                            "--side",
                            str(side),
                            "--threads_intra",
                            str(intra_th),
                            "--tile_omp",
                            str(omp),
                            "--tiles",
                            tiles_str,
                            "--bench",
                            str(args.bench_iters),
                            "--warmup",
                            str(args.warmup),
                        ]

                        suff = f"{tiles_str}_{omp}_{intra_th}_side{side}"
                        print(f"[bench] {suff}")
                        run_full_bench(cmd, suff, str(output_dir), image_path)


if __name__ == "__main__":
    main()
