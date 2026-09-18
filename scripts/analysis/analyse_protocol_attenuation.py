#!/usr/bin/env python3
"""Reproduce paired protocol attenuation and small-sample sensitivity results."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "derived"
N_BOOT = 20_000
SEED = 20260815
METRICS = ("PSNR", "SSIM", "LPIPS")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_rows(data_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(data_dir / "blocked_full_metrics_rows.csv"):
        if row["kind"] == "baseline" and row["method"] == "ours_7000" and row["protocol"] in {"al", "blocked_v1"}:
            rows.append({**row, "method_family": "3DGS"})
    for row in read_csv(data_dir / "nerfacto_10k_metrics_rows.csv"):
        if row["protocol"] in {"al", "blocked_v1"}:
            rows.append(
                {
                    **row,
                    "method_family": "Nerfacto",
                    "PSNR": row["psnr"],
                    "SSIM": row["ssim"],
                    "LPIPS": row["lpips"],
                }
            )
    return rows


def paired_effects(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    index = {(r["method_family"], r["object"], r["condition"], r["protocol"]): r for r in rows}
    effects: list[dict[str, object]] = []
    for method in sorted({r["method_family"] for r in rows}):
        objects = sorted({r["object"] for r in rows if r["method_family"] == method})
        if len(objects) != 7:
            raise ValueError(f"{method}: expected seven objects, found {len(objects)}")
        for obj in objects:
            for metric in METRICS:
                deltas = {}
                for protocol in ("al", "blocked_v1"):
                    inside = index[(method, obj, "IN", protocol)]
                    outside = index[(method, obj, "OUT", protocol)]
                    deltas[protocol] = float(outside[metric]) - float(inside[metric])
                effects.append(
                    {
                        "method": method,
                        "object": obj,
                        "metric": metric,
                        "interleaved_effect": deltas["al"],
                        "blocked_effect": deltas["blocked_v1"],
                        "attenuation": deltas["al"] - deltas["blocked_v1"],
                    }
                )
    return effects


def summarise(effects: list[dict[str, object]]):
    rng = np.random.default_rng(SEED)
    summary: list[dict[str, object]] = []
    leave_one_out: list[dict[str, object]] = []
    sign_flip: list[dict[str, object]] = []
    for method in sorted({str(r["method"]) for r in effects}):
        for metric in METRICS:
            group = [r for r in effects if r["method"] == method and r["metric"] == metric]
            attenuation = np.asarray([r["attenuation"] for r in group], dtype=float)
            interleaved = np.asarray([r["interleaved_effect"] for r in group], dtype=float)
            blocked = np.asarray([r["blocked_effect"] for r in group], dtype=float)
            draws = rng.integers(0, len(attenuation), size=(N_BOOT, len(attenuation)))
            bootstrap = attenuation[draws].mean(axis=1)
            low, high = np.percentile(bootstrap, [2.5, 97.5])
            summary.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_objects": len(group),
                    "interleaved_effect": interleaved.mean(),
                    "blocked_effect": blocked.mean(),
                    "protocol_induced_attenuation": attenuation.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_resamples": N_BOOT,
                    "seed": SEED,
                }
            )
            observed = abs(float(attenuation.sum()))
            assignments = list(itertools.product((-1, 1), repeat=len(attenuation)))
            extreme = sum(
                abs(sum(sign * value for sign, value in zip(signs, attenuation))) >= observed - 1e-12
                for signs in assignments
            )
            sign_flip.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_objects": len(group),
                    "mean_attenuation": attenuation.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "extreme_assignments": extreme,
                    "total_assignments": len(assignments),
                    "two_sided_p": extreme / len(assignments),
                }
            )
            for index, row in enumerate(group):
                keep = np.arange(len(attenuation)) != index
                leave_one_out.append(
                    {
                        "method": method,
                        "metric": metric,
                        "omitted_object": row["object"],
                        "n_objects": int(keep.sum()),
                        "interleaved_effect": interleaved[keep].mean(),
                        "blocked_effect": blocked[keep].mean(),
                        "protocol_induced_attenuation": attenuation[keep].mean(),
                    }
                )
    return summary, leave_one_out, sign_flip


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    effects = paired_effects(load_rows(args.data_dir))
    summary, leave_one_out, sign_flip = summarise(effects)
    write_csv(args.output_dir / "protocol_attenuation_object_rows.csv", effects)
    write_csv(args.output_dir / "protocol_attenuation_summary.csv", summary)
    write_csv(args.output_dir / "protocol_attenuation_leave_one_out.csv", leave_one_out)
    write_csv(args.output_dir / "sign_flip_sensitivity.csv", sign_flip)
    print(f"wrote {len(effects)} object-metric effects and {len(sign_flip)} sign-flip rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
