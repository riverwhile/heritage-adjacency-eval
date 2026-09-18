#!/usr/bin/env python3
"""Reproduce the four-object Nerfacto alternative-position summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "derived" / "nerfacto_v2v3_metrics_rows.csv"
OBJECTS = ("bronze_cup", "bronze_statue", "silver_comb", "stone_horse")
CONDITIONS = ("IN", "OUT")
PROTOCOLS = ("blocked_v2", "blocked_v3")
METRICS = ("psnr", "ssim", "lpips")
N_BOOT = 20_000
SEED = 20260815


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.input)
    expected = {(obj, condition, protocol) for obj in OBJECTS for condition in CONDITIONS for protocol in PROTOCOLS}
    actual = {(row["object"], row["condition"], row["protocol"]) for row in rows}
    if actual != expected or len(rows) != 16:
        raise ValueError(f"Nerfacto v2/v3 matrix mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    index = {(row["object"], row["condition"], row["protocol"]): row for row in rows}

    object_deltas = []
    summary = []
    rng = np.random.default_rng(SEED)
    for protocol in PROTOCOLS:
        for metric in METRICS:
            values = []
            for obj in OBJECTS:
                delta = float(index[(obj, "OUT", protocol)][metric]) - float(index[(obj, "IN", protocol)][metric])
                values.append(delta)
                object_deltas.append({"object": obj, "protocol": protocol, "metric": metric.upper(), "out_minus_in": delta})
            array = np.asarray(values, dtype=float)
            draws = rng.integers(0, len(array), size=(N_BOOT, len(array)))
            low, high = np.percentile(array[draws].mean(axis=1), [2.5, 97.5])
            better = sum(value < 0 for value in values) if metric == "lpips" else sum(value > 0 for value in values)
            summary.append(
                {
                    "method": "Nerfacto",
                    "protocol": protocol,
                    "metric": metric.upper(),
                    "n_objects": len(values),
                    "mean_out_minus_in": array.mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "out_better_objects": better,
                    "bootstrap_resamples": N_BOOT,
                    "seed": SEED,
                }
            )

    write_csv(args.output_dir / "nerfacto_v2v3_object_deltas.csv", object_deltas)
    write_csv(args.output_dir / "nerfacto_v2v3_summary.csv", summary)
    print(f"validated {len(rows)} metric rows and wrote {len(summary)} summary rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
