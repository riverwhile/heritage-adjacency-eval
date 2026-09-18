#!/usr/bin/env python3
"""Reproduce the training-count-matched 3DGS control from public rows."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "derived"
OBJECTS = ("bronze_cup", "bronze_statue", "cloth_hat", "cloth_shoe", "porcelain_plate", "silver_comb", "stone_horse")
METRICS = ("PSNR", "SSIM", "LPIPS")
SEEDS = (2024, 2025, 2026)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    return sum(values) / len(values)


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap(values, seed, samples=20_000):
    rng = random.Random(seed)
    draws = [mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples)]
    return mean(values), percentile(draws, 0.025), percentile(draws, 0.975)


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

    seed_rows = read_csv(args.data_dir / "count_matched_seed_scores.csv")
    expected = {(obj, condition, seed) for obj in OBJECTS for condition in ("IN", "OUT") for seed in SEEDS}
    actual = {(row["object"], row["condition"], int(row["seed"])) for row in seed_rows}
    if actual != expected:
        raise ValueError(f"count-matched matrix mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")

    grouped = defaultdict(lambda: defaultdict(list))
    for row in seed_rows:
        for metric in METRICS:
            grouped[(row["object"], row["condition"])][metric].append(float(row[metric]))
    condition_rows = []
    for (obj, condition), values in sorted(grouped.items()):
        condition_rows.append(
            {
                "object": obj,
                "condition": condition,
                **{metric: mean(values[metric]) for metric in METRICS},
                **{f"{metric}_seed_range": max(values[metric]) - min(values[metric]) for metric in METRICS},
            }
        )

    baseline = {}
    for row in read_csv(args.data_dir / "blocked_full_metrics_rows.csv"):
        if row["kind"] == "baseline" and row["method"] == "ours_7000" and row["protocol"] in {"al", "blocked_v1"}:
            baseline[(row["object"], row["condition"], row["protocol"])] = {metric: float(row[metric]) for metric in METRICS}
    count_matched = {(row["object"], row["condition"]): row for row in condition_rows}

    object_rows = []
    for obj in OBJECTS:
        for metric in METRICS:
            interleaved = baseline[(obj, "OUT", "al")][metric] - baseline[(obj, "IN", "al")][metric]
            matched = count_matched[(obj, "OUT")][metric] - count_matched[(obj, "IN")][metric]
            blocked = baseline[(obj, "OUT", "blocked_v1")][metric] - baseline[(obj, "IN", "blocked_v1")][metric]
            object_rows.append(
                {
                    "object": obj,
                    "metric": metric,
                    "interleaved_effect": interleaved,
                    "count_matched_effect": matched,
                    "blocked_v1_effect": blocked,
                    "training_count_effect_al_minus_cm": interleaved - matched,
                    "adjacency_effect_cm_minus_blocked": matched - blocked,
                    "total_attenuation_al_minus_blocked": interleaved - blocked,
                }
            )

    summary = []
    quantities = (
        ("interleaved_effect", "interleaved"),
        ("count_matched_effect", "interleaved_count_matched"),
        ("blocked_v1_effect", "blocked_v1"),
        ("training_count_effect_al_minus_cm", "training_count_component_al_minus_count_matched"),
        ("adjacency_effect_cm_minus_blocked", "adjacency_component_count_matched_minus_blocked"),
        ("total_attenuation_al_minus_blocked", "total_al_minus_blocked"),
    )
    for metric in METRICS:
        rows = [row for row in object_rows if row["metric"] == metric]
        for key, label in quantities:
            estimate, low, high = bootstrap([row[key] for row in rows], 20260818 + len(summary))
            summary.append({"metric": metric, "quantity": label, "mean": estimate, "ci_low": low, "ci_high": high})

    write_csv(args.output_dir / "count_matched_condition_means.csv", condition_rows)
    write_csv(args.output_dir / "count_matched_object_effects.csv", object_rows)
    write_csv(args.output_dir / "count_matched_summary.csv", summary)
    print(f"validated {len(seed_rows)} deletion-seed rows and wrote {len(summary)} summary rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
