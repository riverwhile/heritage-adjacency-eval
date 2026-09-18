#!/usr/bin/env python3
"""Aggregate the fixed 28-model Nerfacto 10k protocol matrix."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from statistics import mean


METRICS = ("psnr", "ssim", "lpips")
OBJECTS = (
    "bronze_cup",
    "bronze_statue",
    "cloth_hat",
    "cloth_shoe",
    "porcelain_plate",
    "silver_comb",
    "stone_horse",
)
PROTOCOLS = ("al", "blocked_v1")
CONDITIONS = ("IN", "OUT")
NAME_RE = re.compile(r"^(?P<object>.+)_(?P<condition>IN|OUT)_(?P<protocol>al|blocked_v1)$")


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty percentile input")
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_ci(values: list[float], seed: int, samples: int = 20000) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    boot = [mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    boot.sort()
    return [percentile(boot, 0.025), percentile(boot, 0.975)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    values: dict[tuple[str, str, str], dict[str, float]] = {}
    for path in sorted(args.input_dir.glob("*.json")):
        match = NAME_RE.match(path.stem)
        if not match:
            raise ValueError(f"unexpected result filename: {path.name}")
        key = (match["object"], match["condition"], match["protocol"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["results"]
        metric_values = {metric: float(result[metric]) for metric in METRICS}
        values[key] = metric_values
        rows.append(
            {
                "object": key[0],
                "condition": key[1],
                "protocol": key[2],
                **metric_values,
                "checkpoint_name": Path(payload.get("checkpoint", "")).name,
                "source_json": path.name,
            }
        )

    expected = {(obj, condition, protocol) for obj in OBJECTS for condition in CONDITIONS for protocol in PROTOCOLS}
    missing = sorted(expected - set(values))
    extra = sorted(set(values) - expected)
    if missing or extra or len(rows) != 28:
        raise RuntimeError(f"matrix mismatch: rows={len(rows)} missing={missing} extra={extra}")

    rows_path = args.output_dir / "nerfacto_10k_metrics_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "matrix_rows": len(rows),
        "objects": len(OBJECTS),
        "iterations": 10000,
        "seed": 2024,
        "camera_optimizer": "off",
        "protocols": {},
        "difference_of_differences_blocked_minus_al": {},
        "blocked_minus_al_absolute_difficulty": {},
        "object_deltas": [],
    }

    object_deltas = []
    for obj in OBJECTS:
        record: dict[str, object] = {"object": obj}
        for protocol in PROTOCOLS:
            for metric in METRICS:
                record[f"{protocol}_{metric}_out_minus_in"] = (
                    values[(obj, "OUT", protocol)][metric] - values[(obj, "IN", protocol)][metric]
                )
        object_deltas.append(record)
    summary["object_deltas"] = object_deltas

    for protocol_index, protocol in enumerate(PROTOCOLS):
        protocol_summary = {}
        for metric_index, metric in enumerate(METRICS):
            deltas = [float(row[f"{protocol}_{metric}_out_minus_in"]) for row in object_deltas]
            better = sum(delta < 0 for delta in deltas) if metric == "lpips" else sum(delta > 0 for delta in deltas)
            protocol_summary[metric] = {
                "mean_out_minus_in": mean(deltas),
                "bootstrap_95_ci": paired_bootstrap_ci(deltas, 2024 + 10 * protocol_index + metric_index),
                "out_better_objects": better,
                "total_objects": len(deltas),
            }
        summary["protocols"][protocol] = protocol_summary

    for metric_index, metric in enumerate(METRICS):
        did = [
            float(row[f"blocked_v1_{metric}_out_minus_in"]) - float(row[f"al_{metric}_out_minus_in"])
            for row in object_deltas
        ]
        summary["difference_of_differences_blocked_minus_al"][metric] = {
            "mean": mean(did),
            "bootstrap_95_ci": paired_bootstrap_ci(did, 2124 + metric_index),
        }
        difficulty = [
            values[(obj, condition, "blocked_v1")][metric] - values[(obj, condition, "al")][metric]
            for obj in OBJECTS
            for condition in CONDITIONS
        ]
        summary["blocked_minus_al_absolute_difficulty"][metric] = {
            "mean": mean(difficulty),
            "bootstrap_95_ci": paired_bootstrap_ci(difficulty, 2224 + metric_index),
        }

    summary_path = args.output_dir / "nerfacto_10k_protocol_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Nerfacto 10k protocol summary",
        "",
        "Raw deltas are OUT minus IN; lower LPIPS is better.",
        "",
        "| Protocol | PSNR delta (95% CI), wins | SSIM delta (95% CI), wins | LPIPS delta (95% CI), wins |",
        "|---|---:|---:|---:|",
    ]
    for protocol in PROTOCOLS:
        cells = []
        for metric in METRICS:
            item = summary["protocols"][protocol][metric]
            lo, hi = item["bootstrap_95_ci"]
            cells.append(
                f"{item['mean_out_minus_in']:+.6f} [{lo:+.6f}, {hi:+.6f}], {item['out_better_objects']}/7"
            )
        lines.append(f"| {protocol} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Object-level OUT-IN deltas",
            "",
            "| Object | al PSNR | blocked PSNR | al SSIM | blocked SSIM | al LPIPS | blocked LPIPS |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in object_deltas:
        lines.append(
            "| {object} | {al_psnr_out_minus_in:+.4f} | {blocked_v1_psnr_out_minus_in:+.4f} | "
            "{al_ssim_out_minus_in:+.4f} | {blocked_v1_ssim_out_minus_in:+.4f} | "
            "{al_lpips_out_minus_in:+.4f} | {blocked_v1_lpips_out_minus_in:+.4f} |".format(**row)
        )
    (args.output_dir / "nerfacto_10k_protocol_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
