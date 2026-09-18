#!/usr/bin/env python3
"""Render the public blocked-protocol summary and comparison figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "derived" / "blocked_protocol_master_table.csv"

def read_rows(path: Path) -> list[dict[str, object]]:
    numeric = {
        "n_objects",
        "temporal_gap_frames",
        "center_distance_norm",
        "forward_angle_deg",
        "max_train_ssim",
        "psnr_delta",
        "psnr_ci_low",
        "psnr_ci_high",
        "ssim_delta",
        "ssim_ci_low",
        "ssim_ci_high",
        "lpips_delta",
        "lpips_ci_low",
        "lpips_ci_high",
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    expected = ["al", "blocked_v1", "blocked_v2", "blocked_v3"]
    if [row["protocol_key"] for row in rows] != expected:
        raise ValueError("unexpected protocol order or missing protocol")
    return rows


def copy_summary(rows: list[dict[str, object]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input)
    copy_summary(rows, args.output_dir / "blocked_protocol_master_table.csv")

    x = np.arange(len(rows))
    labels = ["Inter.\nleaved", "Block\nv1", "Block\nv2", "Block\nv3"]
    colours = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    for axis, metric, title in zip(axes[:2], ("psnr", "lpips"), ("OUT - IN PSNR", "OUT - IN LPIPS")):
        values = np.asarray([row[f"{metric}_delta"] for row in rows], dtype=float)
        low = np.asarray([row[f"{metric}_ci_low"] for row in rows], dtype=float)
        high = np.asarray([row[f"{metric}_ci_high"] for row in rows], dtype=float)
        axis.bar(x, values, color=colours, width=0.68)
        axis.errorbar(x, values, yerr=np.vstack([values - low, high - values]), fmt="none", ecolor="black", capsize=3, lw=1)
        axis.axhline(0, color="black", lw=0.8)
        axis.set_title(title)
        axis.set_xticks(x, labels)
        axis.set_ylabel("dB" if metric == "psnr" else "LPIPS")

    base = rows[0]
    separation = np.asarray(
        [
            [
                row["temporal_gap_frames"] / base["temporal_gap_frames"],
                row["center_distance_norm"] / base["center_distance_norm"],
                row["forward_angle_deg"] / base["forward_angle_deg"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    for index, name in enumerate(("Temporal gap", "Camera-centre gap", "Angular gap")):
        axes[2].plot(x, separation[:, index], marker="o", lw=1.8, label=name)
    axes[2].axhline(1, color="black", lw=0.8)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Ratio to interleaved")
    axes[2].set_title("Train-test separation")
    axes[2].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output_dir / "blocked_protocol_leakage_and_metrics.png", dpi=300, bbox_inches="tight")
    figure.savefig(args.output_dir / "blocked_protocol_leakage_and_metrics.pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"validated {len(rows)} protocols and wrote summary figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
