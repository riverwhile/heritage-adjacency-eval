#!/usr/bin/env python3
"""Build manuscript-ready protocol tables, statistics, figure, and result notes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper" / "sci_assets" / "tables"
FIGURES = ROOT / "paper" / "sci_assets" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

metric_data = json.loads((TABLES / "blocked_full_metrics_summary.json").read_text(encoding="utf-8"))
leak_data = json.loads((TABLES / "split_leakage_audit.json").read_text(encoding="utf-8"))

protocols = ["al", "blocked_v1", "blocked_v2", "blocked_v3"]
labels = {"al": "Interleaved", "blocked_v1": "Blocked v1",
          "blocked_v2": "Blocked v2", "blocked_v3": "Blocked v3"}
metrics = ["PSNR", "SSIM", "LPIPS"]
rng = np.random.default_rng(20260814)


def ci(values, n=20000):
    a = np.asarray(values, dtype=float)
    boot = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    return [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]


rows = []
for p in protocols:
    comp = metric_data["comparisons"][f"baseline_{p}_OUT_vs_IN"]
    lr = leak_data["summary"][p]
    row = {
        "protocol": labels[p], "protocol_key": p, "n_objects": comp["summary"]["n"],
        "temporal_gap_frames": lr["temporal_gap_frames"]["mean"],
        "center_distance_norm": lr["temporal_neighbor_center_distance_norm"]["mean"],
        "forward_angle_deg": lr["temporal_neighbor_angle_deg"]["mean"],
        "max_train_ssim": lr["max_train_ssim"]["mean"],
    }
    for m in metrics:
        key = f"delta_OUT_minus_IN_{m}"
        vals = [x[key] for x in comp["rows"]]
        low, high = ci(vals)
        row[f"{m.lower()}_delta"] = float(np.mean(vals))
        row[f"{m.lower()}_ci_low"] = low
        row[f"{m.lower()}_ci_high"] = high
        row[f"{m.lower()}_out_better"] = sum(v > 0 for v in vals) if m != "LPIPS" else sum(v < 0 for v in vals)
    rows.append(row)

csv_path = TABLES / "blocked_protocol_master_table.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

md = [
    "# Leakage-resistant protocol comparison",
    "",
    "Positive PSNR/SSIM and negative LPIPS deltas favour OUT. Confidence intervals are object-level percentile bootstrap intervals (20,000 resamples).",
    "",
    "| Protocol | n | Temporal gap | Center distance | Angle | Max train SSIM | ΔPSNR [95% CI] | ΔSSIM [95% CI] | ΔLPIPS [95% CI] |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    md.append(
        f"| {r['protocol']} | {r['n_objects']} | {r['temporal_gap_frames']:.2f} | "
        f"{r['center_distance_norm']:.4f} | {r['forward_angle_deg']:.2f}° | {r['max_train_ssim']:.3f} | "
        f"{r['psnr_delta']:+.3f} [{r['psnr_ci_low']:+.3f}, {r['psnr_ci_high']:+.3f}] | "
        f"{r['ssim_delta']:+.4f} [{r['ssim_ci_low']:+.4f}, {r['ssim_ci_high']:+.4f}] | "
        f"{r['lpips_delta']:+.4f} [{r['lpips_ci_low']:+.4f}, {r['lpips_ci_high']:+.4f}] |"
    )
(TABLES / "blocked_protocol_master_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
x = np.arange(len(rows))
colors = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"]
for ax, m, title in zip(axes[:2], ["PSNR", "LPIPS"], ["OUT − IN PSNR", "OUT − IN LPIPS"]):
    vals = np.array([r[f"{m.lower()}_delta"] for r in rows])
    lo = np.array([r[f"{m.lower()}_ci_low"] for r in rows])
    hi = np.array([r[f"{m.lower()}_ci_high"] for r in rows])
    ax.bar(x, vals, color=colors, width=.68)
    ax.errorbar(x, vals, yerr=np.vstack([vals-lo, hi-vals]), fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axhline(0, color="black", lw=.8)
    ax.set_title(title); ax.set_xticks(x, ["Inter.\nleaved", "Block\nv1", "Block\nv2", "Block\nv3"])
    ax.set_ylabel("dB" if m == "PSNR" else "LPIPS")

ax = axes[2]
base = rows[0]
separation = np.array([[r["temporal_gap_frames"]/base["temporal_gap_frames"],
                        r["center_distance_norm"]/base["center_distance_norm"],
                        r["forward_angle_deg"]/base["forward_angle_deg"]] for r in rows])
for i, name in enumerate(["Temporal gap", "Camera-center gap", "Angular gap"]):
    ax.plot(x, separation[:, i], marker="o", lw=1.8, label=name)
ax.axhline(1, color="black", lw=.8)
ax.set_xticks(x, ["Inter.\nleaved", "Block\nv1", "Block\nv2", "Block\nv3"])
ax.set_ylabel("Ratio to interleaved"); ax.set_title("Train–test separation")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "blocked_protocol_leakage_and_metrics.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "blocked_protocol_leakage_and_metrics.pdf", bbox_inches="tight")
plt.close(fig)

# Per-object PSNR conclusion-reversal heatmap. Missing v2/v3 cells are intentionally blank.
objects = sorted({x["object"] for x in metric_data["comparisons"]["baseline_al_OUT_vs_IN"]["rows"]})
heat = np.full((len(objects), len(protocols)), np.nan)
for j, p in enumerate(protocols):
    for item in metric_data["comparisons"][f"baseline_{p}_OUT_vs_IN"]["rows"]:
        heat[objects.index(item["object"]), j] = item["delta_OUT_minus_IN_PSNR"]
fig, ax = plt.subplots(figsize=(7.0, 4.8))
masked = np.ma.masked_invalid(heat)
im = ax.imshow(masked, cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
for i in range(len(objects)):
    for j in range(len(protocols)):
        if np.isfinite(heat[i, j]):
            ax.text(j, i, f"{heat[i,j]:+.2f}", ha="center", va="center",
                    color="white" if abs(heat[i,j]) > 2.7 else "black", fontsize=9)
        else:
            ax.text(j, i, "—", ha="center", va="center", color="#777777")
ax.set_xticks(range(len(protocols)), [labels[p] for p in protocols])
ax.set_yticks(range(len(objects)), [x.replace("_", " ") for x in objects])
ax.set_title("Per-object OUT − IN PSNR (dB)")
cbar = fig.colorbar(im, ax=ax, shrink=.85); cbar.set_label("PSNR difference (dB)")
fig.tight_layout()
fig.savefig(FIGURES / "blocked_protocol_object_psnr_reversal.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURES / "blocked_protocol_object_psnr_reversal.pdf", bbox_inches="tight")
plt.close(fig)

v1 = rows[1]
notes = f"""# Manuscript-ready result notes: split leakage and conclusion reversal

The conventional every-eighth-frame split placed held-out views only {base['temporal_gap_frames']:.2f} frames from the nearest training view on average. Its normalized camera-center separation was {base['center_distance_norm']:.4f}, its angular separation was {base['forward_angle_deg']:.2f} degrees, and the maximum training-image SSIM averaged {base['max_train_ssim']:.3f}. Under this protocol, OUT outperformed IN on all seven objects, with a mean PSNR difference of {base['psnr_delta']:+.3f} dB (95% bootstrap CI {base['psnr_ci_low']:+.3f} to {base['psnr_ci_high']:+.3f}).

The predeclared contiguous blocked-v1 protocol increased the nearest temporal gap to {v1['temporal_gap_frames']:.2f} frames, normalized camera-center separation to {v1['center_distance_norm']:.4f}, and angular separation to {v1['forward_angle_deg']:.2f} degrees, while maximum training-image SSIM fell to {v1['max_train_ssim']:.3f}. The PSNR difference contracted to {v1['psnr_delta']:+.3f} dB (95% CI {v1['psnr_ci_low']:+.3f} to {v1['psnr_ci_high']:+.3f}) and OUT was better on only {v1['psnr_out_better']}/7 objects. Two additional block placements produced PSNR differences of {rows[2]['psnr_delta']:+.3f} and {rows[3]['psnr_delta']:+.3f} dB, with confidence intervals spanning zero. Thus, the unanimous interleaved result does not survive leakage-resistant evaluation.

LPIPS remained directionally more consistent than PSNR under blocked v1, but its object-level confidence interval and the inconsistent SSIM/PSNR outcomes do not support a universal across-metric improvement claim. The defensible finding is protocol sensitivity: adjacent-frame interpolation can transform a heterogeneous, split-dependent result into an apparently unanimous conclusion.

## Claim boundary

- Do not describe the blocked protocol as fully leakage-free; call it leakage-resistant or adjacency-controlled.
- Do not infer that glass reflection alone causes the IN/OUT difference without fixed-pose glass-on/off reference capture.
- Do not claim the veiling branch universally improves reconstruction; report small mean changes and object-level failures.
- Treat blocked v2/v3 as split-sensitivity evidence because they cover four objects, not the full seven-object cohort.
"""
(TABLES / "blocked_protocol_results_notes.md").write_text(notes, encoding="utf-8")
(TABLES / "blocked_protocol_remaining_evidence.md").write_text("""# Remaining evidence decision after blocked-protocol audit

## No-GPU work that should be completed first

1. Integrate the master table and two protocol figures into the manuscript.
2. Rewrite the abstract, methods, results, discussion, and limitations around leakage-sensitive evaluation rather than universal de-reflection gains.
3. Add exact split-list and bootstrap procedures to the supplement/reproducibility material.
4. Select representative render triplets for one stable positive object, one sign-reversal object, and one stable negative object.

## GPU experiments that are not immediately required

- More iterations or additional veiling hyperparameter sweeps: current limitation is protocol validity, not convergence.
- Repeating v2/v3 on all seven objects: useful strengthening, but the present four-object subset already demonstrates block-placement sensitivity and should be described as a sensitivity subset.
- Additional seed sweeps on every object: disproportionate cost given the small and inconsistent method effect.

## Only high-value optional GPU extension

If reviewers or the final manuscript structure require a complete split-sensitivity matrix, run blocked v2/v3 for the three currently missing objects (`cloth_hat`, `cloth_shoe`, `porcelain_plate`), IN and OUT, for 12 additional 7k models. Predeclare this as completion of the matrix, not model selection. Do this only after the manuscript tables expose a clear need.

## Non-GPU evidence that remains scientifically more important

A fixed-pose or tightly pose-matched glass-on/glass-off reference capture is required for a causal claim about glass reflection or successful de-reflection. No amount of rerunning the present videos can replace that reference evidence. Without new capture, constrain the paper to evaluation-protocol sensitivity and reconstruction robustness in cultural-heritage video.
""", encoding="utf-8")
print(json.dumps({"rows": rows, "outputs": [str(csv_path), str(FIGURES / 'blocked_protocol_leakage_and_metrics.png')]}, indent=2))
