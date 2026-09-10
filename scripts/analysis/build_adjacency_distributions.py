"""Build descriptive train--test adjacency distribution assets.

The inferential unit in the manuscript remains the object.  These outputs
describe held-out-view geometry and are not used for frame-level inference.
"""

from pathlib import Path
import csv
import statistics

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper" / "sci_assets" / "tables"
FIGURES = ROOT / "paper" / "sci_assets" / "figures"
SOURCE = TABLES / "split_leakage_audit_rows_with_cm.csv"

PROTOCOLS = ["al", "interleaved_cm", "blocked_v1", "blocked_v2", "blocked_v3"]
LABELS = ["Interleaved", "Interleaved-CM", "Blocked-v1", "Blocked-v2", "Blocked-v3"]
METRICS = [
    ("temporal_gap_frames", "Nearest temporal gap (frames)"),
    ("temporal_neighbor_center_distance_norm", "Camera-centre distance (normalised)"),
    ("temporal_neighbor_angle_deg", "Viewing-direction angle (degrees)"),
    ("max_train_ssim", "Maximum training-image SSIM"),
]


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
    raw_rows = list(csv.DictReader(handle))

for row in raw_rows:
    for key, _ in METRICS:
        row[key] = float(row[key])

# CM uses the same held-out views with three deterministic training-removal
# seeds. Average seed-specific adjacency for each held-out view so that the
# descriptive sample count remains the number of distinct test views rather
# than treating seeds as independent observations.
rows = [r for r in raw_rows if not r["protocol"].startswith("interleaved_count_matched_seed")]
cm_raw = [r for r in raw_rows if r["protocol"].startswith("interleaved_count_matched_seed")]
cm_keys = sorted({(r["object"], r["condition"], r["test_name"]) for r in cm_raw})
for obj, condition, test_name in cm_keys:
    group = [r for r in cm_raw if (r["object"], r["condition"], r["test_name"])
             == (obj, condition, test_name)]
    if len(group) != 3:
        raise ValueError(f"Expected three CM seeds for {(obj, condition, test_name)}, got {len(group)}")
    row = dict(group[0])
    row["protocol"] = "interleaved_cm"
    for key, _ in METRICS:
        row[key] = statistics.mean(r[key] for r in group)
    rows.append(row)

# Overall descriptive summary: median, IQR, and full observed range.
summary_rows = []
for protocol, label in zip(PROTOCOLS, LABELS):
    subset = [r for r in rows if r["protocol"] == protocol]
    for key, metric_label in METRICS:
        values = [r[key] for r in subset]
        summary_rows.append({
            "protocol": protocol,
            "protocol_label": label,
            "metric": key,
            "metric_label": metric_label,
            "n_test_views": len(values),
            "median": statistics.median(values),
            "q1": percentile(values, 25),
            "q3": percentile(values, 75),
            "minimum": min(values),
            "maximum": max(values),
        })

with (TABLES / "adjacency_distribution_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

# Per-object medians combine IN and OUT test views, while retaining protocol.
object_rows = []
objects = sorted({r["object"] for r in rows})
for obj in objects:
    for protocol, label in zip(PROTOCOLS, LABELS):
        subset = [r for r in rows if r["object"] == obj and r["protocol"] == protocol]
        out = {
            "object": obj,
            "protocol": protocol,
            "protocol_label": label,
            "n_test_views": len(subset),
        }
        for key, _ in METRICS:
            out[f"median_{key}"] = statistics.median(r[key] for r in subset)
        object_rows.append(out)

with (TABLES / "adjacency_per_object_medians.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=object_rows[0].keys())
    writer.writeheader()
    writer.writerows(object_rows)


def fmt(metric, value):
    if metric == "temporal_gap_frames":
        return f"{value:.1f}"
    if metric == "temporal_neighbor_center_distance_norm":
        return f"{value:.4f}"
    if metric == "temporal_neighbor_angle_deg":
        return f"{value:.2f}"
    return f"{value:.3f}"


with (TABLES / "adjacency_distribution_summary.tex").open("w", encoding="ascii", newline="\n") as handle:
    for row in summary_rows:
        metric_short = {
            "temporal_gap_frames": "Temporal gap",
            "temporal_neighbor_center_distance_norm": "Centre distance",
            "temporal_neighbor_angle_deg": "Viewing angle",
            "max_train_ssim": "Max. train SSIM",
        }[row["metric"]]
        med = fmt(row["metric"], row["median"])
        q1 = fmt(row["metric"], row["q1"])
        q3 = fmt(row["metric"], row["q3"])
        lo = fmt(row["metric"], row["minimum"])
        hi = fmt(row["metric"], row["maximum"])
        handle.write(
            f'{row["protocol_label"]} & {metric_short} & {row["n_test_views"]} & '
            f'{med} & [{q1}, {q3}] & [{lo}, {hi}] \\\\\n'
        )
    handle.write(r"\bottomrule" + "\n")

with (TABLES / "adjacency_per_object_medians.tex").open("w", encoding="ascii", newline="\n") as handle:
    for row in object_rows:
        obj = row["object"].replace("_", r"\_")
        handle.write(
            f'{obj} & {row["protocol_label"]} & {row["n_test_views"]} & '
            f'{row["median_temporal_gap_frames"]:.1f} & '
            f'{row["median_temporal_neighbor_center_distance_norm"]:.4f} & '
            f'{row["median_temporal_neighbor_angle_deg"]:.2f} & '
            f'{row["median_max_train_ssim"]:.3f} \\\\\n'
        )
    handle.write(r"\bottomrule" + "\n")

# Four-panel distribution plot. Jittered points expose the empirical support;
# violins and embedded boxes summarise it without implying frame-level tests.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
})
colors = ["#4C78A8", "#9C6ADE", "#E45756", "#72B7B2", "#F2CF5B"]
rng = np.random.default_rng(20260818)
fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.1), constrained_layout=True)

for panel, (ax, (key, ylabel)) in enumerate(zip(axes.flat, METRICS)):
    groups = [[r[key] for r in rows if r["protocol"] == p] for p in PROTOCOLS]
    positions = np.arange(1, len(PROTOCOLS) + 1)
    violins = ax.violinplot(groups, positions=positions, widths=0.82,
                            showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
    boxes = ax.boxplot(groups, positions=positions, widths=0.24,
                       patch_artist=True, showfliers=False,
                       medianprops={"color": "black", "linewidth": 1.2},
                       whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8})
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_linewidth(0.8)
    for pos, (values, color) in enumerate(zip(groups, colors), start=1):
        x = pos + rng.uniform(-0.13, 0.13, len(values))
        ax.scatter(x, values, s=4.5, alpha=0.18, color=color, linewidths=0, rasterized=True)
    ax.set_xticks(positions, LABELS, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(chr(65 + panel))
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

fig.savefig(FIGURES / "adjacency_distributions.pdf", bbox_inches="tight")
fig.savefig(FIGURES / "adjacency_distributions.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"rows={len(rows)} objects={len(objects)} protocols={len(PROTOCOLS)}")
print(TABLES / "adjacency_distribution_summary.csv")
print(TABLES / "adjacency_per_object_medians.csv")
print(FIGURES / "adjacency_distributions.pdf")
