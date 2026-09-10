#!/usr/bin/env python3
"""Validate and aggregate the 16-model Nerfacto blocked-v2/v3 subset."""

from pathlib import Path
import csv
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "tmp" / "nerfacto_v2v3_results"
OUT = ROOT / "paper" / "sci_assets" / "tables"
OBJECTS = ("bronze_cup", "bronze_statue", "silver_comb", "stone_horse")
CONDITIONS = ("IN", "OUT")
PROTOCOLS = ("blocked_v2", "blocked_v3")
METRICS = ("psnr", "ssim", "lpips")
PATTERN = re.compile(r"^(?P<object>.+)_(?P<condition>IN|OUT)_(?P<protocol>blocked_v[23])$")
N_BOOT = 20_000
SEED = 20260815


def percentile_bootstrap(values, rng):
    a = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    means = a[draws].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


rows = []
seen = set()
for path in sorted(INPUT.glob("*.json")):
    match = PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"unexpected filename: {path.name}")
    key = (match["object"], match["condition"], match["protocol"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("results", {})
    missing = [m for m in METRICS if m not in result]
    checkpoint = payload.get("checkpoint", "")
    if missing or not checkpoint.endswith("step-000009999.ckpt"):
        raise ValueError(f"incomplete result {path.name}: missing={missing}, checkpoint={checkpoint}")
    seen.add(key)
    rows.append({
        "object": key[0], "condition": key[1], "protocol": key[2],
        **{m: float(result[m]) for m in METRICS},
        "checkpoint": checkpoint, "source_json": str(path),
    })

expected = {(o, c, p) for o in OBJECTS for c in CONDITIONS for p in PROTOCOLS}
if seen != expected or len(rows) != 16:
    raise ValueError(f"matrix mismatch: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}, n={len(rows)}")

index = {(r["object"], r["condition"], r["protocol"]): r for r in rows}
object_deltas = []
summary = []
rng = np.random.default_rng(SEED)
for protocol in PROTOCOLS:
    for metric in METRICS:
        values = []
        for obj in OBJECTS:
            delta = index[(obj, "OUT", protocol)][metric] - index[(obj, "IN", protocol)][metric]
            values.append(delta)
            object_deltas.append({"object": obj, "protocol": protocol, "metric": metric.upper(), "out_minus_in": delta})
        lo, hi = percentile_bootstrap(values, rng)
        out_better = sum(v < 0 for v in values) if metric == "lpips" else sum(v > 0 for v in values)
        summary.append({
            "method": "Nerfacto", "protocol": protocol, "metric": metric.upper(),
            "n_objects": len(values), "mean_out_minus_in": float(np.mean(values)),
            "ci95_low": float(lo), "ci95_high": float(hi),
            "out_better_objects": out_better, "bootstrap_resamples": N_BOOT, "seed": SEED,
        })


def write_csv(path, data):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)


write_csv(OUT / "nerfacto_v2v3_metrics_rows.csv", rows)
write_csv(OUT / "nerfacto_v2v3_object_deltas.csv", object_deltas)
write_csv(OUT / "nerfacto_v2v3_summary.csv", summary)

lines = [
    "# Nerfacto blocked-v2/v3 four-object sensitivity subset", "",
    "Positive PSNR/SSIM and negative LPIPS favour OUT. Intervals use 20,000 object-level bootstrap resamples.", "",
    "| Protocol | Metric | n | Mean OUT-IN | 95% CI | OUT better |",
    "|---|---|---:|---:|---:|---:|",
]
for r in summary:
    lines.append(f'| {r["protocol"]} | {r["metric"]} | {r["n_objects"]} | {r["mean_out_minus_in"]:+.4f} | [{r["ci95_low"]:+.4f}, {r["ci95_high"]:+.4f}] | {r["out_better_objects"]}/{r["n_objects"]} |')
(OUT / "nerfacto_v2v3_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"validated {len(rows)}/16 results and wrote {len(summary)} summaries")
