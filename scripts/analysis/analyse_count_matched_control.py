#!/usr/bin/env python3
"""Aggregate the fixed-seed training-count-matched interleaved control."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "tmp" / "count_matched_results"
TABLES = ROOT / "paper" / "sci_assets" / "tables"
ABS = TABLES / "absolute_scores_all_models.csv"
OBJECTS = ["bronze_cup", "bronze_statue", "cloth_hat", "cloth_shoe",
           "porcelain_plate", "silver_comb", "stone_horse"]
METRICS = ("PSNR", "SSIM", "LPIPS")
SEEDS = (2024, 2025, 2026)


def mean(xs):
    return sum(xs) / len(xs)


def percentile(xs, p):
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] * (hi - k) + ys[hi] * (k - lo)


def boot(values, seed=20260818, n=20000):
    rng = random.Random(seed)
    sims = [mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(n)]
    return mean(values), percentile(sims, .025), percentile(sims, .975)


seed_rows = []
for p in sorted(INPUT.rglob("results.json")):
    parts = p.parts
    tag = p.parent.name
    seed = int(tag.rsplit("seed", 1)[1])
    condition = p.parent.parent.name
    obj = p.parent.parent.parent.name
    scores = json.loads(p.read_text())["ours_7000"]
    seed_rows.append({"object": obj, "condition": condition, "seed": seed,
                      **{m: float(scores[m]) for m in METRICS}})

expected = {(o, c, s) for o in OBJECTS for c in ("IN", "OUT") for s in SEEDS}
actual = {(r["object"], r["condition"], r["seed"]) for r in seed_rows}
assert actual == expected, (sorted(expected - actual), sorted(actual - expected))

with (TABLES / "count_matched_seed_scores.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(seed_rows[0]))
    w.writeheader(); w.writerows(seed_rows)

by_condition = defaultdict(lambda: defaultdict(list))
for r in seed_rows:
    for m in METRICS:
        by_condition[(r["object"], r["condition"])][m].append(r[m])
condition_rows = []
for (obj, condition), vals in sorted(by_condition.items()):
    condition_rows.append({"object": obj, "condition": condition,
                           **{m: mean(vals[m]) for m in METRICS},
                           **{f"{m}_seed_range": max(vals[m])-min(vals[m]) for m in METRICS}})
with (TABLES / "count_matched_condition_means.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(condition_rows[0]))
    w.writeheader(); w.writerows(condition_rows)

cm = {(r["object"], r["condition"]): r for r in condition_rows}
baseline = {}
with ABS.open(newline="") as f:
    for r in csv.DictReader(f):
        if r["method"] == "3DGS" and r["protocol"] in ("al", "blocked_v1"):
            baseline[(r["object"], r["condition"], r["protocol"])] = {
                m: float(r[m]) for m in METRICS}

object_rows = []
for obj in OBJECTS:
    for m in METRICS:
        al = baseline[(obj, "OUT", "al")][m] - baseline[(obj, "IN", "al")][m]
        count = cm[(obj, "OUT")][m] - cm[(obj, "IN")][m]
        blocked = (baseline[(obj, "OUT", "blocked_v1")][m] -
                   baseline[(obj, "IN", "blocked_v1")][m])
        object_rows.append({"object": obj, "metric": m,
                            "interleaved_effect": al,
                            "count_matched_effect": count,
                            "blocked_v1_effect": blocked,
                            "training_count_effect_al_minus_cm": al-count,
                            "adjacency_effect_cm_minus_blocked": count-blocked,
                            "total_attenuation_al_minus_blocked": al-blocked})
with (TABLES / "count_matched_object_effects.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(object_rows[0]))
    w.writeheader(); w.writerows(object_rows)

summary = []
for m in METRICS:
    rr = [r for r in object_rows if r["metric"] == m]
    for key, label in (
        ("interleaved_effect", "interleaved"),
        ("count_matched_effect", "interleaved_count_matched"),
        ("blocked_v1_effect", "blocked_v1"),
        ("training_count_effect_al_minus_cm", "training_count_component_al_minus_count_matched"),
        ("adjacency_effect_cm_minus_blocked", "adjacency_component_count_matched_minus_blocked"),
        ("total_attenuation_al_minus_blocked", "total_al_minus_blocked")):
        est, lo, hi = boot([r[key] for r in rr], seed=20260818 + len(summary))
        summary.append({"metric": m, "quantity": label, "mean": est,
                        "ci_low": lo, "ci_high": hi})
with (TABLES / "count_matched_summary.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader(); w.writerows(summary)

lines = ["# Training-count-matched interleaved control", "",
         "Three fixed deletion seeds were averaged within each object-condition before object-level inference.", "",
         "| Metric | Quantity | Mean [95% object bootstrap CI] |", "|---|---|---:|"]
for r in summary:
    lines.append(f"| {r['metric']} | {r['quantity']} | {r['mean']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
lines += ["", "Interpretation: if the training-count component is small relative to the adjacency component, the blocked attenuation cannot be explained by image-count reduction alone."]
(TABLES / "count_matched_summary.md").write_text("\n".join(lines) + "\n")

print(json.dumps({"n_seed_rows": len(seed_rows), "n_object_rows": len(object_rows),
                  "summary": summary}, indent=2))
