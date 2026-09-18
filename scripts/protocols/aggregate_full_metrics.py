#!/usr/bin/env python3
"""Aggregate completed blocked-split PSNR/SSIM/LPIPS results."""
import csv
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ["HERITAGE_DATA_ROOT"])
OUT_JSON = ROOT / "blocked_full_metrics_summary.json"
OUT_CSV = ROOT / "blocked_full_metrics_rows.csv"
METRICS = ("PSNR", "SSIM", "LPIPS")


def better(metric, delta):
    return delta > 0 if metric != "LPIPS" else delta < 0


def load_metric(path, method):
    return json.loads(path.read_text())[method]


rows = []
for path in sorted(ROOT.glob("*/*/*/results.json")):
    obj, condition, model = path.parts[-4:-1]
    kind = None
    protocol = None
    seed = None
    if re.fullmatch(r"gs7k_(al|blocked_v[123])", model):
        kind = "baseline"
        protocol = model.removeprefix("gs7k_")
        methods = ["ours_7000"]
    elif re.fullmatch(r"(plain|veil)10k_blocked_v1_seed[0-2]", model):
        kind = "plain" if model.startswith("plain") else "veil"
        protocol = "blocked_v1"
        seed = int(model.rsplit("seed", 1)[1])
        methods = ["ours_10000"] + (["composite_10000"] if kind == "veil" else [])
    else:
        continue
    data = json.loads(path.read_text())
    for method in methods:
        if method not in data:
            continue
        r = {"object": obj, "condition": condition, "model": model,
             "kind": kind, "protocol": protocol, "seed": seed,
             "method": method}
        r.update({m: float(data[method][m]) for m in METRICS})
        rows.append(r)


def compare(pair_rows, left_name, right_name):
    out = []
    for obj in sorted({r["object"] for r in pair_rows}):
        rr = [r for r in pair_rows if r["object"] == obj]
        left = next((r for r in rr if r["condition"] == left_name), None)
        right = next((r for r in rr if r["condition"] == right_name), None)
        if not left or not right:
            continue
        item = {"object": obj}
        for m in METRICS:
            item[f"{left_name}_{m}"] = left[m]
            item[f"{right_name}_{m}"] = right[m]
            item[f"delta_{right_name}_minus_{left_name}_{m}"] = right[m] - left[m]
        out.append(item)
    summary = {"n": len(out)}
    for m in METRICS:
        ds = [x[f"delta_{right_name}_minus_{left_name}_{m}"] for x in out]
        summary[m] = {
            "mean_delta": sum(ds) / len(ds) if ds else None,
            f"{right_name}_better": sum(better(m, x) for x in ds),
            f"{left_name}_better": sum(better(m, -x) for x in ds),
        }
    return {"rows": out, "summary": summary}


comparisons = {}
for protocol in ("al", "blocked_v1", "blocked_v2", "blocked_v3"):
    selected = [r for r in rows if r["kind"] == "baseline" and r["protocol"] == protocol]
    comparisons[f"baseline_{protocol}_OUT_vs_IN"] = compare(selected, "IN", "OUT")

plain = [r for r in rows if r["kind"] == "plain" and r["seed"] == 0 and r["method"] == "ours_10000"]
for method in ("ours_10000", "composite_10000"):
    veil = [r for r in rows if r["kind"] == "veil" and r["seed"] == 0 and r["method"] == method]
    joined = []
    for p in plain:
        v = next((x for x in veil if x["object"] == p["object"]), None)
        if v:
            joined.extend([{**p, "condition": "plain"}, {**v, "condition": "veil"}])
    comparisons[f"seed0_veil_{method}_vs_plain"] = compare(joined, "plain", "veil")

out = {"n_metric_rows": len(rows),
       "metrics": list(METRICS), "comparisons": comparisons}
OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    fields = ["object", "condition", "model", "kind", "protocol", "seed", "method", *METRICS]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(json.dumps({"n_metric_rows": len(rows),
                  "summary": {k: v["summary"] for k, v in comparisons.items()}}, indent=2))
