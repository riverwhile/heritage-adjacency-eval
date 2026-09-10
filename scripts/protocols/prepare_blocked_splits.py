#!/usr/bin/env python3
"""Create deterministic contiguous test blocks and two-frame buffers."""
from pathlib import Path
import json
import os

ROOT = Path(os.environ["HERITAGE_DATA_ROOT"])
OBJECTS = ["bronze_cup", "bronze_statue", "cloth_hat", "cloth_shoe",
           "porcelain_plate", "silver_comb", "stone_horse"]

def build(names):
    n = len(names)
    if n < 30:
        raise RuntimeError(f"too few registered views: {n}")
    lengths = (4, 4, 2)
    centers = (round(0.20*n), round(0.50*n), round(0.80*n))
    test_idx = set()
    for center, length in zip(centers, lengths):
        start = max(2, min(n-length-2, center-length//2))
        test_idx.update(range(start, start+length))
    buffer_idx = set()
    for i in test_idx:
        for j in range(max(0, i-2), min(n, i+3)):
            if j not in test_idx:
                buffer_idx.add(j)
    train_idx = set(range(n)) - test_idx - buffer_idx
    return ([names[i] for i in sorted(train_idx)],
            [names[i] for i in sorted(test_idx)],
            [names[i] for i in sorted(buffer_idx)])

summary = {}
for obj in OBJECTS:
    summary[obj] = {}
    for cond in ("IN", "OUT"):
        scene = ROOT / obj / cond / "dense_al"
        names = sorted(p.name for p in (scene / "images").iterdir() if p.is_file())
        train, test, buffer = build(names)
        out = ROOT / obj / cond / "blocked_v1"
        out.mkdir(exist_ok=True)
        (out / "train.txt").write_text("\n".join(train)+"\n")
        (out / "test.txt").write_text("\n".join(test)+"\n")
        (out / "buffer.txt").write_text("\n".join(buffer)+"\n")
        summary[obj][cond] = {"all": len(names), "train": len(train),
                              "test": len(test), "buffer": len(buffer),
                              "test_names": test}
(ROOT / "blocked_v1_split_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
