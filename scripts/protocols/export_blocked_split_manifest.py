#!/usr/bin/env python3
"""Validate and export machine-readable blocked-v1/v2/v3 definitions."""
import csv
import json
import os
from pathlib import Path

ROOT = Path(os.environ["HERITAGE_DATA_ROOT"])
OUT = Path(os.environ.get("HERITAGE_OUTPUT_ROOT", ROOT))
OBJECTS = ["bronze_cup", "bronze_statue", "cloth_hat", "cloth_shoe",
           "porcelain_plate", "silver_comb", "stone_horse"]
ANCHORS = {"blocked_v1": (0.20, 0.50, 0.80),
           "blocked_v2": (0.12, 0.42, 0.72),
           "blocked_v3": (0.28, 0.58, 0.88)}
LENGTHS = (4, 4, 2)
BUFFER_RADIUS = 2


def read(path):
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def generate(n, anchors):
    test = set()
    sectors = []
    for anchor, length in zip(anchors, LENGTHS):
        center = round(anchor * n)
        start = max(BUFFER_RADIUS,
                    min(n - length - BUFFER_RADIUS, center - length // 2))
        indices = list(range(start, start + length))
        test.update(indices)
        sectors.append({"anchor": anchor, "rounded_center_index": center,
                        "start_index": start, "end_index_inclusive": start + length - 1,
                        "length": length})
    buffer = set()
    for i in test:
        for j in range(max(0, i - BUFFER_RADIUS), min(n, i + BUFFER_RADIUS + 1)):
            if j not in test:
                buffer.add(j)
    train = set(range(n)) - test - buffer
    return sorted(train), sorted(test), sorted(buffer), sectors


rows = []
manifest = {"ordering": "lexicographic order of registered image filenames",
            "indexing": "zero-based", "rounding": "Python built-in round",
            "sector_lengths": list(LENGTHS), "buffer_radius_images": BUFFER_RADIUS,
            "protocol_anchors": {k: list(v) for k, v in ANCHORS.items()}, "orbits": []}
for obj in OBJECTS:
    for condition in ("IN", "OUT"):
        scene = ROOT / obj / condition / "dense_al"
        names = sorted(p.name for p in (scene / "images").iterdir() if p.is_file())
        n = len(names)
        for protocol, anchors in ANCHORS.items():
            train, test, buffer, sectors = generate(n, anchors)
            split = ROOT / obj / condition / protocol
            actual_test = read(split / "test.txt")
            actual_buffer = read(split / "buffer.txt")
            expected_test = [names[i] for i in test]
            expected_buffer = [names[i] for i in buffer]
            if actual_test != expected_test or actual_buffer != expected_buffer:
                raise RuntimeError(f"split mismatch: {obj} {condition} {protocol}")
            item = {"object": obj, "condition": condition, "protocol": protocol,
                    "n_registered": n, "anchors": list(anchors), "sectors": sectors,
                    "test_indices_zero_based": test, "buffer_indices_zero_based": buffer,
                    "train_indices_zero_based": train, "test_names": expected_test,
                    "buffer_names": expected_buffer, "n_train": len(train),
                    "n_test": len(test), "n_buffer": len(buffer)}
            manifest["orbits"].append(item)
            rows.append({"object": obj, "condition": condition, "protocol": protocol,
                         "n_registered": n, "anchors": ";".join(map(str, anchors)),
                         "test_indices_zero_based": ";".join(map(str, test)),
                         "buffer_indices_zero_based": ";".join(map(str, buffer)),
                         "n_train": len(train), "n_test": len(test),
                         "n_buffer": len(buffer),
                         "test_names": ";".join(expected_test),
                         "buffer_names": ";".join(expected_buffer)})

(OUT / "blocked_split_manifest.json").write_text(json.dumps(manifest, indent=2))
with (OUT / "blocked_split_manifest.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"VALIDATED_BLOCKED_SPLITS orbits={len(manifest['orbits'])}")
