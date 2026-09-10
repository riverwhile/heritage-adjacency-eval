#!/usr/bin/env python3
"""Summarise global nearest-camera geometry from an adjacency-audit JSON."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


KEYS = (
    "nearest_center_distance_norm",
    "nearest_center_angle_deg",
    "nearest_orientation_angle_deg",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.audit_json.read_text(encoding="utf-8"))["rows"]
    output = []
    for protocol in sorted({row["protocol"] for row in rows}):
        selected = [row for row in rows if row["protocol"] == protocol]
        record = {"protocol": protocol, "n_test_rows": len(selected)}
        for key in KEYS:
            values = np.asarray([row[key] for row in selected], dtype=float)
            record[f"{key}_mean"] = float(values.mean())
            record[f"{key}_median"] = float(np.median(values))
        output.append(record)

    fieldnames = list(output[0])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
