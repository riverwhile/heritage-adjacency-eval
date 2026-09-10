#!/usr/bin/env python3
"""Audit temporal and camera-pose separation of interleaved/blocked splits."""
import csv
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from skimage.metrics import structural_similarity

ROOT = Path(os.environ["HERITAGE_DATA_ROOT"])
OBJECTS = ["bronze_cup", "bronze_statue", "cloth_hat", "cloth_shoe",
           "porcelain_plate", "silver_comb", "stone_horse"]
PROTOCOLS = ("al", "blocked_v1", "blocked_v2", "blocked_v3",
             "interleaved_count_matched_seed2024",
             "interleaved_count_matched_seed2025",
             "interleaved_count_matched_seed2026")
OUTPUT_ROOT = Path(os.environ.get("HERITAGE_OUTPUT_ROOT", ROOT))
OUT_JSON = OUTPUT_ROOT / "split_leakage_audit_with_cm.json"
OUT_CSV = OUTPUT_ROOT / "split_leakage_audit_rows_with_cm.csv"


def frame_no(name):
    m = re.findall(r"\d+", Path(name).stem)
    return int(m[-1]) if m else None


def read_names(path):
    return {x.strip() for x in path.read_text().splitlines() if x.strip()}


def split_names(scene, protocol, names):
    if protocol == "al":
        test = set(names[::8])
        return test, set(names) - test, set()
    sd = scene.parent / protocol
    if protocol.startswith("interleaved_count_matched_seed"):
        if not (sd / "test.txt").exists():
            return None
        test = read_names(sd / "test.txt")
        train = read_names(sd / "train.txt")
        removed = read_names(sd / "removed.txt")
        return test, train, removed
    if not (sd / "test.txt").exists():
        return None
    test = read_names(sd / "test.txt")
    buffer = read_names(sd / "buffer.txt")
    return test, set(names) - test - buffer, buffer


def camera_pose(image):
    transform = image.cam_from_world()
    R = transform.rotation.matrix()
    t = np.asarray(transform.translation)
    center = -R.T @ t
    forward = R.T @ np.array([0.0, 0.0, 1.0])
    forward /= np.linalg.norm(forward)
    return center, forward


def angle_deg(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1, 1))))


def thumb(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    h, w = im.shape
    scale = 256.0 / max(h, w)
    return cv2.resize(im, (max(8, round(w * scale)), max(8, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


rows = []
for obj in OBJECTS:
    for condition in ("IN", "OUT"):
        scene = ROOT / obj / condition / "dense_al"
        recon = pycolmap.Reconstruction(str(scene / "sparse" / "0"))
        images = {im.name: im for im in recon.images.values()}
        names = sorted(images)
        poses = {n: camera_pose(images[n]) for n in names}
        centers = np.stack([poses[n][0] for n in names])
        scene_diameter = float(np.linalg.norm(centers.max(0) - centers.min(0))) or 1.0
        thumbs = {n: thumb(scene / "images" / n) for n in names}
        for protocol in PROTOCOLS:
            sp = split_names(scene, protocol, names)
            if sp is None:
                continue
            test, train, buffer = sp
            test, train = sorted(test & set(names)), sorted(train & set(names))
            for name in test:
                c, f = poses[name]
                fn = frame_no(name)
                temporal = min(train, key=lambda n: abs(frame_no(n) - fn))
                center_near = min(train, key=lambda n: np.linalg.norm(poses[n][0] - c))
                orient_near = min(train, key=lambda n: angle_deg(poses[n][1], f))
                ti = thumbs[name]
                sims = []
                if ti is not None:
                    for n in train:
                        tr = thumbs[n]
                        if tr is None:
                            continue
                        h, w = min(ti.shape[0], tr.shape[0]), min(ti.shape[1], tr.shape[1])
                        a = cv2.resize(ti, (w, h)); b = cv2.resize(tr, (w, h))
                        sims.append((structural_similarity(a, b, data_range=255), n))
                best_ssim, best_ssim_name = max(sims) if sims else (None, None)
                tc, tf = poses[temporal]
                row = {
                    "object": obj, "condition": condition, "protocol": protocol,
                    "test_name": name, "n_registered": len(names), "n_train": len(train),
                    "n_test": len(test), "n_buffer": len(buffer & set(names)),
                    "nearest_temporal_name": temporal,
                    "temporal_gap_frames": abs(frame_no(temporal) - fn),
                    "temporal_neighbor_center_distance_norm": float(np.linalg.norm(tc-c)/scene_diameter),
                    "temporal_neighbor_angle_deg": angle_deg(tf, f),
                    "nearest_center_name": center_near,
                    "nearest_center_distance_norm": float(np.linalg.norm(poses[center_near][0]-c)/scene_diameter),
                    "nearest_center_angle_deg": angle_deg(poses[center_near][1], f),
                    "nearest_orientation_name": orient_near,
                    "nearest_orientation_angle_deg": angle_deg(poses[orient_near][1], f),
                    "max_train_ssim": best_ssim, "max_train_ssim_name": best_ssim_name,
                }
                rows.append(row)
                print(f"AUDIT {obj} {condition} {protocol} {name}", flush=True)

numeric = ["temporal_gap_frames", "temporal_neighbor_center_distance_norm",
           "temporal_neighbor_angle_deg", "nearest_center_distance_norm",
           "nearest_center_angle_deg", "nearest_orientation_angle_deg", "max_train_ssim"]
summary = {}
for protocol in PROTOCOLS:
    rr = [r for r in rows if r["protocol"] == protocol]
    if not rr:
        continue
    summary[protocol] = {"n_test_rows": len(rr)}
    for key in numeric:
        vals = np.array([r[key] for r in rr if r[key] is not None], dtype=float)
        summary[protocol][key] = {
            "mean": float(vals.mean()), "median": float(np.median(vals)),
            "q25": float(np.quantile(vals, .25)), "q75": float(np.quantile(vals, .75))}

OUT_JSON.write_text(json.dumps({"definition": {
    "temporal_gap": "absolute ordered-frame difference to the temporally nearest registered training image",
    "center_distance_norm": "Euclidean camera-center distance divided by reconstruction AABB diagonal",
    "center_and_angle_reference": "both camera-center distance and viewing angle are measured to the temporally nearest training image",
    "angle_deg": "angle between COLMAP camera forward axes",
    "max_train_ssim": "maximum grayscale SSIM against any registered training image, long side resized to 256 px"
}, "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print("AUDIT_ALL_DONE")
print(json.dumps(summary, indent=2))
