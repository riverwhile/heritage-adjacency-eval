#!/usr/bin/env python3
"""Create Nerfstudio filename-split datasets from existing COLMAP poses."""
import json
import os
from pathlib import Path

import numpy as np
import pycolmap

ROOT = Path(os.environ["HERITAGE_DATA_ROOT"])
OUT = Path(os.environ["HERITAGE_NERFACTO_DATA"])
OBJECTS = ["bronze_cup", "bronze_statue", "cloth_hat", "cloth_shoe",
           "porcelain_plate", "silver_comb", "stone_horse"]
POSITION_SENSITIVITY_OBJECTS = {"bronze_cup", "bronze_statue", "silver_comb", "stone_horse"}


def read_names(path):
    return {x.strip() for x in path.read_text().splitlines() if x.strip()}


def camera_fields(camera):
    p = list(camera.params)
    common = {"w": camera.width, "h": camera.height, "camera_model": "OPENCV"}
    if camera.model_name == "PINHOLE":
        fx, fy, cx, cy = p
        return {**common, "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
                "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}
    if camera.model_name == "SIMPLE_PINHOLE":
        f, cx, cy = p
        return {**common, "fl_x": f, "fl_y": f, "cx": cx, "cy": cy,
                "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}
    if camera.model_name == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        return {**common, "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
                "k1": k1, "k2": k2, "p1": p1, "p2": p2}
    raise RuntimeError(f"unsupported camera model: {camera.model_name}")


def ns_transform(image):
    tf = image.cam_from_world()
    w2c = np.eye(4)
    w2c[:3, :3] = tf.rotation.matrix()
    w2c[:3, 3] = np.asarray(tf.translation)
    c2w = np.linalg.inv(w2c)
    # Nerfstudio's official COLMAP conversion: OpenCV -> OpenGL, then y/z swap.
    c2w[0:3, 1:3] *= -1
    c2w = c2w[[0, 2, 1, 3], :]
    c2w[2, :] *= -1
    return c2w.tolist()


summary = {"root": str(OUT), "datasets": []}
for obj in OBJECTS:
    for condition in ("IN", "OUT"):
        scene = ROOT / obj / condition / "dense_al"
        recon = pycolmap.Reconstruction(str(scene / "sparse" / "0"))
        registered = {im.name: im for im in recon.images.values()}
        names = sorted(registered)
        cameras = {im.camera_id for im in registered.values()}
        if len(cameras) != 1:
            raise RuntimeError(f"{obj}/{condition}: expected one camera, got {cameras}")
        cam = recon.cameras[next(iter(cameras))]
        protocols = ["al", "blocked_v1"]
        if obj in POSITION_SENSITIVITY_OBJECTS:
            protocols.extend(["blocked_v2", "blocked_v3"])
        for protocol in protocols:
            if protocol == "al":
                test = set(names[::8]); buffer = set()
            else:
                split = ROOT / obj / condition / protocol
                test = read_names(split / "test.txt")
                buffer = read_names(split / "buffer.txt")
            train = set(names) - test - buffer
            if not test <= set(names) or train & test or train & buffer or test & buffer:
                raise RuntimeError(f"invalid split: {obj}/{condition}/{protocol}")
            ds = OUT / obj / condition / protocol
            image_dir = ds / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for name in sorted(train | test):
                split_name = "eval" if name in test else "train"
                linked_name = f"{split_name}_{name}"
                link = image_dir / linked_name
                if not link.exists():
                    link.symlink_to((scene / "images" / name).resolve())
                frames.append({"file_path": f"images/{linked_name}",
                               "transform_matrix": ns_transform(registered[name]),
                               "colmap_im_id": registered[name].image_id,
                               "source_name": name,
                               "split": split_name})
            transforms = {**camera_fields(cam), "frames": frames,
                          "source_colmap": str(scene / "sparse" / "0"),
                          "protocol": protocol}
            (ds / "transforms.json").write_text(json.dumps(transforms, indent=2))
            (ds / "train_names.txt").write_text("\n".join(sorted(train)) + "\n")
            (ds / "eval_names.txt").write_text("\n".join(sorted(test)) + "\n")
            (ds / "buffer_names.txt").write_text("\n".join(sorted(buffer)) + ("\n" if buffer else ""))
            item = {"object": obj, "condition": condition, "protocol": protocol,
                    "registered": len(names), "train": len(train), "eval": len(test),
                    "buffer": len(buffer), "frames_json": len(frames),
                    "camera_model": cam.model_name, "dataset": str(ds),
                    "train_names": sorted(train), "eval_names": sorted(test),
                    "buffer_names": sorted(buffer)}
            summary["datasets"].append(item)
            print("DATASET=" + json.dumps({k: item[k] for k in
                  ("object", "condition", "protocol", "registered", "train", "eval", "buffer")}), flush=True)

summary["totals"] = {"datasets": len(summary["datasets"])}
for protocol in ("al", "blocked_v1", "blocked_v2", "blocked_v3"):
    summary["totals"][f"{protocol}_eval_frames"] = sum(
        x["eval"] for x in summary["datasets"] if x["protocol"] == protocol
    )
(OUT / "dataset_inventory.json").write_text(json.dumps(summary, indent=2))
print("NERFACTO_DATASETS_ALL_DONE=" + json.dumps(summary["totals"]))
