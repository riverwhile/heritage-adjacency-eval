#!/usr/bin/env python3
"""Run deterministic training-count-matched interleaved 3DGS controls.

The every-eighth test set is unchanged. Additional training images are ignored
until the retained training count equals blocked-v1. Immediate ordered
neighbours (+/-1) of every test image are protected so that this control keeps
the defining short-baseline adjacency of the interleaved protocol. Three fixed
seeds quantify sensitivity to which non-neighbour training images are removed.
"""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ["HERITAGE_DATA_ROOT"])
GS = Path(os.environ["GAUSSIAN_SPLATTING_ROOT"])
PY = os.environ.get("HERITAGE_PYTHON", sys.executable)
OBJECTS = ["bronze_cup", "bronze_statue", "cloth_hat", "cloth_shoe",
           "porcelain_plate", "silver_comb", "stone_horse"]
SEEDS = (2024, 2025, 2026)
GPU = os.environ["COUNT_MATCH_GPU"]
SHARD = int(os.environ.get("COUNT_MATCH_SHARD", "0"))
N_SHARDS = int(os.environ.get("COUNT_MATCH_N_SHARDS", "1"))


def read_names(path):
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def make_split(names, target_train, seed):
    test_idx = list(range(0, len(names), 8))
    test = {names[i] for i in test_idx}
    train = [n for n in names if n not in test]
    protected = set()
    for i in test_idx:
        for j in (i - 1, i + 1):
            if 0 <= j < len(names) and names[j] not in test:
                protected.add(names[j])
    n_remove = len(train) - target_train
    candidates = [n for n in train if n not in protected]
    if n_remove < 0 or n_remove > len(candidates):
        raise ValueError((len(names), len(train), target_train, len(candidates)))
    rng = random.Random(seed)
    removed = sorted(rng.sample(candidates, n_remove))
    retained = sorted(set(train) - set(removed))
    assert len(retained) == target_train
    assert protected <= set(retained)
    return sorted(test), removed, retained, sorted(protected)


def run(cmd, env):
    return subprocess.run(cmd, cwd=GS, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


jobs = [(obj, cond, seed) for obj in OBJECTS for cond in ("IN", "OUT")
        for seed in SEEDS]
jobs = [job for i, job in enumerate(jobs) if i % N_SHARDS == SHARD]
status_path = ROOT / f"count_matched_status_shard{SHARD}.json"
report = {"protocol": "interleaved-count-matched", "gpu": GPU,
          "shard": SHARD, "n_shards": N_SHARDS, "seeds": list(SEEDS),
          "started": time.strftime("%FT%T%z"), "jobs": []}

for obj, cond, seed in jobs:
    scene = ROOT / obj / cond / "dense_al"
    names = sorted(p.name for p in (scene / "images").iterdir() if p.is_file())
    blocked_test = read_names(ROOT / obj / cond / "blocked_v1" / "test.txt")
    blocked_buffer = read_names(ROOT / obj / cond / "blocked_v1" / "buffer.txt")
    target_train = len(names) - len(blocked_test) - len(blocked_buffer)
    test, removed, retained, protected = make_split(names, target_train, seed)
    split = ROOT / obj / cond / f"interleaved_count_matched_seed{seed}"
    split.mkdir(exist_ok=True)
    (split / "test.txt").write_text("\n".join(test) + "\n")
    (split / "removed.txt").write_text("\n".join(removed) + "\n")
    (split / "train.txt").write_text("\n".join(retained) + "\n")
    (split / "protected_neighbours.txt").write_text("\n".join(protected) + "\n")
    manifest = {"object": obj, "condition": cond, "seed": seed,
                "n_registered": len(names), "n_test": len(test),
                "n_train": len(retained), "n_removed": len(removed),
                "blocked_v1_target_train": target_train,
                "test_rule": "ordered images[::8]",
                "protected_rule": "ordered immediate neighbours (+/-1) of each test image",
                "removal_rule": "fixed-seed sample from non-test, non-protected images"}
    (split / "manifest.json").write_text(json.dumps(manifest, indent=2))

    model = ROOT / obj / cond / f"gs7k_interleaved_count_matched_seed{seed}"
    model.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU, GS_TEST_FILE=str(split / "test.txt"),
               GS_IGNORE_FILE=str(split / "removed.txt"), GS_VEILING="",
               OPENBLAS_NUM_THREADS="4")
    train_cmd = [PY, "train.py", "-s", str(scene), "-m", str(model), "--eval",
                 "--iterations", "7000", "--test_iterations", "7000",
                 "--save_iterations", "7000", "--disable_viewer", "--quiet"]
    p = run(train_cmd, env)
    (model / "train.log").write_text(p.stdout)
    item = dict(manifest, train_rc=p.returncode, model=str(model))
    if p.returncode == 0:
        q = run([PY, "render.py", "-m", str(model), "--iteration", "7000",
                 "--eval", "--quiet"], env)
        (model / "render.log").write_text(q.stdout)
        item["render_rc"] = q.returncode
        if q.returncode == 0:
            m = run([PY, "metrics.py", "-m", str(model)], env)
            (model / "metric_eval.log").write_text(m.stdout)
            item["metric_rc"] = m.returncode
            item["has_results"] = (model / "results.json").exists()
    report["jobs"].append(item)
    status_path.write_text(json.dumps(report, indent=2))
    print("COUNT_MATCH_ITEM=" + json.dumps(item), flush=True)

report["finished"] = time.strftime("%FT%T%z")
report["success"] = all(j.get("train_rc") == 0 and j.get("render_rc") == 0
                        and j.get("metric_rc") == 0 and j.get("has_results")
                        for j in report["jobs"])
status_path.write_text(json.dumps(report, indent=2))
print(f"COUNT_MATCH_ALL_DONE shard={SHARD} success={report['success']}", flush=True)
