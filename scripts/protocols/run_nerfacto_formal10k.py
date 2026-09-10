#!/usr/bin/env python3
"""Run the fixed Nerfacto 10k cross-protocol matrix."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


ENV_PYTHON = Path(os.environ.get("NERFSTUDIO_PYTHON", sys.executable))
NS_TRAIN = Path(os.environ.get("NS_TRAIN", "ns-train"))
NS_EVAL = Path(os.environ.get("NS_EVAL", "ns-eval"))
DATA_ROOT = Path(os.environ["HERITAGE_NERFACTO_DATA"])
RUN_ROOT = Path(os.environ["HERITAGE_NERFACTO_RUNS"])
STATUS_PATH = RUN_ROOT / "status.json"
FIRST_ERROR_PATH = RUN_ROOT / "FIRST_ERROR.json"
DONE_SUCCESS = RUN_ROOT / "DONE_SUCCESS"
DONE_WITH_ERRORS = RUN_ROOT / "DONE_WITH_ERRORS"
TIMESTAMP = "seed2024_10k"
OBJECTS = (
    "bronze_cup",
    "bronze_statue",
    "cloth_hat",
    "cloth_shoe",
    "porcelain_plate",
    "silver_comb",
    "stone_horse",
)
CONDITIONS = ("IN", "OUT")
PROTOCOLS = ("al", "blocked_v1")
GPUS = tuple(int(value) for value in os.environ.get("CUDA_DEVICES", "0").split(","))
REUSED_NAME = None
REUSED_INFO = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def task_name(obj: str, condition: str, protocol: str) -> str:
    return f"{obj}_{condition}_{protocol}"


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] COMMAND: {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write(f"\n[{utc_now()}] RETURN_CODE={completed.returncode}\n")
        return completed.returncode


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "logs").mkdir(exist_ok=True)
    (RUN_ROOT / "results").mkdir(exist_ok=True)
    (RUN_ROOT / "renders").mkdir(exist_ok=True)

    tasks = []
    for obj in OBJECTS:
        for condition in CONDITIONS:
            for protocol in PROTOCOLS:
                name = task_name(obj, condition, protocol)
                if name != REUSED_NAME:
                    tasks.append({"name": name, "object": obj, "condition": condition, "protocol": protocol})

    state = {
        "schema_version": 1,
        "started_utc": utc_now(),
        "finished_utc": None,
        "fixed_config": {
            "iterations": 10000,
            "seed": 2024,
            "camera_optimizer": "off",
            "eval_mode": "filename",
            "physical_gpus": list(GPUS),
        },
        "reused": REUSED_INFO,
        "total_formal_tasks": len(tasks),
        "completed": [],
        "failed": [],
        "running": {},
        "pending": [task["name"] for task in tasks],
    }
    lock = threading.Lock()
    atomic_json(STATUS_PATH, state)

    work: queue.Queue[dict | None] = queue.Queue()
    for task in tasks:
        work.put(task)
    for _ in GPUS:
        work.put(None)

    def save_state() -> None:
        atomic_json(STATUS_PATH, state)

    def mark_failure(task: dict, gpu: int, stage: str, rc: int, log_path: Path) -> None:
        failure = {
            **task,
            "gpu": gpu,
            "stage": stage,
            "return_code": rc,
            "log": str(log_path),
            "failed_utc": utc_now(),
        }
        with lock:
            state["running"].pop(str(gpu), None)
            state["failed"].append(failure)
            save_state()
            if not FIRST_ERROR_PATH.exists():
                atomic_json(FIRST_ERROR_PATH, failure)

    def worker(gpu: int) -> None:
        while True:
            task = work.get()
            if task is None:
                work.task_done()
                return

            name = task["name"]
            dataset = DATA_ROOT / task["object"] / task["condition"] / task["protocol"]
            log_path = RUN_ROOT / "logs" / f"{name}.log"
            result_path = RUN_ROOT / "results" / f"{name}.json"
            render_path = RUN_ROOT / "renders" / name
            config_path = RUN_ROOT / name / "nerfacto" / TIMESTAMP / "config.yml"
            checkpoint = RUN_ROOT / name / "nerfacto" / TIMESTAMP / "nerfstudio_models" / "step-000009999.ckpt"

            with lock:
                state["pending"].remove(name)
                state["running"][str(gpu)] = {**task, "started_utc": utc_now(), "log": str(log_path)}
                save_state()

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            train_cmd = [
                str(NS_TRAIN),
                "nerfacto",
                "--data", str(dataset),
                "--output-dir", str(RUN_ROOT),
                "--experiment-name", name,
                "--timestamp", TIMESTAMP,
                "--machine.seed", "2024",
                "--max-num-iterations", "10000",
                "--steps-per-save", "2000",
                "--steps-per-eval-image", "1000",
                "--steps-per-eval-all-images", "10000",
                "--vis", "tensorboard",
                "--viewer.quit-on-train-completion", "True",
                "--pipeline.model.camera-optimizer.mode", "off",
                "nerfstudio-data",
                "--eval-mode", "filename",
            ]
            train_rc = run_logged(train_cmd, log_path, env)
            if train_rc != 0 or not checkpoint.is_file():
                mark_failure(task, gpu, "train", train_rc, log_path)
                work.task_done()
                continue

            eval_cmd = [
                str(NS_EVAL),
                "--load-config", str(config_path),
                "--output-path", str(result_path),
                "--render-output-path", str(render_path),
            ]
            eval_rc = run_logged(eval_cmd, log_path, env)
            valid_result = False
            if eval_rc == 0 and result_path.is_file():
                try:
                    json.loads(result_path.read_text(encoding="utf-8"))
                    valid_result = True
                except (OSError, json.JSONDecodeError):
                    pass
            if not valid_result:
                mark_failure(task, gpu, "eval", eval_rc, log_path)
                work.task_done()
                continue

            with lock:
                started = state["running"].pop(str(gpu))["started_utc"]
                state["completed"].append(
                    {
                        **task,
                        "gpu": gpu,
                        "started_utc": started,
                        "finished_utc": utc_now(),
                        "checkpoint": str(checkpoint),
                        "result": str(result_path),
                        "renders": str(render_path),
                        "log": str(log_path),
                    }
                )
                save_state()
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}") for gpu in GPUS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with lock:
        state["finished_utc"] = utc_now()
        save_state()
    marker = DONE_WITH_ERRORS if state["failed"] else DONE_SUCCESS
    marker.write_text(
        json.dumps(
            {
                "finished_utc": state["finished_utc"],
                "completed": len(state["completed"]),
                "failed": len(state["failed"]),
                "reused": REUSED_NAME,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 1 if state["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
