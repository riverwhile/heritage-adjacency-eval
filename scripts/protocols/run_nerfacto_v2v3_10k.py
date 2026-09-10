#!/usr/bin/env python3
"""Run the four-object Nerfacto blocked-v2/v3 position-sensitivity matrix."""

import os
from pathlib import Path

import run_nerfacto_formal10k as runner


runner.RUN_ROOT = Path(os.environ["HERITAGE_NERFACTO_POSITION_RUNS"])
runner.STATUS_PATH = runner.RUN_ROOT / "status.json"
runner.FIRST_ERROR_PATH = runner.RUN_ROOT / "FIRST_ERROR.json"
runner.DONE_SUCCESS = runner.RUN_ROOT / "DONE_SUCCESS"
runner.DONE_WITH_ERRORS = runner.RUN_ROOT / "DONE_WITH_ERRORS"
runner.OBJECTS = ("bronze_cup", "bronze_statue", "silver_comb", "stone_horse")
runner.PROTOCOLS = ("blocked_v2", "blocked_v3")
runner.REUSED_NAME = None
runner.REUSED_INFO = None


if __name__ == "__main__":
    raise SystemExit(runner.main())
