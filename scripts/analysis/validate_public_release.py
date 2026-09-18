#!/usr/bin/env python3
"""Validate the matrices, metadata, statistics, and publication hygiene."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from analyse_protocol_attenuation import load_rows, paired_effects, summarise


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "derived"
MANIFESTS = ROOT / "data" / "manifests"
OBJECTS = {"bronze_cup", "bronze_statue", "cloth_hat", "cloth_shoe", "porcelain_plate", "silver_comb", "stone_horse"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_matrices() -> None:
    manifest = read_csv(MANIFESTS / "blocked_split_manifest.csv")
    require(len(manifest) == 42, f"expected 42 blocked manifest rows, found {len(manifest)}")
    require({row["object"] for row in manifest} == OBJECTS, "blocked manifest object set differs from the manuscript cohort")
    require({row["protocol"] for row in manifest} == {"blocked_v1", "blocked_v2", "blocked_v3"}, "blocked manifest protocol set is incomplete")
    require(all(int(row["n_test"]) == 10 and int(row["n_buffer"]) == 12 for row in manifest), "unexpected blocked split sizes")

    full = read_csv(DATA / "blocked_full_metrics_rows.csv")
    baseline = [row for row in full if row["kind"] == "baseline" and row["method"] == "ours_7000"]
    require(len(baseline) == 56, f"expected 56 3DGS baseline rows, found {len(baseline)}")

    matched = read_csv(DATA / "count_matched_seed_scores.csv")
    require(len(matched) == 42, f"expected 42 count-matched seed rows, found {len(matched)}")
    require({int(row["seed"]) for row in matched} == {2024, 2025, 2026}, "count-matched deletion seeds are incomplete")

    nerfacto_main = read_csv(DATA / "nerfacto_10k_metrics_rows.csv")
    require(len(nerfacto_main) == 28, f"expected 28 main Nerfacto rows, found {len(nerfacto_main)}")
    nerfacto_positions = read_csv(DATA / "nerfacto_v2v3_metrics_rows.csv")
    require(len(nerfacto_positions) == 16, f"expected 16 alternative-position Nerfacto rows, found {len(nerfacto_positions)}")

    video = read_csv(DATA / "source_video_metadata.csv")
    require(len(video) == 7 and {row["object"] for row in video} == OBJECTS, "source-video metadata cohort is incomplete")
    require(all((row["encoded_width"], row["encoded_height"], row["video_fps"]) == ("1280", "720", "30") for row in video), "unexpected source-video encoding metadata")
    require(all(float(row["in_duration_s"]) > float(row["out_duration_s"]) for row in video), "duration ordering differs from Supplementary Table S12")


def validate_small_sample_results() -> None:
    effects = paired_effects(load_rows(DATA))
    _, _, computed = summarise(effects)
    published = read_csv(DATA / "sign_flip_sensitivity.csv")
    expected = {(row["method"], row["metric"]): row for row in published}
    require(len(computed) == len(expected) == 6, "sign-flip table must contain six comparisons")
    for row in computed:
        key = (str(row["method"]), str(row["metric"]))
        require(key in expected, f"missing sign-flip row {key}")
        recorded = expected[key]
        require(int(row["extreme_assignments"]) == int(recorded["extreme_assignments"]), f"sign-flip count differs for {key}")
        require(abs(float(row["two_sided_p"]) - float(recorded["two_sided_p"])) < 1e-12, f"sign-flip value differs for {key}")


def validate_publication_hygiene() -> None:
    suffixes = {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".csv"}
    path_pattern = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|Users)/)")
    problems = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path_pattern.search(text):
            problems.append(f"machine-specific path in {path.relative_to(ROOT)}")
    require(not problems, "; ".join(problems))


def main() -> int:
    validate_matrices()
    validate_small_sample_results()
    validate_publication_hygiene()
    print("public release validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
