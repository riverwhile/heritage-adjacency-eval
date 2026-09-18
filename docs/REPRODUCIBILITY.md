# Reproducibility scope

## What can be verified publicly

The public release contains complete blocked-split assignments, object-condition metric rows, count-matched deletion-seed scores, Nerfacto protocol scores, compact adjacency summaries, source-video stream metadata, and object-level uncertainty inputs. These materials support independent checks of the numerical aggregation without disclosing collection-controlled imagery.

The public analysis commands are:

| Command | Public input | Principal output |
| --- | --- | --- |
| `python scripts/analysis/validate_public_release.py` | `data/` | Matrix, manifest, metadata, path-hygiene, and S11 checks |
| `python scripts/analysis/analyse_protocol_attenuation.py --output-dir artifacts` | 3DGS and Nerfacto metric rows | Object attenuation, bootstrap intervals, leave-one-out values, exact sign-flip values |
| `python scripts/analysis/analyse_count_matched_control.py --output-dir artifacts` | Count-matched seed scores and 3DGS metric rows | Count-matched condition means, object effects, and summaries |
| `python scripts/analysis/aggregate_nerfacto_v2v3.py --output-dir artifacts` | Nerfacto v2/v3 metric rows | Four-object position-sensitivity summaries |
| `python scripts/analysis/build_blocked_protocol_results.py --output-dir artifacts` | Blocked-protocol master table | Protocol summary copy and figures |

Bootstrap confidence intervals use 20,000 resamples of objects. Frames are not treated as independent inferential units. Supplementary Table S11 additionally enumerates all 128 sign assignments for each seven-object paired attenuation vector.

Install `requirements.txt` for these public checks. Authorised restricted-data workflows additionally require `requirements-restricted.txt` and the original 3DGS or Nerfstudio environments described below.

## Protocol definitions

The interleaved protocol holds out approximately every eighth ordered image. Buffered blocked protocols hold out three contiguous sectors of lengths 4, 4, and 2, with a two-image exclusion buffer. Normalised anchors are 0.20/0.50/0.80 for blocked-v1, 0.12/0.42/0.72 for blocked-v2, and 0.28/0.58/0.88 for blocked-v3.

The training-count-matched control preserves the interleaved test set and its immediate temporal neighbours, then deterministically removes other training images to match blocked-v1 training counts. Seeds 2024--2026 are averaged within each object-condition before object-level inference. These are deletion seeds, not model optimisation seeds.

## Restricted-data workflows

The following inputs are not public:

- cultural-object RGB images and source videos;
- COLMAP reconstructions and collection metadata;
- trained Gaussian and Nerfacto checkpoints;
- rendered evaluation images;
- operational logs and machine status files.

The omission of images and cameras prevents full model retraining and recomputation of per-view geometric audits. Authorised users can supply paths through these variables:

- `HERITAGE_DATA_ROOT`: registered images, COLMAP reconstruction, and split directories;
- `HERITAGE_OUTPUT_ROOT`: manifest and adjacency-audit output directory;
- `GAUSSIAN_SPLATTING_ROOT`: official 3DGS checkout;
- `HERITAGE_PYTHON`: Python executable used for 3DGS commands;
- `HERITAGE_NERFACTO_DATA`: prepared Nerfstudio datasets;
- `HERITAGE_NERFACTO_RUNS`: main Nerfacto output root;
- `HERITAGE_NERFACTO_POSITION_RUNS`: alternative-position Nerfacto output root;
- `NERFSTUDIO_PYTHON`, `NS_TRAIN`, and `NS_EVAL`: optional executable overrides;
- `CUDA_DEVICES`: comma-separated device indices;
- `COUNT_MATCH_GPU`, `COUNT_MATCH_SHARD`, and `COUNT_MATCH_N_SHARDS`: count-matched runner allocation.

Example:

```bash
export HERITAGE_DATA_ROOT=/path/to/authorised/data
export HERITAGE_OUTPUT_ROOT=/path/to/results
python scripts/protocols/export_blocked_split_manifest.py
python scripts/protocols/audit_split_leakage.py
```

No host address, account name, credential, notification endpoint, or machine-specific project path is required by the repository.

The fixed Nerfacto launch command and non-default settings are recorded in `docs/NERFACTO_CONFIGURATION.md`.
