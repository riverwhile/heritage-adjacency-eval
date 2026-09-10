# Reproducibility scope

## Included evidence

The repository includes the complete blocked split manifest, per-object metric rows, protocol summaries, count-matched sensitivity replicates, adjacency summaries, and compact Nerfacto evaluation JSON files. Bootstrap confidence intervals in the manuscript use 20,000 resamples of objects; frames are never treated as independent inferential units.

## Protocol definitions

The interleaved protocol holds out approximately every eighth ordered image. Buffered blocked protocols hold out three contiguous sectors of lengths 4, 4, and 2, with a two-image exclusion buffer. Normalised anchors are 0.20/0.50/0.80 for blocked-v1, 0.12/0.42/0.72 for blocked-v2, and 0.28/0.58/0.88 for blocked-v3.

The training-count-matched control preserves the interleaved test set and its immediate temporal neighbours, then deterministically removes other training images to match blocked-v1 training counts. Seeds 2024--2026 are averaged within each object-condition before object-level inference.

## Not included

- cultural-object RGB images;
- camera reconstructions and collection metadata requiring permission;
- trained Gaussian or Nerfacto checkpoints;
- rendered evaluation images;
- package caches and third-party source trees;
- SSH credentials, notification credentials, host addresses, and operational logs.

The omission of controlled image data means that numerical aggregation can be audited from this repository, while complete model retraining requires separately authorised data access.

## Configurable paths

The protocol scripts use environment variables instead of machine-specific paths:

- `HERITAGE_DATA_ROOT`: object/condition data root containing the registered images, COLMAP reconstruction, and split directories;
- `HERITAGE_OUTPUT_ROOT`: output directory for manifests and adjacency audits;
- `GAUSSIAN_SPLATTING_ROOT`: checkout of the official 3DGS implementation;
- `HERITAGE_PYTHON`: Python executable used for 3DGS commands;
- `HERITAGE_NERFACTO_DATA`: prepared Nerfstudio dataset root;
- `HERITAGE_NERFACTO_RUNS`: output root for the main Nerfacto matrix;
- `HERITAGE_NERFACTO_POSITION_RUNS`: output root for the blocked-v2/v3 subset;
- `NERFSTUDIO_PYTHON`, `NS_TRAIN`, and `NS_EVAL`: optional executable overrides;
- `CUDA_DEVICES`: comma-separated GPU indices, for example `0,1`.

Example:

```bash
export HERITAGE_DATA_ROOT=/path/to/authorised/data
export HERITAGE_OUTPUT_ROOT=/path/to/results
python scripts/protocols/export_blocked_split_manifest.py
python scripts/protocols/audit_split_leakage.py
```
