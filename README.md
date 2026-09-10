# Heritage adjacency evaluation

Code and non-image reproducibility materials for the manuscript **Adjacent frame evaluation can overstate capture condition effects in handheld cultural heritage reconstruction**.

The repository supports the paper's central evaluation claims: construction of interleaved and buffered blocked splits, train--test adjacency auditing, training-count-matched controls, object-level aggregation and bootstrap inference, and the 3DGS/Nerfacto protocol comparisons.

## Repository contents

- `scripts/protocols/`: split generation, adjacency audit, 3DGS count-matched runs, Nerfacto data preparation/runs, and metric aggregation.
- `scripts/analysis/`: object-level statistical aggregation and figure/table preparation.
- `data/manifests/`: machine-readable blocked-v1/v2/v3 train/test/buffer assignments for all 14 acquisition orbits.
- `data/derived/`: anonymised object-level and protocol-level metric tables used by the manuscript.
- `docs/`: protocol notes and reproducibility boundaries.

## Data availability boundary

The cultural-object images, COLMAP camera files, rendered images, and model checkpoints are not included because release requires permission from the holding collection. The public materials contain object identifiers, split definitions, and numerical results sufficient to audit the reported aggregation. They do not support retraining without separately authorised image and camera data.

## Software environments

- 3DGS: official implementation, commit `54c035f`; PyTorch 2.1.2; CUDA 12.1; 7,000 iterations; seed 0.
- Nerfacto: Nerfstudio 1.1.5; PyTorch 2.1.0; CUDA 11.8; 10,000 steps; seed 2024.
- Statistical scripts: Python 3.10+ with the packages in `requirements.txt`.

All filesystem locations are supplied through environment variables; the repository contains no machine-specific paths. See `docs/REPRODUCIBILITY.md` for the variable names and workflow. The analysis scripts operate on the compact CSV/JSON evidence in `data/` and are the primary audit surface.

## Quick checks

```bash
python -m compileall scripts
python -m json.tool data/manifests/blocked_split_manifest.json > /dev/null
python scripts/analysis/summarize_global_nearest_camera.py --help
```

## Citation

Citation metadata will be added after the manuscript record is available. Until then, please cite the repository URL and manuscript title.

## Licence

Code is released under the MIT License. Numerical outputs are provided for research verification; underlying cultural-object imagery remains subject to collection permission.
