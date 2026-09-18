# Heritage adjacency evaluation

Code and non-image reproducibility materials for the manuscript **Adjacent frame evaluation can overstate capture condition effects in handheld cultural heritage reconstruction**.

The repository supports four parts of the reported evaluation: construction of interleaved and buffered blocked splits, train--test adjacency auditing, object-level aggregation and uncertainty analysis, and comparison of 3D Gaussian Splatting with Nerfacto across evaluation protocols.

## Repository contents

- `scripts/protocols/`: split construction, adjacency auditing, training launchers, and raw metric aggregation for authorised data holders.
- `scripts/analysis/`: analyses that run directly on the public numerical evidence.
- `data/manifests/`: train, test, and exclusion-buffer assignments for all 14 acquisition orbits and three blocked protocols.
- `data/derived/`: anonymised object-level and protocol-level values used in the manuscript.
- `docs/`: reproducibility scope, protocol details, subset provenance, and the sanitised Nerfacto configuration record.

## Public verification

Create an isolated Python environment, install the listed dependencies, and run:

```bash
python -m pip install -r requirements.txt
python scripts/analysis/validate_public_release.py
python scripts/analysis/analyse_protocol_attenuation.py --output-dir artifacts
python scripts/analysis/analyse_count_matched_control.py --output-dir artifacts
python scripts/analysis/aggregate_nerfacto_v2v3.py --output-dir artifacts
python scripts/analysis/build_blocked_protocol_results.py --output-dir artifacts
```

The validation command checks the published matrices, split manifest, source-video metadata table, and the exact sign-flip values reported in Supplementary Table S11. The remaining commands regenerate public analysis tables and protocol figures under `artifacts/`. The output directory is ignored by Git.

## Data boundary

The cultural-object images, COLMAP camera files, rendered images, and trained checkpoints are not included because release requires permission from the holding collection. Consequently, the public repository supports numerical and split-level audit but not model retraining. Scripts requiring restricted inputs use environment variables rather than machine-specific paths; see `docs/REPRODUCIBILITY.md`.

## Recorded environments

- 3DGS: official implementation at commit `54c035f`; PyTorch 2.1.2; CUDA 12.1; 7,000 iterations; seed 0.
- Nerfacto: Nerfstudio 1.1.5; PyTorch 2.1.0; CUDA 11.8; 10,000 steps; seed 2024; camera optimisation disabled.
- Public analysis: Python 3.10 or newer with `requirements.txt`.

The three count-matched seeds vary training-image deletion, not model initialisation. They must not be interpreted as optimisation-seed replications.

## Citation

Citation metadata will be added after a manuscript record or archived release is available. Until then, cite this repository URL and the manuscript title.

## Licence

Code is released under the MIT License. Numerical outputs are provided for research verification; the underlying cultural-object imagery remains subject to collection permission.
