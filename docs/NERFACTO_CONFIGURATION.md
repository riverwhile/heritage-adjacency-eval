# Nerfacto configuration record

The main and alternative-position Nerfacto matrices used Nerfstudio 1.1.5, PyTorch 2.1.0, CUDA 11.8, and the `nerfacto` method. Each model used 10,000 optimisation steps and seed 2024. Camera optimisation was disabled.

The launch command constructed by `scripts/protocols/run_nerfacto_formal10k.py` is equivalent to:

```bash
ns-train nerfacto \
  --data DATASET_DIRECTORY \
  --output-dir OUTPUT_DIRECTORY \
  --experiment-name OBJECT_CONDITION_PROTOCOL \
  --timestamp seed2024_10k \
  --machine.seed 2024 \
  --max-num-iterations 10000 \
  --steps-per-save 2000 \
  --steps-per-eval-image 1000 \
  --steps-per-eval-all-images 10000 \
  --vis tensorboard \
  --viewer.quit-on-train-completion True \
  --pipeline.model.camera-optimizer.mode off \
  nerfstudio-data --eval-mode filename
```

The saved configuration recorded 4,096 training and evaluation rays per batch, two proposal iterations, an appearance embedding dimension of 32, mixed precision, the `tcnn` implementation, 48 NeRF samples per ray, proposal sample counts of 256 and 96, 16 hash-grid levels, base resolution 16, maximum resolution 2,048, hash-map size 19, and evaluation chunks of 32,768 rays. Parameters not overridden by the command retained the Nerfstudio 1.1.5 `nerfacto` defaults.

Dataset and output locations are deliberately represented by placeholders. Machine paths, host information, and checkpoint locations are not part of the public configuration record.
