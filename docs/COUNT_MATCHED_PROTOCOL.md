# Training-count-matched interleaved control

Status: protocol fixed before formal results were inspected.

Purpose: separate the effect of train--test adjacency from the reduction in
training-image count caused by the buffered blocked-v1 protocol.

For every object-condition orbit and each fixed seed (2024, 2025, 2026):

1. Keep the original ordered every-eighth test set unchanged (`images[::8]`).
2. Set the target training count equal to that orbit's blocked-v1 training count.
3. Protect the immediate ordered predecessor and successor of every test image.
4. Sample the required number of removals from the remaining non-test,
   non-protected images using Python `random.Random(seed).sample`.
5. Train official 3DGS for 7,000 iterations with the same configuration as the
   formal interleaved and blocked-v1 baselines.
6. Evaluate the unchanged interleaved test images with PSNR, SSIM and LPIPS.

Primary analysis: average each object-condition score across the three fixed
removal seeds, form OUT--IN within object, and compare this contrast with the
original interleaved and blocked-v1 contrasts. Seed-level dispersion is a
sensitivity diagnostic, not an independent inferential sample.

The design deliberately protects immediate neighbours. It is therefore a
training-count control, not another adjacency-controlled split.
