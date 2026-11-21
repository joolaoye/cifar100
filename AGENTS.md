# AGENTS.md

## Project

This project examines how training-schedule duration and convolutional depth affect CIFAR-100 image classification using All-CNN-C.

Keep the implementation focused, reproducible, and scientifically sound. Avoid unnecessary infrastructure, speculative abstractions, and changes unrelated to the research objectives.

## Sources of truth

- `configs/e0.yaml`, `configs/e1.yaml`, and `configs/e2.yaml` define the locked experimental configurations.
- `reports/main.tex` documents the research questions, methodology, limitations, and results.
- `src/cifar100_study/` contains the implementation.
- `tests/` verifies the scientific and operational invariants.

If the configurations, implementation, and report disagree, surface the inconsistency rather than silently changing the protocol.

## Experimental invariants

- Use the official CIFAR-100 dataset: 50,000 training images, 10,000 held-out test images, and 100 fine-label classes.
- Train on all 50,000 training images. No validation split, augmentation, pretrained weights, early stopping, or test-based model selection.
- Preprocess using pixel scaling, per-image GCN, and train-fitted ZCA whitening. Never fit dataset-level statistics on the test set.
- E0: nine-convolution All-CNN-C, 350 epochs, LR milestones 200/250/300.
- E1: the same architecture, independently trained from scratch for 700 epochs, LR milestones 400/500/600.
- E2: eighteen-convolution extension, 350 epochs, LR milestones 200/250/300.
- Shared settings: seed 42, batch size 256, SGD momentum 0.9, initial LR 0.05, decay factor 0.1, and categorical cross-entropy with one-hot labels.
- Preserve the original architecture's activation behavior, including ReLU on the final class convolution.
- Preserve coupled L2 weight decay: `keras.regularizers.L2(0.0005)` on convolution kernels gives the intended gradient coefficient of 0.001. Do not add optimizer weight decay.
- Use the final scheduled epoch, not the best intermediate checkpoint.
- Evaluate completed final models on the official test set only through the explicit, guarded evaluation command.
- E1 changes both total epochs and LR timing; E2 changes depth and computational cost. Do not describe these as isolated changes to epoch count or compute-matched comparisons.
- Results are single-seed observations, not statistical significance claims.

Changes to the scientific protocol require explicit user approval.

## Run and data integrity

- Preserve configuration, Git, dependency, dataset, cache, and device provenance.
- Keep preprocessing caches, runs, checkpoints, and generated outputs out of Git.
- Never overwrite or silently delete an existing experiment run.
- Preserve validated checkpoints, optimizer state, history, and learning-rate progression across interrupted-run recovery.
- A diverged or failed run must not publish a final model.
- Preserve the one-time evaluation safeguards and committed-result integrity checks.
- Only compare completed, compatible official E0/E1/E2 runs.
- Never use official test data in synthetic tests, diagnostics, or performance benchmarks.

## Engineering workflow

- Use Python 3.12, `uv`, and the committed dependency lockfile.
- Run relevant tests after changes; run `uv run pytest` before accepting substantive changes.
- Keep changes focused. Avoid speculative abstractions, generic frameworks, and unnecessary refactoring.
- Independent reviews should prioritize scientific correctness, data leakage, reproducibility, and concrete defects.
- Do not fabricate metrics, edit reported results without evidence, or automatically rewrite conclusions.
- Do not commit, push, change scientific configurations, launch production training, or evaluate official test data without explicit user authorization.

For operational commands, see `README.md`.