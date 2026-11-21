# CIFAR-100 All-CNN-C Study

This project investigates how extending the training schedule and increasing convolutional depth affect image classification accuracy on CIFAR-100, using architectures based on All-CNN-C.

Research questions: Does doubling the training schedule improve classification accuracy? Does doubling convolutional depth produce similar improvements? Which approach offers greater accuracy gains relative to its computational cost?

Hypothesis: Both modifications may improve classification accuracy, but their effects are expected to differ. Increasing depth may introduce optimization difficulties and greater computational costs, potentially limiting its advantages over extended training.

The study compares three experiments:

| Experiment | Architecture | Training | LR decay milestones |
|---|---|---|---|
| E0 | All-CNN-C, 9 convolutions | 350 epochs | 200, 250, 300 |
| E1 | All-CNN-C, 9 convolutions | 700 epochs | 400, 500, 600 |
| E2 | Extended All-CNN-C, 18 convolutions | 350 epochs | 200, 250, 300 |

All experiments use the official CIFAR-100 training partition, shared train-fitted GCN/ZCA preprocessing, and fixed training settings.

The research methodology and results are documented in [`reports/main.tex`](reports/main.tex). The experiment configurations are in [`configs/`](configs/).

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The lockfile pins TensorFlow 2.18.1. On Apple Silicon it also installs `tensorflow-metal` 1.2.0, which provides the Metal GPU. Other platforms keep the CPU build. Confirm the GPU after syncing:

```bash
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

A Metal machine should list `GPU:0`. Training accepts `--device gpu` on that machine.

## Usage

**Prepare the dataset**

```bash
uv run python scripts/prepare.py --config configs/e0.yaml
```

Prepares a versioned CIFAR-100 cache using pixel scaling, global contrast normalization, and ZCA whitening fitted on the training partition. E0, E1, and E2 share the same preprocessing cache.

**Train an experiment**

```bash
uv run python scripts/train.py --config configs/e0.yaml --device cpu
```

Replace `e0.yaml` with `e1.yaml` or `e2.yaml` for the other experiments.

To resume an interrupted run:

```bash
uv run python scripts/train.py --resume runs/<run-id> --device cpu
```

The device option accepts `auto`, `cpu`, or `gpu`. Requesting an unavailable GPU fails explicitly.

Each experiment starts in a new run directory and saves its configuration, metadata, training history, checkpoints, and final model.

**Evaluate a completed run**

```bash
uv run python scripts/evaluate.py \
  --run runs/<run-id> \
  --confirm-final-test
```

Evaluation uses the official CIFAR-100 held-out test partition and requires explicit confirmation. A valid existing evaluation is reused rather than recomputed.

**Generate the comparison report**

```bash
uv run python scripts/report.py \
  --runs runs/<e0-id> runs/<e1-id> runs/<e2-id>
```

Generates training curves and comparison artifacts in `outputs/report/` using completed, compatible official experiments. It does not edit the LaTeX research report.

## Tests

```bash
uv run pytest
```

The test suite uses synthetic fixtures and does not evaluate official CIFAR-100 test data.

## Repository structure

- `configs/`: locked E0, E1, and E2 configurations
- `src/cifar100_study/`: preprocessing, models, training, evaluation, and reporting
- `scripts/`: command-line entry points
- `tests/`: automated tests
- `reports/main.tex`: research report
- `AGENTS.md`: engineering and scientific constraints

Datasets, fitted caches, experiment runs, checkpoints, generated reports, and LaTeX build artifacts are kept out of version control.