"""Train one locked All-CNN-C experiment. Imports TensorFlow only after parsing."""

import argparse
from pathlib import Path

from cifar100_study.config import ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=(
            "Train E0, E1, or E2 from scratch, or resume an interrupted run. "
            "The final model is the last scheduled epoch. E1 does not continue E0. "
            "This command does not evaluate the test set."
        ),
    )
    parser.add_argument(
        "--config",
        help="Path to configs/e0.yaml, configs/e1.yaml, or configs/e2.yaml.",
    )
    parser.add_argument(
        "--resume",
        help="Existing run directory to resume, for example runs/<run-id>.",
    )
    parser.add_argument(
        "--run-root",
        default="runs",
        help="Directory for new run folders. Default: runs/",
    )
    parser.add_argument(
        "--cache-root",
        default="cache",
        help="Root of the shared preprocessing cache. Default: cache/",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Compute device. gpu fails if no GPU is available. Default: auto.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.config and not args.resume:
        parser.error("provide --config to start a run or --resume to continue one")
    from cifar100_study.data import DataError
    from cifar100_study.trainer import TrainingError, TrainingInterrupted, train_command

    try:
        result = train_command(
            config_path=Path(args.config) if args.config else None,
            resume=Path(args.resume) if args.resume else None,
            run_root=Path(args.run_root),
            cache_root=Path(args.cache_root),
            device_name=args.device,
            git_root=Path.cwd(),
        )
    except TrainingInterrupted as exc:
        parser.exit(1, f"train.py: interrupted: {exc}\n")
    except (ConfigError, DataError, TrainingError) as exc:
        parser.exit(1, f"train.py: error: {exc}\n")
    print(f"run: {result['run_dir']}")
    print(f"experiment: {result['experiment']}")
    print(f"status: {result['status']}")
    print(f"device: {result['device']} {result['device_name']}")
    print(f"completed_epochs: {result['completed_epochs']}")
    print(f"epoch_seconds_total: {result['epoch_seconds_total']}")
    print(f"cache: {result['cache_key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
