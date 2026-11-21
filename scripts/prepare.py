"""Prepare the shared CIFAR-100 cache. Imports Keras only when preparation runs."""

import argparse
from pathlib import Path

from cifar100_study.config import ConfigError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare.py",
        description=(
            "Prepare CIFAR-100 for the locked study. Pixels are scaled to [0, 1], "
            "each image is global-contrast normalized, full ZCA is fit on the "
            "training images only, and the 100 fine labels are one-hot encoded. "
            "E0, E1, and E2 share the cache."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/e0.yaml, configs/e1.yaml, or configs/e2.yaml.",
    )
    parser.add_argument(
        "--cache-root",
        default="cache",
        help="Directory for the shared preprocessing cache. Default: cache/",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from cifar100_study.data import DataError, prepare_command
    from cifar100_study.preprocess import PreprocessError

    try:
        result = prepare_command(Path(args.config), Path(args.cache_root))
    except (ConfigError, DataError, PreprocessError) as exc:
        parser.exit(1, f"prepare.py: error: {exc}\n")
    print(f"cache: {result.path}")
    print(f"reused: {'yes' if result.reused else 'no'}")
    print(f"train_images: {result.train_images_shape} float32")
    print(f"train_labels: {result.train_labels_shape} float32")
    print(f"test_images: {result.test_images_shape} float32")
    print(f"test_labels: {result.test_labels_shape} float32")
    print("gcn_min_divisor: 1e-8")
    print("zca_regularization: 0.1")
    print("covariance_divisor: N")
    print("fit_on: train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
