"""Module entrypoint. Lists the milestone scripts and does not train."""

import argparse

from cifar100_study import __version__

COMMANDS = """\
python scripts/prepare.py --help    # M0.1 CIFAR-100 preparation
python scripts/train.py --help       # M0.3 training
python scripts/evaluate.py --help    # M0.4 guarded final test
python scripts/report.py --help      # M0.4 figures and tables from artifacts
uv run pytest
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cifar100_study",
        description=(
            "CIFAR-100 All-CNN-C study harness (M0.0 foundation). "
            "This entrypoint does not load CIFAR-100 or train a model."
        ),
        epilog="Commands:\n" + COMMANDS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"cifar100_study {__version__}",
    )
    parser.parse_args(argv)
    print(COMMANDS, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
