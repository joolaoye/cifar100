"""Write the E0/E1/E2 comparison and training curves from saved artifacts.

The command does not load the test set, does not run inference, and does not
edit reports/main.tex.
"""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="report.py",
        description=(
            "Write training curves and an E0/E1/E2 comparison from completed "
            "run artifacts. This command does not train, evaluate, or edit the paper."
        ),
    )
    parser.add_argument(
        "--runs",
        nargs=3,
        required=True,
        metavar="RUN",
        help="Completed E0, E1, and E2 run directories, in any order.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/report",
        help="Directory for the table and curve figures. Default: outputs/report/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from cifar100_study.evaluation import EvaluationError
    from cifar100_study.reporting import ReportingError, write_report
    from cifar100_study.runs import TrainingError

    try:
        result = write_report([Path(run) for run in args.runs], Path(args.output_dir))
    except (ReportingError, EvaluationError, TrainingError) as exc:
        parser.exit(1, f"report.py: error: {exc}\n")
    print(f"output_dir: {result['output_dir']}")
    for row in result["rows"]:
        print(
            f"{row['experiment']}: top1_accuracy={row['top1_accuracy']} "
            f"difference_vs_e0_pp={row['accuracy_difference_vs_e0_percentage_points']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
