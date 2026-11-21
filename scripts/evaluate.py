"""Evaluate one completed run on the held-out test partition.

TensorFlow is imported only after arguments are parsed and
``--confirm-final-test`` is present. A valid saved evaluation is reused.
"""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=(
            "Evaluate a completed final checkpoint once on its test partition. "
            "Official runs load the cached CIFAR-100 test set only after "
            "--confirm-final-test. Existing valid results are reused. "
            "This command does not train or edit the paper."
        ),
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Completed run directory, for example runs/<run-id>.",
    )
    parser.add_argument(
        "--cache-root",
        default="cache",
        help="Root of the shared preprocessing cache. Default: cache/.",
    )
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Authorize one final test evaluation. Required before test arrays are read.",
    )
    parser.add_argument(
        "--repeat-inference-reason",
        default=None,
        help=(
            "Recorded reason for repeating inference when saved predictions cannot "
            "be recovered. A completed evaluation is not rerun."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.confirm_final_test:
        parser.error("refusing to evaluate without --confirm-final-test")
    from cifar100_study.data import DataError
    from cifar100_study.evaluation import EvaluationError, evaluate_run
    from cifar100_study import contracts

    try:
        summary = evaluate_run(
            Path(args.run),
            confirm_final_test=contracts.FINAL_TEST_GUARD,
            cache_root=Path(args.cache_root),
            repeat_inference_reason=args.repeat_inference_reason,
        )
    except (EvaluationError, DataError) as exc:
        parser.exit(1, f"evaluate.py: error: {exc}\n")
    print(f"run: {summary['run_id']}")
    print(f"experiment: {summary['experiment']}")
    print(f"reused: {summary['reused']}")
    print(f"test_count: {summary['test_count']}")
    print(f"correct_predictions: {summary['correct_predictions']}")
    print(f"top1_accuracy: {summary['top1_accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
