"""Training curves and the E0/E1/E2 comparison, built only from saved artifacts.

This module does not load test images and does not run a model. Convolution
MACs are multiply-accumulates per image. Completed-epoch time is the recorded
sum of epoch durations, not wall-clock time and not a FLOP count.
"""

import csv
from pathlib import Path

from cifar100_study import contracts
from cifar100_study.allcnn import REGISTERED_COMPLEXITY
from cifar100_study.evaluation import (
    EvaluationError,
    load_evaluation_summary,
    validate_completed_run,
)
from cifar100_study.runs import read_metadata

COMPARISON_COLUMNS = (
    "experiment",
    "convolutional_layers",
    "epochs",
    "top1_accuracy",
    "accuracy_difference_vs_e0_percentage_points",
    "trainable_parameters",
    "macs_per_image",
    "epoch_seconds_total",
    "device",
)
_EXPERIMENTS = ("e0", "e1", "e2")
_UNIT_NOTE = (
    "Convolution MACs are multiply-accumulates per image for the forward "
    "convolution layers. They are not training FLOPs. Completed-epoch time is "
    "the sum of recorded epoch durations in seconds, excluding startup, "
    "preprocessing, and time between sessions."
)


class ReportingError(ValueError):
    """Saved runs cannot be compared or plotted."""


def training_series(run_dir: Path) -> dict:
    """Return the recorded training loss and accuracy for a completed run."""
    completed = validate_completed_run(Path(run_dir))
    metadata = completed["metadata"]
    _require_reportable(metadata)
    rows = completed["history"]
    return {
        "experiment": metadata["experiment"],
        "epoch": [row["epoch"] for row in rows],
        "loss": [row["loss"] for row in rows],
        "accuracy": [row["accuracy"] for row in rows],
    }


def write_training_curves(run_dir: Path, output_dir: Path) -> dict:
    """Write one loss figure and one accuracy figure from history.csv."""
    series = training_series(run_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    experiment = series["experiment"]
    _plot_series(
        destination / f"{experiment}-training-loss.png",
        series["epoch"],
        series["loss"],
        ylabel="Training loss",
        title=f"{experiment.upper()} training loss",
    )
    _plot_series(
        destination / f"{experiment}-training-accuracy.png",
        series["epoch"],
        series["accuracy"],
        ylabel="Training categorical accuracy",
        title=f"{experiment.upper()} training accuracy",
    )
    return series


def comparison_rows(run_dirs: list[Path], *, allow_synthetic: bool = False) -> list[dict]:
    """Build one row per experiment from one controlled E0, E1, and E2 comparison.

    ``allow_synthetic`` exists for fixture tables. Production reporting leaves
    it false and accepts only official runs that share one dataset.
    """
    if len(run_dirs) != 3:
        raise ReportingError("the comparison needs the completed E0, E1, and E2 runs")
    completed_by_experiment = {}
    for run_dir in run_dirs:
        metadata = read_metadata(Path(run_dir))
        experiment = metadata.get("experiment")
        if experiment not in _EXPERIMENTS:
            raise ReportingError(f"comparison does not include experiment {experiment!r}")
        if experiment in completed_by_experiment:
            raise ReportingError(f"comparison received more than one {experiment} run")
        completed = validate_completed_run(Path(run_dir))
        metadata = completed["metadata"]
        _require_reportable(metadata)
        completed_by_experiment[experiment] = (Path(run_dir), completed)
    _require_controlled_comparison(
        [completed_by_experiment[experiment][1]["metadata"] for experiment in _EXPERIMENTS],
        allow_synthetic=allow_synthetic,
    )
    rows_by_experiment = {}
    for experiment in _EXPERIMENTS:
        run_dir, completed = completed_by_experiment[experiment]
        history = completed["history"]
        status = completed["status"]
        try:
            summary = load_evaluation_summary(run_dir)
        except EvaluationError as exc:
            raise ReportingError(f"{experiment} evaluation is missing or inconsistent: {exc}") from exc
        if int(summary["epochs"]) != len(history):
            raise ReportingError(f"{experiment} evaluation does not cover the completed history")
        if abs(float(summary["epoch_seconds_total"]) - float(status["epoch_seconds_total"])) > 1e-6:
            raise ReportingError(f"{experiment} evaluation time does not match the run")
        rows_by_experiment[experiment] = _row(summary)
    baseline = rows_by_experiment["e0"]["top1_accuracy"]
    ordered = []
    for experiment in _EXPERIMENTS:
        row = dict(rows_by_experiment[experiment])
        row["accuracy_difference_vs_e0_percentage_points"] = (
            row["top1_accuracy"] - baseline
        ) * 100.0
        ordered.append(row)
    return ordered


def write_report(run_dirs: list[Path], output_dir: Path, *, allow_synthetic: bool = False) -> dict:
    """Write the comparison table and training curves. Does not edit the paper."""
    rows = comparison_rows(run_dirs, allow_synthetic=allow_synthetic)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "comparison.csv", rows)
    _write_markdown(destination / "comparison.md", rows)
    _write_latex(destination / "comparison.tex", rows)
    curves = [write_training_curves(Path(run_dir), destination) for run_dir in run_dirs]
    return {"rows": rows, "curves": curves, "output_dir": str(destination)}


def _require_controlled_comparison(metadata_rows: list[dict], *, allow_synthetic: bool) -> None:
    sources = {row.get("data_source") for row in metadata_rows}
    if sources == {"synthetic", "official"}:
        raise ReportingError("comparison mixes synthetic and official runs")
    if sources != {"official"}:
        if not (allow_synthetic and sources == {"synthetic"}):
            raise ReportingError("comparison requires three official runs from the same dataset")
    _require_same(metadata_rows, "cache_key", "preprocessing cache key")
    _require_same(metadata_rows, "source_fingerprint", "preprocessing source fingerprint")
    _require_same(metadata_rows, "seed", "training seed")
    _require_same(metadata_rows, "batch_size", "batch size")
    _require_same(metadata_rows, "study", "study id")
    learning_rates = [_schedule_field(row, "initial_learning_rate") for row in metadata_rows]
    decays = [_schedule_field(row, "decay_factor") for row in metadata_rows]
    if len(set(learning_rates)) != 1:
        raise ReportingError("comparison runs do not share an initial learning rate")
    if len(set(decays)) != 1:
        raise ReportingError("comparison runs do not share a learning-rate decay factor")
    if sources != {"official"}:
        return
    if any(row.get("cache_key") in (None, "") or row.get("source_fingerprint") in (None, "") for row in metadata_rows):
        raise ReportingError("official comparison is missing a shared preprocessing identity")
    for row in metadata_rows:
        experiment = row["experiment"]
        spec = contracts.EXPERIMENT_SPECS[experiment]
        schedule = row["schedule"]
        milestones = tuple(int(item) for item in schedule.get("milestones") or ())
        if int(schedule["epochs"]) != spec.epochs or milestones != spec.lr_milestones:
            raise ReportingError(f"{experiment} schedule is not the approved configuration")
        expected_depth, expected_parameters, _expected_macs = REGISTERED_COMPLEXITY[experiment]
        if (
            row.get("convolutional_layers") != expected_depth
            or row.get("trainable_parameters") != expected_parameters
        ):
            raise ReportingError(f"{experiment} architecture is not the approved configuration")


def _require_same(metadata_rows: list[dict], field: str, label: str) -> None:
    values = [row.get(field) for row in metadata_rows]
    if len(set(values)) != 1:
        raise ReportingError(f"comparison runs do not share a {label}")


def _schedule_field(metadata: dict, field: str):
    schedule = metadata.get("schedule")
    if type(schedule) is not dict or field not in schedule:
        raise ReportingError(f"comparison run {metadata.get('experiment')} has no {field}")
    return schedule[field]


def _require_reportable(metadata: dict) -> None:
    experiment = metadata.get("experiment")
    spec = contracts.EXPERIMENT_SPECS.get(experiment)
    if spec is None:
        raise ReportingError(f"unknown experiment {experiment!r}")
    schedule = metadata.get("schedule")
    if type(schedule) is not dict or type(schedule.get("epochs")) is not int:
        raise ReportingError(f"{experiment} has no epoch budget")
    if metadata.get("data_source") == "official" and schedule["epochs"] != spec.epochs:
        raise ReportingError(f"{experiment} history budget is not the registered schedule")


def _row(summary: dict) -> dict:
    device = summary.get("device")
    if type(device) is not dict or type(device.get("selected_type")) is not str:
        raise ReportingError("evaluation is missing the device used for training")
    name = device.get("selected_name")
    label = device["selected_type"] if type(name) is not str or not name else f"{device['selected_type']} {name}"
    try:
        macs = int(summary["macs_per_image"])
        parameters = int(summary["trainable_parameters"])
        depth = int(summary["convolutional_layers"])
        epochs = int(summary["epochs"])
        accuracy = float(summary["top1_accuracy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportingError("evaluation is missing a comparison field") from exc
    if summary.get("mac_unit") != "multiply-accumulates per image":
        raise ReportingError("evaluation MAC unit is not multiply-accumulates per image")
    return {
        "experiment": summary["experiment"],
        "convolutional_layers": depth,
        "epochs": epochs,
        "top1_accuracy": accuracy,
        "trainable_parameters": parameters,
        "macs_per_image": macs,
        "epoch_seconds_total": float(summary["epoch_seconds_total"]),
        "device": label,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row, column) for column in COMPARISON_COLUMNS})


def _csv_value(row: dict, column: str):
    if column == "accuracy_difference_vs_e0_percentage_points":
        return format(row[column], ".10g")
    if column in {"top1_accuracy", "epoch_seconds_total"}:
        return format(row[column], ".10g")
    return row[column]


def _write_markdown(path: Path, rows: list[dict]) -> None:
    header = (
        "| Experiment | Convolutional layers | Training epochs | Top-1 test accuracy | "
        "Difference vs E0 (percentage points) | Parameters | Convolution MACs per image | "
        "Completed-epoch training time (s) | Device |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
    )
    body = "".join(_markdown_row(row) for row in rows)
    path.write_text(header + body + "\n" + _UNIT_NOTE + "\n", encoding="utf-8")


def _markdown_row(row: dict) -> str:
    values = [
        row["experiment"],
        str(row["convolutional_layers"]),
        str(row["epochs"]),
        format(row["top1_accuracy"], ".6f"),
        format(row["accuracy_difference_vs_e0_percentage_points"], ".6f"),
        str(row["trainable_parameters"]),
        str(row["macs_per_image"]),
        format(row["epoch_seconds_total"], ".6f"),
        row["device"],
    ]
    return "| " + " | ".join(values) + " |\n"


def _write_latex(path: Path, rows: list[dict]) -> None:
    lines = [
        "% Comparison fragment. Convolution MACs are multiply-accumulates per image, not FLOPs.",
        "% Completed-epoch time is seconds of recorded epoch durations, not wall-clock time.",
        "\\begin{tabular}{lrrrrrrll}",
        "\\hline",
        "Experiment & Layers & Epochs & Top-1 & Diff.\\ vs E0 (pp) & Parameters & MACs/image & Time (s) & Device \\\\",
        "\\hline",
    ]
    for row in rows:
        cells = [
            row["experiment"],
            str(row["convolutional_layers"]),
            str(row["epochs"]),
            format(row["top1_accuracy"], ".6f"),
            format(row["accuracy_difference_vs_e0_percentage_points"], ".6f"),
            str(row["trainable_parameters"]),
            str(row["macs_per_image"]),
            format(row["epoch_seconds_total"], ".6f"),
            _latex_escape(row["device"]),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _latex_escape(value: str) -> str:
    escaped = value
    for character in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        escaped = escaped.replace(character, "\\" + character)
    return escaped


def _plot_series(path: Path, epochs: list[int], values: list[float], *, ylabel: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 3.6))
    axis.plot(epochs, values, color="black", linewidth=1.0)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, linewidth=0.3, alpha=0.4)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
