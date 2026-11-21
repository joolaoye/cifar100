"""Guarded final-test evaluation for one completed run.

The official test arrays are loaded only after ``--confirm-final-test``.
A valid ``evaluation/`` directory is reused. Partial or inconsistent files are
rejected and are not overwritten. Evaluation does not train, does not edit
``history.csv``, and does not select a checkpoint.
"""

import csv
import hashlib
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cifar100_study import contracts
from cifar100_study.allcnn import REGISTERED_COMPLEXITY
from cifar100_study.config import ConfigError, load_config
from cifar100_study.runs import read_metadata, read_status

EVALUATION_DIRNAME = "evaluation"
PREDICTIONS_NAME = "predictions.csv"
CONFUSION_NAME = "confusion.csv"
EXAMPLES_NAME = "examples.json"
EXAMPLES_FIGURE_NAME = "examples.png"
SUMMARY_NAME = "evaluation.json"
ATTEMPT_NAME = "attempt.json"
PREDICTION_MANIFEST_NAME = "prediction_manifest.json"
EXAMPLE_LIMIT = 4
PREDICTION_COLUMNS = ("index", "true_label", "predicted_label", "confidence")
MAC_UNIT = "multiply-accumulates per image"
EXAMPLE_RULE = (
    "first four correct predictions and first four incorrect predictions, "
    "in official test-set order"
)

# Official CIFAR-100 fine-label order from the dataset meta file.
CIFAR100_FINE_LABELS = (
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
)

_SUMMARY_KEYS = (
    "cache_key",
    "config_sha256",
    "confusion_sha256",
    "convolutional_layers",
    "correct_predictions",
    "device",
    "epoch_seconds_total",
    "epochs",
    "examples_json_sha256",
    "examples_png_sha256",
    "experiment",
    "format_version",
    "mac_unit",
    "macs_per_image",
    "model_sha256",
    "output_classes",
    "predictions_sha256",
    "run_id",
    "source_fingerprint",
    "test_count",
    "top1_accuracy",
    "trainable_parameters",
)


class EvaluationError(ValueError):
    """Final evaluation cannot be published or reused safely."""


def class_name(label: int, num_classes: int) -> str:
    """Return the CIFAR-100 fine name, or a synthetic class id."""
    if num_classes == contracts.NUM_CLASSES:
        if label < 0 or label >= len(CIFAR100_FINE_LABELS):
            raise EvaluationError(f"fine label {label} is outside 0..99")
        return CIFAR100_FINE_LABELS[label]
    return f"class {label}"


def score_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Score one-hot labels with argmax Top-1. Each example has equal weight."""
    if probabilities.ndim != 2 or labels.shape != probabilities.shape:
        raise EvaluationError(
            "probabilities and one-hot labels must share shape "
            f"(N, classes); got {probabilities.shape} and {getattr(labels, 'shape', None)}"
        )
    if probabilities.shape[0] < 1 or probabilities.shape[1] < 1:
        raise EvaluationError("evaluation needs at least one example and one class")
    if not np.isfinite(probabilities).all():
        raise EvaluationError("model probabilities are not finite")
    _require_one_hot(labels)
    truth = np.argmax(labels, axis=1).astype(np.int64)
    predicted = np.argmax(probabilities, axis=1).astype(np.int64)
    rows = np.arange(probabilities.shape[0])
    confidence = np.asarray(probabilities[rows, predicted], dtype=np.float64)
    correct = predicted == truth
    count = int(probabilities.shape[0])
    correct_count = int(np.count_nonzero(correct))
    return {
        "test_count": count,
        "correct_predictions": correct_count,
        "top1_accuracy": correct_count / count,
        "truth": truth,
        "predicted": predicted,
        "confidence": confidence,
        "correct": correct,
    }


def confusion_counts(truth: np.ndarray, predicted: np.ndarray, classes: int) -> np.ndarray:
    """Return counts with true class as the row and predicted class as the column."""
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    return matrix


def select_examples(correct: np.ndarray, limit: int = EXAMPLE_LIMIT) -> dict[str, list[int]]:
    """Select the first correct and incorrect test indices, at most ``limit`` each."""
    if limit < 1:
        raise EvaluationError("example limit must be positive")
    correct_indices = [int(index) for index in np.flatnonzero(correct)[:limit]]
    incorrect_indices = [int(index) for index in np.flatnonzero(~correct)[:limit]]
    return {"correct": correct_indices, "incorrect": incorrect_indices}


def load_evaluation_summary(run_dir: Path) -> dict:
    """Re-validate a saved evaluation against the current run files. Does not predict."""
    run_dir = Path(run_dir)
    completed = validate_completed_run(run_dir)
    summary = _validate_published(run_dir, completed["metadata"], completed)
    summary["reused"] = True
    return summary


def validate_completed_run(run_dir: Path) -> dict:
    """Re-read a completed run from disk before evaluation or reporting.

    The final model digest is the one recorded when training published
    ``model.keras``. Evaluation does not create that identity.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise EvaluationError(f"run directory does not exist: {run_dir}")
    metadata = read_metadata(run_dir)
    status = read_status(run_dir)
    if status.get("status") != "completed":
        raise EvaluationError(
            f"run status is {status.get('status')!r}; evaluation requires a completed run"
        )
    schedule = metadata.get("schedule")
    if type(schedule) is not dict or type(schedule.get("epochs")) is not int:
        raise EvaluationError("run metadata has no integer epoch budget")
    if metadata.get("data_source") == "official":
        spec = contracts.EXPERIMENT_SPECS.get(metadata.get("experiment"))
        if spec is None or schedule["epochs"] != spec.epochs:
            raise EvaluationError("official run epoch budget is not the registered schedule")
        if tuple(int(item) for item in schedule.get("milestones") or ()) != spec.lr_milestones:
            raise EvaluationError("official run learning-rate milestones are not registered")
    _require_final_model(run_dir, status)
    _require_config_identity(run_dir, metadata)
    history = read_completed_history(run_dir, int(schedule["epochs"]))
    _require_history_identity(run_dir, history, status)
    return {"metadata": metadata, "status": status, "history": history}


def read_completed_history(run_dir: Path, expected_epochs: int) -> list[dict]:
    """Read history without rewriting it. The epoch budget must be contiguous."""
    if type(expected_epochs) is not int or expected_epochs < 1:
        raise EvaluationError(f"epoch budget is invalid: {expected_epochs!r}")
    path = Path(run_dir) / "history.csv"
    if not path.is_file():
        raise EvaluationError(f"{path} is missing")
    with path.open(newline="", encoding="utf-8") as handle:
        parsed = list(csv.reader(handle))
    if not parsed or tuple(parsed[0]) != contracts.HISTORY_COLUMNS:
        raise EvaluationError("history.csv header does not match the training record")
    rows = []
    for line_number, cells in enumerate(parsed[1:], start=2):
        if cells in ([], [""]):
            continue
        row = _parse_history_cells(cells)
        if row is None:
            raise EvaluationError(f"history.csv row {line_number} is malformed: {cells!r}")
        rows.append(row)
    if len(rows) != expected_epochs:
        raise EvaluationError(
            f"history.csv has {len(rows)} epochs; the completed run budget is {expected_epochs}"
        )
    for index, row in enumerate(rows, start=1):
        if row["epoch"] != index:
            raise EvaluationError(
                f"history.csv is not contiguous at position {index}: found epoch {row['epoch']}"
            )
    return rows


def evaluate_run(
    run_dir: Path,
    *,
    confirm_final_test: str | None,
    cache_root: Path | None = None,
    test_images: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    display_images: np.ndarray | None = None,
    repeat_inference_reason: str | None = None,
) -> dict:
    """Evaluate one completed run, reuse it, or finish a saved prediction."""
    _require_confirmation(confirm_final_test)
    run_dir = Path(run_dir)
    completed = validate_completed_run(run_dir)
    metadata = completed["metadata"]
    summary_path = run_dir / EVALUATION_DIRNAME / SUMMARY_NAME
    if _committed_evaluation(summary_path) is not None:
        summary = _validate_published(run_dir, metadata, completed)
        summary["reused"] = True
        summary["inference"] = False
        return summary
    if _predictions_recoverable(run_dir, completed):
        summary = _finish_saved_predictions(
            run_dir,
            completed,
            cache_root=cache_root,
            display_images=display_images,
        )
        summary["reused"] = False
        summary["inference"] = False
        return summary
    _authorize_inference(run_dir, completed, repeat_inference_reason)
    if metadata.get("data_source") == "official":
        _require_official_cache(run_dir, metadata, cache_root)
    history_bytes = (run_dir / "history.csv").read_bytes()
    model_path = run_dir / "model.keras"
    model_hash = completed["status"]["final_model_sha256"]
    images, labels, display = _resolve_test_arrays(
        run_dir,
        metadata,
        cache_root=cache_root,
        test_images=test_images,
        test_labels=test_labels,
        display_images=display_images,
    )
    model = _load_saved_model(model_path)
    probabilities = _predict_probabilities(model, images, int(metadata["batch_size"]))
    if probabilities.shape[0] != images.shape[0]:
        raise EvaluationError("model prediction count does not match the test set")
    if _sha256_file(model_path) != model_hash or (run_dir / "history.csv").read_bytes() != history_bytes:
        raise EvaluationError("evaluation changed model.keras or history.csv")
    scored = score_probabilities(probabilities, labels)
    _require_class_range(scored, int(metadata["output_classes"]))
    counts = confusion_counts(scored["truth"], scored["predicted"], int(metadata["output_classes"]))
    _save_prediction_artifacts(run_dir, completed, scored, counts)
    summary = _finish_saved_predictions(
        run_dir,
        completed,
        cache_root=cache_root,
        display_images=display,
    )
    if _sha256_file(model_path) != model_hash or (run_dir / "history.csv").read_bytes() != history_bytes:
        raise EvaluationError("evaluation changed model.keras or history.csv")
    summary["reused"] = False
    summary["inference"] = True
    return summary


def _require_confirmation(confirm_final_test: str | None) -> None:
    if confirm_final_test != contracts.FINAL_TEST_GUARD:
        raise EvaluationError(
            "final evaluation requires authorization "
            f"{contracts.FINAL_TEST_GUARD!r} before any test access"
        )


def _require_final_model(run_dir: Path, status: dict) -> None:
    recorded = status.get("final_model_sha256")
    path = run_dir / "model.keras"
    if not _is_sha256(recorded):
        raise EvaluationError(
            "completed run has no final model digest; evaluation will not choose one"
        )
    if not path.is_file() or path.stat().st_size == 0:
        raise EvaluationError("completed run has no model.keras")
    if _sha256_file(path) != recorded:
        raise EvaluationError(
            "model.keras does not match the final model recorded when training completed"
        )


def _require_config_identity(run_dir: Path, metadata: dict) -> None:
    snapshot = run_dir / "config.yaml"
    if not snapshot.is_file():
        raise EvaluationError("run config.yaml is missing")
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if digest != metadata.get("config_sha256"):
        raise EvaluationError("config.yaml does not match the digest recorded for this run")
    try:
        config = load_config(snapshot)
    except ConfigError as exc:
        raise EvaluationError(f"saved config violates the locked study: {exc}") from exc
    if config.experiment != metadata.get("experiment"):
        raise EvaluationError(
            f"config experiment {config.experiment} does not match run {metadata.get('experiment')}"
        )
    if config.study != metadata.get("study"):
        raise EvaluationError("config study id does not match the completed run")
    _require_metadata_matches_config(metadata, config)


def _require_metadata_matches_config(metadata: dict, config) -> None:
    """Reject metadata that disagrees with the validated config snapshot.

    The config digest only proves the snapshot file is unchanged. Fields copied
    into ``metadata.json`` are compared to that snapshot. A synthetic run may
    record a shorter epoch budget than the study config; that budget is not a
    second copy of ``training.epochs``.
    """
    mismatches = []

    def check(label: str, recorded: object, expected: object) -> None:
        if not _same_recorded_value(recorded, expected):
            mismatches.append(f"{label}: metadata {recorded!r}, configuration {expected!r}")

    check("study", metadata.get("study"), config.study)
    check("experiment", metadata.get("experiment"), config.experiment)
    check("seed", metadata.get("seed"), config.training.seed)
    check("architecture", metadata.get("architecture"), config.model.architecture)
    if "effective_l2" in metadata:
        check("effective_l2", metadata.get("effective_l2"), config.training.l2_coefficient)
    if "keras_kernel_l2" in metadata:
        check("keras_kernel_l2", metadata.get("keras_kernel_l2"), contracts.KERAS_KERNEL_L2)
    schedule = metadata.get("schedule")
    if type(schedule) is dict:
        check(
            "initial_learning_rate",
            schedule.get("initial_learning_rate"),
            config.training.initial_learning_rate,
        )
        check("lr_decay_factor", schedule.get("decay_factor"), config.training.lr_decay_factor)
        if metadata.get("data_source") == "official":
            check("epochs", schedule.get("epochs"), config.training.epochs)
            check(
                "lr_milestones",
                tuple(int(item) for item in schedule.get("milestones") or ()),
                config.training.lr_milestones,
            )
            check("batch_size", metadata.get("batch_size"), config.training.batch_size)
            check("output_classes", metadata.get("output_classes"), config.dataset.num_classes)
    if mismatches:
        raise EvaluationError(
            "run metadata disagrees with the saved configuration: " + "; ".join(mismatches)
        )


def _same_recorded_value(recorded: object, expected: object) -> bool:
    if type(recorded) is float or type(expected) is float:
        try:
            return math.isclose(float(recorded), float(expected), rel_tol=0, abs_tol=0)
        except (TypeError, ValueError):
            return False
    return recorded == expected


def _require_history_identity(run_dir: Path, rows: list[dict], status: dict) -> None:
    _require_history_time(rows, status)
    recorded = status.get("history_sha256")
    if not _is_sha256(recorded):
        raise EvaluationError("completed run has no history digest recorded at completion")
    if _sha256_file(run_dir / "history.csv") != recorded:
        raise EvaluationError("history.csv does not match the digest recorded when training completed")


def _predict_probabilities(model, images: np.ndarray, batch_size: int) -> np.ndarray:
    return np.asarray(model.predict(images, batch_size=batch_size, verbose=0))


def _require_class_range(scored: dict, classes: int) -> None:
    if scored["predicted"].max(initial=0) >= classes or scored["truth"].max(initial=0) >= classes:
        raise EvaluationError("a predicted or true class is outside the run's output classes")


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _resolve_test_arrays(
    run_dir: Path,
    metadata: dict,
    *,
    cache_root: Path | None,
    test_images: np.ndarray | None,
    test_labels: np.ndarray | None,
    display_images: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = int(metadata["output_classes"])
    if test_images is None or test_labels is None:
        if metadata.get("data_source") != "official":
            raise EvaluationError("a synthetic run needs explicit test arrays supplied by the caller")
        if display_images is not None:
            raise EvaluationError("official evaluation loads display images inside the authorized command")
        images, labels = _load_official_test(run_dir, metadata, cache_root)
        display = _official_display_images(labels)
    else:
        if metadata.get("data_source") == "official":
            raise EvaluationError("official evaluation must load the cached test partition")
        images, labels, display = test_images, test_labels, display_images
    if labels.ndim != 2 or labels.shape[1] != classes or labels.dtype != np.float32:
        raise EvaluationError(
            f"test labels must be float32 one-hot with {classes} columns, got {labels.shape} {labels.dtype}"
        )
    if images.shape[0] != labels.shape[0]:
        raise EvaluationError("test images and labels have different lengths")
    if images.dtype != np.float32:
        raise EvaluationError(f"preprocessed test images must be float32, got {images.dtype}")
    if metadata.get("data_source") == "official":
        if classes != contracts.NUM_CLASSES or labels.shape != (contracts.TEST_SIZE, contracts.NUM_CLASSES):
            raise EvaluationError("official evaluation requires 10,000 one-hot fine labels")
        if images.shape != (contracts.TEST_SIZE, *contracts.IMAGE_SHAPE):
            raise EvaluationError(
                f"official preprocessed test images must have shape {(contracts.TEST_SIZE, *contracts.IMAGE_SHAPE)}"
            )
    _require_display_images(display, labels.shape[0])
    return images, labels, display


def _require_official_cache(run_dir: Path, metadata: dict, cache_root: Path | None) -> Path:
    from cifar100_study.data import cache_key, preprocessing_identity

    if cache_root is None:
        raise EvaluationError("official evaluation requires the preprocessing cache root")
    cache_dir = Path(cache_root) / str(metadata.get("cache_key"))
    if not cache_dir.is_dir():
        raise EvaluationError(f"preprocessing cache {cache_dir.name} is not under {cache_root}")
    manifest_path = cache_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"preprocessing cache manifest is unreadable: {exc}") from exc
    if manifest.get("source_fingerprint") != metadata.get("source_fingerprint"):
        raise EvaluationError("preprocessing fingerprint does not match the completed run")
    identity = manifest.get("identity")
    config = load_config(run_dir / "config.yaml")
    if identity != preprocessing_identity(config):
        raise EvaluationError("preprocessing identity does not match the completed run")
    fingerprint = manifest.get("source_fingerprint")
    if type(fingerprint) is not str or type(identity) is not dict:
        raise EvaluationError("preprocessing cache manifest is missing its identity")
    if cache_dir.name != cache_key(identity, fingerprint):
        raise EvaluationError("cache directory is not the published cache key")
    return cache_dir


def _load_official_test(
    run_dir: Path,
    metadata: dict,
    cache_root: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    from cifar100_study.data import load_test_data

    cache_dir = _require_official_cache(run_dir, metadata, cache_root)
    return load_test_data(cache_dir, confirm_final_test=contracts.FINAL_TEST_GUARD)


def _official_display_images(one_hot: np.ndarray) -> np.ndarray:
    from cifar100_study.data import canonical_labels, load_official_cifar100, validate_raw_images

    (_train_images, _train_labels), (images, labels) = load_official_cifar100()
    del _train_images, _train_labels
    validate_raw_images(images, contracts.IMAGE_SHAPE, contracts.TEST_SIZE, "test")
    truth = np.argmax(one_hot, axis=1).astype(np.int64)
    raw = canonical_labels(labels)
    if raw.shape != truth.shape or not np.array_equal(raw, truth):
        raise EvaluationError("official CIFAR-100 test labels do not match the cached test indices")
    return np.ascontiguousarray(images)


def _require_display_images(images: np.ndarray | None, count: int) -> None:
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8 or images.ndim != 4:
        raise EvaluationError("display images must be uint8 RGB arrays, not preprocessed inputs")
    if images.shape[0] != count or images.shape[-1] != 3:
        raise EvaluationError(
            f"display images must contain {count} RGB examples in test order, got {getattr(images, 'shape', None)}"
        )


def _load_saved_model(model_path: Path):
    from keras.saving import load_model

    try:
        return load_model(model_path)
    except Exception as exc:
        raise EvaluationError(f"model.keras is not loadable: {exc}") from exc


def _measured_complexity(model, run_dir: Path, metadata: dict) -> dict:
    from cifar100_study.allcnn import architecture_report

    measured = architecture_report(model)
    if measured.trainable_parameters != metadata.get("trainable_parameters"):
        raise EvaluationError("loaded model parameter count does not match the run")
    if measured.convolutional_layers != metadata.get("convolutional_layers"):
        raise EvaluationError("loaded model depth does not match the run")
    registered = REGISTERED_COMPLEXITY.get(metadata.get("experiment"))
    measured_tuple = (
        measured.convolutional_layers,
        measured.trainable_parameters,
        measured.macs_per_image,
    )
    if registered is not None and measured_tuple[:2] == registered[:2] and measured_tuple != registered:
        raise EvaluationError("loaded model MACs do not match the registered architecture")
    summary_path = run_dir / "model_summary.txt"
    if summary_path.is_file():
        recorded = _macs_from_summary(summary_path)
        if recorded != measured.macs_per_image:
            raise EvaluationError("model summary MAC count does not match the loaded model")
    return {
        "trainable_parameters": measured.trainable_parameters,
        "convolutional_layers": measured.convolutional_layers,
        "macs_per_image": measured.macs_per_image,
    }


def _macs_from_summary(path: Path) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("convolution_macs_per_image:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError as exc:
                raise EvaluationError("model summary MAC count is not an integer") from exc
    raise EvaluationError("model summary has no convolution MAC count")


def _summary_payload(metadata: dict, scored: dict, measured: dict, model_hash: str, status: dict) -> dict:
    return {
        "format_version": 1,
        "run_id": metadata["run_id"],
        "experiment": metadata["experiment"],
        "test_count": scored["test_count"],
        "correct_predictions": scored["correct_predictions"],
        "top1_accuracy": scored["top1_accuracy"],
        "output_classes": int(metadata["output_classes"]),
        "cache_key": metadata.get("cache_key"),
        "source_fingerprint": metadata.get("source_fingerprint"),
        "config_sha256": metadata["config_sha256"],
        "trainable_parameters": measured["trainable_parameters"],
        "convolutional_layers": measured["convolutional_layers"],
        "macs_per_image": measured["macs_per_image"],
        "mac_unit": MAC_UNIT,
        "epochs": int(metadata["schedule"]["epochs"]),
        "epoch_seconds_total": float(status["epoch_seconds_total"]),
        "device": metadata.get("device"),
        "model_sha256": model_hash,
    }


def _authorize_inference(run_dir: Path, completed: dict, repeat_inference_reason: str | None) -> None:
    directory = run_dir / EVALUATION_DIRNAME
    directory.mkdir(exist_ok=True)
    attempt_path = directory / ATTEMPT_NAME
    if not attempt_path.is_file():
        _write_json(
            attempt_path,
            {
                "format_version": 1,
                "model_sha256": completed["status"]["final_model_sha256"],
                "predictions_sha256": None,
                "repeat_authorizations": [],
                "run_id": completed["metadata"]["run_id"],
                "state": "started",
            },
        )
        return
    attempt = _read_json(attempt_path)
    try:
        recoverable = attempt.get("state") == "predictions_saved" and _prediction_files_match(
            directory, attempt, completed
        )
    except EvaluationError:
        recoverable = False
    if recoverable:
        return
    reason = (repeat_inference_reason or "").strip()
    if not reason:
        raise EvaluationError(
            "saved predictions cannot be recovered; repeating inference requires "
            "an explicit reason recorded with the run"
        )
    authorizations = list(attempt.get("repeat_authorizations") or [])
    authorizations.append(
        {
            "reason": reason,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    attempt["repeat_authorizations"] = authorizations
    attempt["state"] = "started"
    attempt["predictions_sha256"] = None
    _write_json(attempt_path, attempt)


def _save_prediction_artifacts(run_dir: Path, completed: dict, scored: dict, counts: np.ndarray) -> None:
    directory = run_dir / EVALUATION_DIRNAME
    directory.mkdir(exist_ok=True)
    predictions_path = directory / PREDICTIONS_NAME
    confusion_path = directory / CONFUSION_NAME
    _write_predictions(predictions_path, scored)
    _write_confusion(confusion_path, counts)
    manifest = {
        "confusion_sha256": _sha256_file(confusion_path),
        "correct_predictions": scored["correct_predictions"],
        "model_sha256": completed["status"]["final_model_sha256"],
        "output_classes": int(completed["metadata"]["output_classes"]),
        "predictions_sha256": _sha256_file(predictions_path),
        "test_count": scored["test_count"],
        "top1_accuracy": scored["top1_accuracy"],
    }
    _write_json(directory / PREDICTION_MANIFEST_NAME, manifest)
    attempt = _read_json(directory / ATTEMPT_NAME)
    attempt["state"] = "predictions_saved"
    attempt["predictions_sha256"] = manifest["predictions_sha256"]
    attempt["model_sha256"] = manifest["model_sha256"]
    _write_json(directory / ATTEMPT_NAME, attempt)


def _predictions_recoverable(run_dir: Path, completed: dict) -> bool:
    directory = run_dir / EVALUATION_DIRNAME
    manifest_path = directory / PREDICTION_MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        attempt = _read_json(directory / ATTEMPT_NAME)
        return _prediction_files_match(directory, attempt, completed)
    except EvaluationError:
        return False


def _prediction_files_match(directory: Path, attempt: dict, completed: dict) -> bool:
    if attempt.get("state") != "predictions_saved":
        return False
    manifest = _read_json(directory / PREDICTION_MANIFEST_NAME)
    model_sha = completed["status"]["final_model_sha256"]
    if manifest.get("model_sha256") != model_sha or attempt.get("model_sha256") != model_sha:
        return False
    predictions_path = directory / PREDICTIONS_NAME
    confusion_path = directory / CONFUSION_NAME
    if not predictions_path.is_file() or not confusion_path.is_file():
        return False
    if _sha256_file(predictions_path) != manifest.get("predictions_sha256"):
        return False
    if _sha256_file(confusion_path) != manifest.get("confusion_sha256"):
        return False
    predictions = _read_predictions(predictions_path)
    if len(predictions) != int(manifest["test_count"]):
        return False
    correct = sum(row["true_label"] == row["predicted_label"] for row in predictions)
    if correct != int(manifest["correct_predictions"]):
        return False
    counts = _read_confusion(confusion_path, int(manifest["output_classes"]))
    if not np.array_equal(counts, _confusion_from_predictions(predictions, int(manifest["output_classes"]))):
        return False
    return True


def _finish_saved_predictions(
    run_dir: Path,
    completed: dict,
    *,
    cache_root: Path | None,
    display_images: np.ndarray | None,
) -> dict:
    directory = run_dir / EVALUATION_DIRNAME
    manifest = _read_json(directory / PREDICTION_MANIFEST_NAME)
    predictions = _read_predictions(directory / PREDICTIONS_NAME)
    classes = int(manifest["output_classes"])
    counts = _confusion_from_predictions(predictions, classes)
    if not np.array_equal(counts, _read_confusion(directory / CONFUSION_NAME, classes)):
        raise EvaluationError("confusion.csv does not match the saved predictions")
    display = _display_for_saved_predictions(
        completed["metadata"],
        predictions,
        display_images,
        cache_root,
    )
    scored = _scored_from_predictions(predictions)
    measured = _measured_complexity(_load_saved_model(run_dir / "model.keras"), run_dir, completed["metadata"])
    summary = _summary_payload(
        completed["metadata"],
        scored,
        measured,
        completed["status"]["final_model_sha256"],
        completed["status"],
    )
    _write_examples(directory / EXAMPLES_NAME, directory / EXAMPLES_FIGURE_NAME, display, scored, classes)
    summary["predictions_sha256"] = _sha256_file(directory / PREDICTIONS_NAME)
    summary["confusion_sha256"] = _sha256_file(directory / CONFUSION_NAME)
    summary["examples_json_sha256"] = _sha256_file(directory / EXAMPLES_NAME)
    summary["examples_png_sha256"] = _sha256_file(directory / EXAMPLES_FIGURE_NAME)
    if tuple(sorted(summary)) != _SUMMARY_KEYS:
        raise EvaluationError("evaluation summary fields are incomplete")
    _write_json(directory / SUMMARY_NAME, summary)
    attempt = _read_json(directory / ATTEMPT_NAME)
    attempt["state"] = "published"
    _write_json(directory / ATTEMPT_NAME, attempt)
    published = _validate_published(run_dir, completed["metadata"], completed)
    return published


def _display_for_saved_predictions(
    metadata: dict,
    predictions: list[dict],
    display_images: np.ndarray | None,
    cache_root: Path | None,
) -> np.ndarray:
    del cache_root
    if display_images is not None:
        _require_display_images(display_images, len(predictions))
        return display_images
    if metadata.get("data_source") != "official":
        raise EvaluationError("finishing saved predictions requires the original RGB display images")
    classes = int(metadata["output_classes"])
    one_hot = np.zeros((len(predictions), classes), dtype=np.float32)
    for row in predictions:
        one_hot[row["index"], row["true_label"]] = np.float32(1.0)
    return _official_display_images(one_hot)


def _scored_from_predictions(predictions: list[dict]) -> dict:
    truth = np.asarray([row["true_label"] for row in predictions], dtype=np.int64)
    predicted = np.asarray([row["predicted_label"] for row in predictions], dtype=np.int64)
    confidence = np.asarray([row["confidence"] for row in predictions], dtype=np.float64)
    return {
        "test_count": len(predictions),
        "correct_predictions": int(np.count_nonzero(truth == predicted)),
        "top1_accuracy": int(np.count_nonzero(truth == predicted)) / len(predictions),
        "truth": truth,
        "predicted": predicted,
        "confidence": confidence,
        "correct": truth == predicted,
    }


def _confusion_from_predictions(predictions: list[dict], classes: int) -> np.ndarray:
    truth = np.asarray([row["true_label"] for row in predictions], dtype=np.int64)
    predicted = np.asarray([row["predicted_label"] for row in predictions], dtype=np.int64)
    return confusion_counts(truth, predicted, classes)


def _write_predictions(path: Path, scored: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(PREDICTION_COLUMNS)
        for index, (truth, predicted, confidence) in enumerate(
            zip(scored["truth"], scored["predicted"], scored["confidence"], strict=True)
        ):
            writer.writerow((index, int(truth), int(predicted), format(float(confidence), ".10g")))
        handle.flush()
        os.fsync(handle.fileno())


def _write_confusion(path: Path, counts: np.ndarray) -> None:
    lines = [",".join(str(int(value)) for value in row) for row in counts]
    _write_text(path, "\n".join(lines) + "\n")


def _write_examples(path: Path, figure_path: Path, display: np.ndarray, scored: dict, classes: int) -> None:
    selected = select_examples(scored["correct"])
    payload = {"rule": EXAMPLE_RULE, "correct": [], "incorrect": []}
    for key, indices in selected.items():
        for index in indices:
            image = np.ascontiguousarray(display[index])
            payload[key].append(
                {
                    "confidence": float(scored["confidence"][index]),
                    "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "index": index,
                    "predicted_label": int(scored["predicted"][index]),
                    "true_label": int(scored["truth"][index]),
                }
            )
    _write_json(path, payload)
    _write_example_figure(figure_path, display, scored, selected, classes)


def _write_example_figure(path: Path, display: np.ndarray, scored: dict, selected: dict, classes: int) -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, EXAMPLE_LIMIT, figsize=(8.4, 4.6))
    for row, key in enumerate(("correct", "incorrect")):
        indices = selected[key]
        for column in range(EXAMPLE_LIMIT):
            axis = axes[row, column]
            axis.set_xticks([])
            axis.set_yticks([])
            if column >= len(indices):
                axis.axis("off")
                if column == 0 and not indices:
                    axis.text(0.5, 0.5, f"no {key} examples", ha="center", va="center")
                    axis.axis("on")
                    axis.set_xticks([])
                    axis.set_yticks([])
                continue
            index = indices[column]
            axis.imshow(display[index])
            truth = class_name(int(scored["truth"][index]), classes)
            predicted = class_name(int(scored["predicted"][index]), classes)
            axis.set_title(f"true: {truth}\npred: {predicted}", fontsize=8)
    figure.suptitle("Test examples", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _validate_published(run_dir: Path, metadata: dict, completed: dict | None = None) -> dict:
    directory = run_dir / EVALUATION_DIRNAME
    if not directory.is_dir():
        raise EvaluationError("evaluation outputs are incomplete")
    expected_names = {
        SUMMARY_NAME,
        PREDICTIONS_NAME,
        CONFUSION_NAME,
        EXAMPLES_NAME,
        EXAMPLES_FIGURE_NAME,
        ATTEMPT_NAME,
        PREDICTION_MANIFEST_NAME,
    }
    found = {path.name for path in directory.iterdir() if not path.name.startswith(".")}
    if found != expected_names:
        missing = sorted(expected_names - found)
        extra = sorted(found - expected_names)
        raise EvaluationError(
            f"evaluation outputs are partial or inconsistent: missing {missing}, extra {extra}"
        )
    summary = _read_json(directory / SUMMARY_NAME)
    if tuple(sorted(summary)) != _SUMMARY_KEYS or summary.get("format_version") != 1:
        raise EvaluationError("evaluation.json is inconsistent")
    _require_hash(directory / PREDICTIONS_NAME, summary["predictions_sha256"])
    _require_hash(directory / CONFUSION_NAME, summary["confusion_sha256"])
    _require_hash(directory / EXAMPLES_NAME, summary["examples_json_sha256"])
    _require_hash(directory / EXAMPLES_FIGURE_NAME, summary["examples_png_sha256"])
    _require_hash(run_dir / "model.keras", summary["model_sha256"])
    status = completed["status"] if completed is not None else read_status(run_dir)
    if summary["model_sha256"] != status.get("final_model_sha256"):
        raise EvaluationError("evaluation.json does not name the final model recorded at completion")
    if summary["run_id"] != metadata.get("run_id") or summary["experiment"] != metadata.get("experiment"):
        raise EvaluationError("evaluation.json does not belong to this run")
    if summary["config_sha256"] != metadata.get("config_sha256"):
        raise EvaluationError("evaluation.json does not match the run configuration")
    if summary["cache_key"] != metadata.get("cache_key") or summary["source_fingerprint"] != metadata.get(
        "source_fingerprint"
    ):
        raise EvaluationError("evaluation.json does not match the run preprocessing cache")
    if int(summary["output_classes"]) != int(metadata["output_classes"]):
        raise EvaluationError("evaluation.json class count does not match the run")
    if int(summary["epochs"]) != int(metadata["schedule"]["epochs"]):
        raise EvaluationError("evaluation.json epoch budget does not match the run")
    if summary["device"] != metadata.get("device"):
        raise EvaluationError("evaluation.json device does not match the run")
    if int(summary["trainable_parameters"]) != int(metadata["trainable_parameters"]):
        raise EvaluationError("evaluation.json parameter count does not match the run")
    if int(summary["convolutional_layers"]) != int(metadata["convolutional_layers"]):
        raise EvaluationError("evaluation.json depth does not match the run")
    if summary["mac_unit"] != MAC_UNIT:
        raise EvaluationError("evaluation.json MAC unit is not multiply-accumulates per image")
    predictions = _read_predictions(directory / PREDICTIONS_NAME)
    if len(predictions) != int(summary["test_count"]):
        raise EvaluationError("predictions.csv length does not match evaluation.json")
    correct = sum(row["true_label"] == row["predicted_label"] for row in predictions)
    if correct != int(summary["correct_predictions"]):
        raise EvaluationError("predictions.csv does not match the recorded correct count")
    expected_accuracy = correct / len(predictions)
    if not math.isclose(float(summary["top1_accuracy"]), expected_accuracy, rel_tol=0, abs_tol=1e-12):
        raise EvaluationError("evaluation.json Top-1 accuracy does not match the predictions")
    counts = _read_confusion(directory / CONFUSION_NAME, int(summary["output_classes"]))
    rebuilt = _confusion_from_predictions(predictions, int(summary["output_classes"]))
    if not np.array_equal(counts, rebuilt):
        raise EvaluationError("confusion.csv does not match the predictions in every cell")
    if int(np.trace(counts)) != correct:
        raise EvaluationError("confusion.csv diagonal does not match the correct predictions")
    _require_saved_complexity(run_dir, metadata, summary)
    _require_examples_match(directory / EXAMPLES_NAME, predictions)
    if completed is None:
        history = read_completed_history(run_dir, int(metadata["schedule"]["epochs"]))
        _require_history_identity(run_dir, history, status)
    else:
        history = completed["history"]
    history_total = sum(row["epoch_seconds"] for row in history)
    if not math.isclose(float(summary["epoch_seconds_total"]), history_total, rel_tol=0, abs_tol=1e-9):
        raise EvaluationError("evaluation.json training time does not match the current history")
    if not math.isclose(float(summary["epoch_seconds_total"]), float(status["epoch_seconds_total"]), rel_tol=0, abs_tol=1e-9):
        raise EvaluationError("evaluation.json training time does not match the run status")
    return summary


def _require_saved_complexity(run_dir: Path, metadata: dict, summary: dict) -> None:
    measured = _measured_complexity(_load_saved_model(run_dir / "model.keras"), run_dir, metadata)
    if int(summary["macs_per_image"]) != measured["macs_per_image"]:
        raise EvaluationError("evaluation.json MAC count does not match the final model")
    if int(summary["convolutional_layers"]) != measured["convolutional_layers"]:
        raise EvaluationError("evaluation.json depth does not match the final model")
    if int(summary["trainable_parameters"]) != measured["trainable_parameters"]:
        raise EvaluationError("evaluation.json parameter count does not match the final model")


def _require_examples_match(path: Path, predictions: list[dict]) -> None:
    payload = _read_json(path)
    if payload.get("rule") != EXAMPLE_RULE:
        raise EvaluationError("examples.json does not use the registered selection rule")
    seen = []
    for key in ("correct", "incorrect"):
        rows = payload.get(key)
        if type(rows) is not list or len(rows) > EXAMPLE_LIMIT:
            raise EvaluationError("examples.json selection is invalid")
        for row in rows:
            index = row.get("index")
            if type(index) is not int or index < 0 or index >= len(predictions):
                raise EvaluationError("examples.json index is outside the test set")
            prediction = predictions[index]
            if row.get("true_label") != prediction["true_label"] or row.get("predicted_label") != prediction["predicted_label"]:
                raise EvaluationError("examples.json labels do not match predictions.csv at the same index")
            if key == "correct" and prediction["true_label"] != prediction["predicted_label"]:
                raise EvaluationError("examples.json lists a misclassification as correct")
            if key == "incorrect" and prediction["true_label"] == prediction["predicted_label"]:
                raise EvaluationError("examples.json lists a correct prediction as incorrect")
            if type(row.get("image_sha256")) is not str or len(row["image_sha256"]) != 64:
                raise EvaluationError("examples.json is missing the display-image identity")
            seen.append(index)
    if len(seen) != len(set(seen)):
        raise EvaluationError("examples.json repeats a test index")


def _read_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        parsed = list(csv.reader(handle))
    if not parsed or tuple(parsed[0]) != PREDICTION_COLUMNS:
        raise EvaluationError("predictions.csv header is invalid")
    rows = []
    for line_number, cells in enumerate(parsed[1:], start=2):
        if len(cells) != 4:
            raise EvaluationError(f"predictions.csv row {line_number} is malformed")
        try:
            index = int(cells[0])
            truth = int(cells[1])
            predicted = int(cells[2])
            confidence = float(cells[3])
        except ValueError as exc:
            raise EvaluationError(f"predictions.csv row {line_number} is malformed") from exc
        if index != len(rows) or not math.isfinite(confidence) or confidence < -1e-6 or confidence > 1 + 1e-5:
            raise EvaluationError(f"predictions.csv row {line_number} is inconsistent")
        rows.append(
            {
                "index": index,
                "true_label": truth,
                "predicted_label": predicted,
                "confidence": confidence,
            }
        )
    if not rows:
        raise EvaluationError("predictions.csv has no examples")
    return rows


def _read_confusion(path: Path, classes: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != classes:
        raise EvaluationError("confusion.csv does not have one row per class")
    rows = []
    for line in lines:
        cells = line.split(",")
        if len(cells) != classes:
            raise EvaluationError("confusion.csv is not square")
        try:
            values = [int(cell) for cell in cells]
        except ValueError as exc:
            raise EvaluationError("confusion.csv contains a non-integer count") from exc
        if any(value < 0 for value in values):
            raise EvaluationError("confusion.csv contains a negative count")
        rows.append(values)
    return np.asarray(rows, dtype=np.int64)


def _require_history_time(rows: list[dict], status: dict) -> None:
    total = sum(row["epoch_seconds"] for row in rows)
    if not math.isclose(total, float(status.get("epoch_seconds_total")), rel_tol=0, abs_tol=1e-6):
        raise EvaluationError("status epoch time does not equal the sum of history.csv durations")


def _require_one_hot(labels: np.ndarray) -> None:
    if labels.dtype != np.float32:
        raise EvaluationError(f"one-hot labels must be float32, got {labels.dtype}")
    if np.any((labels != 0.0) & (labels != 1.0)):
        raise EvaluationError("one-hot labels must contain only 0 and 1")
    if not np.allclose(labels.sum(axis=1), 1.0):
        raise EvaluationError("each one-hot label must contain a single class")


def _parse_history_cells(cells: list[str]) -> dict | None:
    if len(cells) != len(contracts.HISTORY_COLUMNS):
        return None
    try:
        epoch = int(cells[0])
        if cells[0].strip() != str(epoch):
            return None
        values = [float(cell) for cell in cells[1:]]
    except ValueError:
        return None
    if epoch < 1 or values[3] < 0 or any(not math.isfinite(value) for value in values):
        return None
    return {
        "epoch": epoch,
        "loss": values[0],
        "accuracy": values[1],
        "learning_rate": values[2],
        "epoch_seconds": values[3],
    }


def _require_hash(path: Path, expected: object) -> None:
    if type(expected) is not str or _sha256_file(path) != expected:
        raise EvaluationError(f"{path.name} does not match the recorded digest")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{path.name} is unreadable") from exc
    if type(payload) is not dict:
        raise EvaluationError(f"{path.name} must be a JSON object")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file, then replace the destination."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _committed_evaluation(path: Path) -> dict | None:
    """Return a structurally complete summary. A truncated write is not one."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if type(payload) is not dict or payload.get("format_version") != 1:
        return None
    if tuple(sorted(payload)) != _SUMMARY_KEYS:
        return None
    return payload


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
