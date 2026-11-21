"""Train one locked All-CNN-C run and resume it from an epoch boundary.

Production training uses the prepared training cache, categorical
cross-entropy on one-hot labels, and SGD with momentum 0.9. The scientific
weight decay is 0.001. It is applied only as Keras ``L2(0.0005)`` on
convolution kernels. Optimizer ``weight_decay`` is left unset, so there is no
second, decoupled decay term.

The learning rate for zero-based epoch ``t`` is

    initial * decay ** (number of milestones <= t)

It is recomputed from that formula at the start of every epoch. Resume does
not multiply the restored rate again.

Shuffling uses the in-memory NumPy arrays and Keras ``fit(..., shuffle=True)``.
The array adapter shuffles indices with ``tf.random.shuffle`` at each epoch.
There is no second copy of the training set as a shuffle buffer. The last
batch of an official epoch contains ``50000 % 256`` examples. Dropout stays
on during training.

Seeding covers Python, NumPy, TensorFlow, and Keras. Deterministic TensorFlow
ops are not forced: that setting slows convolution and can reject device
kernels, and it would still not make a resumed shuffle or dropout stream match
an uninterrupted run. Resume restores weights, momentum slots, the optimizer
iteration, and the completed-epoch index, then continues the schedule. It
re-seeds the process before rebuilding the architecture, so the random stream
after resume is not the continuation of the stream that was interrupted.

E1 is always a new run. It builds a new network and a new optimizer. It does
not open an E0 checkpoint.

Final ``model.keras`` is written only after a verified checkpoint and history
both cover the scheduled epoch, through a temporary ``.keras`` file that is
loaded before it is published. Checkpoint generations are kept. This module
does not load test arrays, fit preprocessing, or select a checkpoint by accuracy.
"""

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cifar100_study import contracts
from cifar100_study.allcnn import (
    REGISTERED_COMPLEXITY,
    architecture_report,
    build_allcnn,
    format_architecture_report,
)
from cifar100_study.config import ExperimentConfig, TrainingConfig, differing_paths, load_config
from cifar100_study.data import cache_key, load_training_data, preprocessing_identity
from cifar100_study.runs import (
    TrainingDivergence,
    TrainingError,
    TrainingInterrupted,
    close_session,
    create_run,
    git_state,
    load_history,
    open_session,
    read_metadata,
    read_status,
    reconcile_history,
    recovered_completed_epochs,
    restore_checkpoint,
    update_status,
)
from cifar100_study.training_callbacks import make_training_callbacks

SHUFFLE_POLICY = (
    "in-memory NumPy arrays; Keras fit shuffle=True uses tf.random.shuffle "
    "on indices each epoch; no full-dataset shuffle buffer"
)
DETERMINISM_NOTE = (
    "Python, NumPy, TensorFlow, and Keras are seeded. Deterministic ops are "
    "not forced. Resume restores weights, momentum, optimizer iterations, and "
    "the epoch index, but not the shuffle or dropout random stream."
)
_REFERENCE = REGISTERED_COMPLEXITY


@dataclass(frozen=True, slots=True)
class EpochSchedule:
    """Absolute epoch-index schedule. ``rate`` does not use the previous rate."""

    initial_learning_rate: float
    decay_factor: float
    milestones: tuple[int, ...]
    epochs: int

    def rate(self, epoch_index: int) -> float:
        if type(epoch_index) is not int or epoch_index < 0:
            raise TrainingError(f"epoch index must be a nonnegative integer, got {epoch_index!r}")
        return scheduled_learning_rate(
            epoch_index,
            initial_learning_rate=self.initial_learning_rate,
            decay_factor=self.decay_factor,
            milestones=self.milestones,
        )


def scheduled_learning_rate(
    epoch_index: int,
    *,
    initial_learning_rate: float,
    decay_factor: float,
    milestones: tuple[int, ...],
) -> float:
    """Return ``initial * decay ** count(milestone <= epoch_index)``."""
    applied = sum(1 for milestone in milestones if epoch_index >= milestone)
    return float(initial_learning_rate) * float(decay_factor) ** applied


def require_registered_schedule(config: ExperimentConfig) -> EpochSchedule:
    """Reject any schedule other than the pre-registered experiment schedule."""
    spec = contracts.EXPERIMENT_SPECS.get(config.experiment)
    if spec is None:
        raise TrainingError(f"unknown experiment {config.experiment!r}")
    training = config.training
    if training.lr_milestone_convention != contracts.LR_MILESTONE_CONVENTION:
        raise TrainingError("production training refuses an unregistered milestone convention")
    if training.epochs != spec.epochs or training.lr_milestones != spec.lr_milestones:
        raise TrainingError(
            f"{config.experiment} refuses schedule epochs={training.epochs} "
            f"milestones={training.lr_milestones}"
        )
    if training.initial_learning_rate != contracts.INITIAL_LEARNING_RATE:
        raise TrainingError("production training refuses an unregistered initial learning rate")
    if training.lr_decay_factor != contracts.LR_DECAY_FACTOR:
        raise TrainingError("production training refuses an unregistered decay factor")
    return EpochSchedule(
        initial_learning_rate=training.initial_learning_rate,
        decay_factor=training.lr_decay_factor,
        milestones=training.lr_milestones,
        epochs=training.epochs,
    )


def seed_runtime(seed: int) -> None:
    """Seed Python, NumPy, TensorFlow, and Keras. Does not enable deterministic ops."""
    import random

    import tensorflow as tf
    from keras.utils import set_random_seed

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    set_random_seed(seed)


def select_device(requested: str) -> dict:
    """Resolve ``auto``, ``cpu``, or ``gpu`` and run a one-op preflight."""
    if requested not in {"auto", "cpu", "gpu"}:
        raise TrainingError(f"device must be auto, cpu, or gpu, got {requested!r}")
    import tensorflow as tf

    gpus = list(tf.config.list_physical_devices("GPU"))
    cpus = list(tf.config.list_physical_devices("CPU"))
    if requested == "gpu" and not gpus:
        raise TrainingError(
            "requested device gpu is unavailable; TensorFlow reported no GPU. "
            "Training will not silently fall back to CPU."
        )
    if requested == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            if tf.config.get_visible_devices("GPU"):
                raise TrainingError(
                    "requested CPU execution but TensorFlow GPUs are already "
                    "initialized and cannot be hidden in this process"
                ) from exc
        selected_type = "CPU"
    elif requested == "gpu" or gpus:
        selected_type = "GPU"
    else:
        selected_type = "CPU"
    logical = list(tf.config.list_logical_devices(selected_type))
    if not logical:
        raise TrainingError(f"TensorFlow exposed no logical {selected_type} device")
    physical = gpus if selected_type == "GPU" else cpus
    preflight_device = _preflight(tf, logical[0].name, selected_type)
    return {
        "requested": requested,
        "selected_type": selected_type,
        "selected_name": physical[0].name if physical else logical[0].name,
        "logical_device": logical[0].name,
        "preflight_device": preflight_device,
        "gpu_devices": [device.name for device in gpus],
        "cpu_devices": [device.name for device in cpus],
        "deterministic_ops": False,
        "determinism_note": DETERMINISM_NOTE,
    }


def compile_study_model(model, training: TrainingConfig) -> None:
    """Compile with the locked SGD, categorical loss, and categorical accuracy."""
    if training.optimizer != contracts.OPTIMIZER:
        raise TrainingError(f"optimizer must be {contracts.OPTIMIZER!r}")
    if training.momentum != contracts.MOMENTUM:
        raise TrainingError(f"momentum must be {contracts.MOMENTUM}")
    if training.initial_learning_rate != contracts.INITIAL_LEARNING_RATE:
        raise TrainingError("initial learning rate is locked")
    if training.l2_coefficient != contracts.L2_COEFFICIENT:
        raise TrainingError(
            "l2_coefficient is the scientific weight decay 0.001, not an optimizer weight_decay"
        )
    if training.loss != contracts.LOSS:
        raise TrainingError(f"loss must be {contracts.LOSS!r}")
    if training.early_stopping or training.pretrained_weights:
        raise TrainingError("early stopping and pretrained weights are outside the study")
    if training.checkpoint_selection != contracts.CHECKPOINT_SELECTION:
        raise TrainingError("checkpoint selection must be the final scheduled epoch")
    from keras import losses, metrics, optimizers

    model.compile(
        optimizer=optimizers.SGD(
            learning_rate=float(training.initial_learning_rate),
            momentum=float(training.momentum),
            nesterov=False,
            weight_decay=None,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            loss_scale_factor=None,
            gradient_accumulation_steps=None,
        ),
        loss=losses.CategoricalCrossentropy(),
        metrics=[metrics.CategoricalAccuracy(name="accuracy")],
    )
    optimizer = model.optimizer
    if optimizer.weight_decay is not None or optimizer.nesterov:
        raise TrainingError("SGD must not use Nesterov momentum or optimizer weight decay")
    if optimizer.clipnorm is not None or optimizer.clipvalue is not None:
        raise TrainingError("gradient clipping is not part of the study")
    if optimizer.use_ema or optimizer.gradient_accumulation_steps not in (None, 0):
        raise TrainingError("EMA and gradient accumulation are not part of the study")


def resolve_official_cache(config: ExperimentConfig, cache_root: Path) -> tuple[Path, str]:
    """Find the one prepared cache for this study. Do not fit preprocessing."""
    identity = preprocessing_identity(config)
    root = Path(cache_root)
    if not root.is_dir():
        raise TrainingError(f"cache root {root} does not exist; run prepare.py before training")
    matches: list[tuple[Path, str]] = []
    for child in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        manifest_path = child / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("identity") != identity:
            continue
        fingerprint = manifest.get("source_fingerprint")
        if type(fingerprint) is not str:
            continue
        if child.name != cache_key(identity, fingerprint):
            continue
        matches.append((child, fingerprint))
    if not matches:
        raise TrainingError(
            "no official CIFAR-100 preprocessing cache matches this study; "
            "synthetic caches are not accepted for a production run"
        )
    if len(matches) > 1:
        raise TrainingError("multiple official caches matched; refusing to choose one")
    return matches[0]


def load_official_training(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the official training arrays. This does not open test files."""
    images, labels = load_training_data(cache_dir)
    if images.shape != (contracts.TRAIN_SIZE, *contracts.IMAGE_SHAPE):
        raise TrainingError(
            f"production training requires the official training images, got {images.shape}"
        )
    if labels.shape != (contracts.TRAIN_SIZE, contracts.NUM_CLASSES) or labels.dtype != np.float32:
        raise TrainingError(
            "production training requires float32 one-hot labels of shape "
            f"({contracts.TRAIN_SIZE}, {contracts.NUM_CLASSES})"
        )
    return images, labels


def create_training_run(
    run_root: Path,
    *,
    config: ExperimentConfig,
    config_text: str,
    schedule: EpochSchedule,
    data_source: str,
    batch_size: int,
    output_classes: int,
    device: dict,
    cache_key_hex: str | None,
    source_fingerprint: str | None,
    trainable_parameters: int | None,
    convolutional_layers: int | None,
    git_root: Path,
) -> Path:
    """Create the immutable metadata for one run. Official and synthetic runs both use it."""
    if data_source == "official":
        registered = require_registered_schedule(config)
        if registered != schedule:
            raise TrainingError("official runs cannot store a shortened schedule")
        if batch_size != config.training.batch_size:
            raise TrainingError("official runs use the configured batch size")
    metadata = {
        "study": config.study,
        "experiment": config.experiment,
        "data_source": data_source,
        "seed": config.training.seed,
        "batch_size": batch_size,
        "output_classes": output_classes,
        "architecture": config.model.architecture,
        "trainable_parameters": trainable_parameters,
        "convolutional_layers": convolutional_layers,
        "schedule": {
            "initial_learning_rate": schedule.initial_learning_rate,
            "decay_factor": schedule.decay_factor,
            "milestones": list(schedule.milestones),
            "epochs": schedule.epochs,
        },
        "cache_key": cache_key_hex,
        "source_fingerprint": source_fingerprint,
        "keras_kernel_l2": contracts.KERAS_KERNEL_L2,
        "effective_l2": contracts.L2_COEFFICIENT,
        "shuffle_policy": SHUFFLE_POLICY,
        "reproducibility": DETERMINISM_NOTE,
        "device": device,
        "git": git_state(git_root),
        "runtime": runtime_environment(),
    }
    return create_run(Path(run_root), metadata, config_text)


def fit_session(
    model,
    images: np.ndarray,
    labels: np.ndarray,
    run_dir: Path,
    schedule: EpochSchedule,
    *,
    batch_size: int,
    device: dict | None = None,
    stop_after_completed_epochs: int | None = None,
    extra_callbacks: list | None = None,
    verbose: int = 0,
) -> dict:
    """Train until the schedule finishes or an epoch-boundary stop is requested."""
    metadata = read_metadata(run_dir)
    _require_recorded_schedule(metadata, schedule)
    _require_one_hot(labels, int(metadata["output_classes"]))
    if images.shape[0] != labels.shape[0]:
        raise TrainingError("training images and labels have different lengths")
    if images.dtype != np.float32 or labels.dtype != np.float32:
        raise TrainingError("training arrays must stay float32")
    if int(metadata["batch_size"]) != int(batch_size) or batch_size < 1:
        raise TrainingError("batch size does not match the run metadata")
    if not getattr(model, "compiled", False):
        raise TrainingError("model must be compiled before training")
    status = read_status(run_dir)
    if status["status"] == "completed":
        raise TrainingError("run is already completed and cannot be resumed")
    if status["status"] == "failed":
        raise TrainingError("run failed and cannot be resumed")
    recovered = restore_checkpoint(model, run_dir)
    reconcile_history(run_dir, recovered, schedule.rate)
    if recovered > schedule.epochs:
        raise TrainingError(
            f"backup epoch {recovered} exceeds the scheduled {schedule.epochs} epochs"
        )
    open_session(run_dir, device)
    if recovered == schedule.epochs:
        close_session(run_dir, "running")
        return {"training_complete": True, "already_complete": True, "restored_epochs": recovered}
    callbacks = make_training_callbacks(
        run_dir,
        schedule.rate,
        stop_after_completed_epochs=stop_after_completed_epochs,
        extra_callbacks=extra_callbacks,
    )
    try:
        model.fit(
            images,
            labels,
            batch_size=batch_size,
            epochs=schedule.epochs,
            initial_epoch=recovered,
            shuffle=True,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=None,
        )
    except TrainingDivergence as exc:
        close_session(run_dir, "failed", str(exc))
        raise
    except (TrainingInterrupted, KeyboardInterrupt) as exc:
        close_session(run_dir, "interrupted", str(exc))
        raise
    except Exception as exc:
        close_session(
            run_dir,
            "interrupted" if _has_recovery_progress(run_dir) else "failed",
            str(exc),
        )
        raise
    final_recovered = recovered_completed_epochs(run_dir)
    final_rows = reconcile_history(run_dir, final_recovered, schedule.rate)
    if final_recovered != schedule.epochs or len(final_rows) != schedule.epochs:
        close_session(
            run_dir,
            "interrupted",
            "fit returned before every scheduled epoch was durable",
        )
        raise TrainingError(
            f"durable epochs {final_recovered} and history rows {len(final_rows)} "
            f"do not match the scheduled {schedule.epochs}"
        )
    close_session(run_dir, "running")
    return {"training_complete": True, "already_complete": False, "restored_epochs": recovered}


def restore_training_state(model, run_dir: Path) -> int:
    """Load the newest consistent checkpoint, including optimizer slots."""
    epoch = restore_checkpoint(model, run_dir)
    if epoch == 0:
        raise TrainingError("there is no recovery backup to restore")
    return epoch


def publish_final_model(model, run_dir: Path, *, restore: bool) -> None:
    """Publish ``model.keras`` only when the scheduled epochs are durable."""
    metadata = read_metadata(run_dir)
    status = read_status(run_dir)
    if status["status"] == "completed":
        raise TrainingError("run is already completed")
    if status["status"] == "failed":
        raise TrainingError("a failed run cannot publish a final model")
    epochs = int(metadata["schedule"]["epochs"])
    if restore:
        recovered = restore_training_state(model, run_dir)
    else:
        recovered = recovered_completed_epochs(run_dir)
    if recovered != epochs:
        raise TrainingError(
            f"refusing to publish a final model at completed epoch {recovered}; "
            f"the schedule has {epochs} epochs"
        )
    if len(load_history(run_dir)) != epochs:
        raise TrainingError("history does not cover every scheduled epoch")
    partial = Path(run_dir) / ".model.partial.keras"
    final = Path(run_dir) / "model.keras"
    try:
        if partial.exists():
            partial.unlink()
        model.save(partial)
        _verify_saved_model(partial, metadata)
        os.replace(partial, final)
        _verify_saved_model(final, metadata)
        model_sha256 = _sha256_file(final)
        history_path = Path(run_dir) / "history.csv"
        if not history_path.is_file():
            raise TrainingError("history.csv is missing, so the final model cannot be published")
        history_sha256 = _sha256_file(history_path)
    except TrainingError:
        update_status(
            run_dir,
            status="interrupted",
            interruption="final model publication failed",
        )
        raise
    except Exception as exc:
        update_status(
            run_dir,
            status="interrupted",
            interruption=f"final model publication failed: {exc}",
        )
        raise TrainingError(f"final model publication failed: {exc}") from exc
    close_session(
        run_dir,
        "completed",
        final_model_sha256=model_sha256,
        history_sha256=history_sha256,
    )


def execute_run(
    model,
    images: np.ndarray,
    labels: np.ndarray,
    run_dir: Path,
    schedule: EpochSchedule,
    *,
    batch_size: int,
    device: dict | None = None,
    stop_after_completed_epochs: int | None = None,
    extra_callbacks: list | None = None,
    verbose: int = 0,
) -> dict:
    """Train, then publish the final model when the schedule is complete."""
    outcome = fit_session(
        model,
        images,
        labels,
        run_dir,
        schedule,
        batch_size=batch_size,
        device=device,
        stop_after_completed_epochs=stop_after_completed_epochs,
        extra_callbacks=extra_callbacks,
        verbose=verbose,
    )
    if outcome["training_complete"]:
        publish_final_model(model, run_dir, restore=outcome["already_complete"])
    return outcome


def train_command(
    *,
    config_path: Path | None,
    resume: Path | None,
    run_root: Path,
    cache_root: Path,
    device_name: str,
    git_root: Path,
) -> dict:
    """Start a production run or resume one. Never reads the test split."""
    device = select_device(device_name)
    if resume is not None:
        return _resume_official(
            _existing_run(resume, run_root),
            config_path,
            cache_root,
            device,
            git_root,
        )
    if config_path is None:
        raise TrainingError("provide --config to start a run or --resume to continue one")
    return _start_official(Path(config_path), run_root, cache_root, device, git_root)


def runtime_environment() -> dict:
    import keras
    import tensorflow as tf

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "backend": keras.config.backend(),
        "deterministic_ops": False,
    }


def _start_official(config_path: Path, run_root: Path, cache_root: Path, device: dict, git_root: Path) -> dict:
    config = load_config(config_path)
    schedule = require_registered_schedule(config)
    cache_dir, fingerprint = resolve_official_cache(config, cache_root)
    images, labels = load_official_training(cache_dir)
    seed_runtime(config.training.seed)
    model = build_allcnn(config)
    compile_study_model(model, config.training)
    report = architecture_report(model)
    _require_reference(config.experiment, report)
    run_dir = create_training_run(
        run_root,
        config=config,
        config_text=config_path.read_text(encoding="utf-8"),
        schedule=schedule,
        data_source="official",
        batch_size=config.training.batch_size,
        output_classes=config.dataset.num_classes,
        device=device,
        cache_key_hex=cache_dir.name,
        source_fingerprint=fingerprint,
        trainable_parameters=report.trainable_parameters,
        convolutional_layers=report.convolutional_layers,
        git_root=git_root,
    )
    (run_dir / "model_summary.txt").write_text(format_architecture_report(report), encoding="utf-8")
    try:
        execute_run(
            model,
            images,
            labels,
            run_dir,
            schedule,
            batch_size=config.training.batch_size,
            device=device,
            verbose=2,
        )
    except TrainingInterrupted:
        raise
    return _result(run_dir, device)


def _resume_official(
    run_dir: Path,
    config_path: Path | None,
    cache_root: Path,
    device: dict,
    git_root: Path,
) -> dict:
    del git_root
    metadata = read_metadata(run_dir)
    if metadata.get("data_source") != "official":
        raise TrainingError("refusing to resume a synthetic run as a production experiment")
    status = read_status(run_dir)
    if status["status"] == "completed":
        raise TrainingError("run is already completed and cannot be resumed")
    if status["status"] == "failed":
        raise TrainingError("run failed and cannot be resumed")
    snapshot_path = run_dir / "config.yaml"
    snapshot_bytes = snapshot_path.read_bytes()
    digest = hashlib.sha256(snapshot_bytes).hexdigest()
    if digest != metadata.get("config_sha256"):
        raise TrainingError("the saved config snapshot no longer matches run metadata")
    snapshot = load_config(snapshot_path)
    if snapshot.experiment != metadata["experiment"]:
        raise TrainingError("run metadata experiment does not match the config snapshot")
    if config_path is not None:
        provided = load_config(config_path)
        paths = differing_paths(snapshot, provided)
        if paths:
            raise TrainingError(
                "resume config does not match the original snapshot: " + ", ".join(paths)
            )
    schedule = require_registered_schedule(snapshot)
    recorded = metadata["schedule"]
    if (
        schedule.epochs != int(recorded["epochs"])
        or tuple(schedule.milestones) != tuple(recorded["milestones"])
    ):
        raise TrainingError("the recorded training schedule does not match the registered experiment")
    _require_backend(metadata)
    cache_dir = Path(cache_root) / str(metadata["cache_key"])
    if not cache_dir.is_dir():
        raise TrainingError(f"preprocessing cache {cache_dir.name} is not under {cache_root}")
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_fingerprint") != metadata.get("source_fingerprint"):
        raise TrainingError("preprocessing fingerprint does not match the run")
    if manifest.get("identity") != preprocessing_identity(snapshot):
        raise TrainingError("preprocessing identity does not match the run")
    if cache_dir.name != cache_key(manifest["identity"], manifest["source_fingerprint"]):
        raise TrainingError("cache directory is not the published cache key")
    images, labels = load_official_training(cache_dir)
    seed_runtime(snapshot.training.seed)
    model = build_allcnn(snapshot)
    compile_study_model(model, snapshot.training)
    report = architecture_report(model)
    _require_reference(snapshot.experiment, report)
    if report.trainable_parameters != metadata.get("trainable_parameters"):
        raise TrainingError("rebuilt model parameter count does not match the run")
    if report.convolutional_layers != metadata.get("convolutional_layers"):
        raise TrainingError("rebuilt architecture does not match the run")
    try:
        execute_run(
            model,
            images,
            labels,
            run_dir,
            schedule,
            batch_size=snapshot.training.batch_size,
            device=device,
            verbose=2,
        )
    except TrainingInterrupted:
        raise
    return _result(run_dir, device)


def _result(run_dir: Path, device: dict) -> dict:
    metadata = read_metadata(run_dir)
    status = read_status(run_dir)
    return {
        "run_dir": str(run_dir),
        "run_id": metadata["run_id"],
        "experiment": metadata["experiment"],
        "status": status["status"],
        "completed_epochs": status["completed_epochs"],
        "epoch_seconds_total": status["epoch_seconds_total"],
        "cache_key": metadata.get("cache_key"),
        "device": device["selected_type"],
        "device_name": device["selected_name"],
    }


def _existing_run(resume: Path, run_root: Path) -> Path:
    candidate = Path(resume)
    if candidate.is_dir():
        return candidate
    joined = Path(run_root) / resume
    if joined.is_dir():
        return joined
    raise TrainingError(f"run directory {resume} does not exist")


def _require_reference(experiment: str, report) -> None:
    expected = _REFERENCE[experiment]
    actual = (report.convolutional_layers, report.trainable_parameters, report.macs_per_image)
    if actual != expected:
        raise TrainingError(
            f"{experiment} architecture measurement {actual} does not match the reference {expected}"
        )


def _require_recorded_schedule(metadata: dict, schedule: EpochSchedule) -> None:
    recorded = metadata.get("schedule")
    if type(recorded) is not dict:
        raise TrainingError("run metadata has no schedule")
    if int(recorded["epochs"]) != schedule.epochs:
        raise TrainingError("epoch budget does not match the run metadata")
    if tuple(int(item) for item in recorded["milestones"]) != tuple(schedule.milestones):
        raise TrainingError("learning-rate milestones do not match the run metadata")
    if float(recorded["initial_learning_rate"]) != float(schedule.initial_learning_rate):
        raise TrainingError("initial learning rate does not match the run metadata")
    if float(recorded["decay_factor"]) != float(schedule.decay_factor):
        raise TrainingError("learning-rate decay does not match the run metadata")
    if metadata.get("data_source") == "official":
        require_registered_schedule_values(metadata["experiment"], schedule)


def require_registered_schedule_values(experiment: str, schedule: EpochSchedule) -> None:
    spec = contracts.EXPERIMENT_SPECS.get(experiment)
    if spec is None:
        raise TrainingError(f"unknown experiment {experiment!r}")
    if schedule.epochs != spec.epochs or tuple(schedule.milestones) != spec.lr_milestones:
        raise TrainingError("official run refuses an unregistered schedule")
    if schedule.initial_learning_rate != contracts.INITIAL_LEARNING_RATE:
        raise TrainingError("official run refuses an unregistered initial learning rate")
    if schedule.decay_factor != contracts.LR_DECAY_FACTOR:
        raise TrainingError("official run refuses an unregistered decay factor")


def _require_one_hot(labels: np.ndarray, classes: int) -> None:
    if labels.ndim != 2 or labels.shape[1] != classes:
        raise TrainingError(
            f"training labels must be one-hot with {classes} columns, got shape {labels.shape}"
        )
    if labels.dtype != np.float32:
        raise TrainingError(f"one-hot labels must be float32, got {labels.dtype}")
    if np.any((labels != 0.0) & (labels != 1.0)):
        raise TrainingError("one-hot labels must contain only 0 and 1")
    if not np.allclose(labels.sum(axis=1), 1.0):
        raise TrainingError("each one-hot label must contain a single class")


def _require_backend(metadata: dict) -> None:
    import tensorflow as tf
    from keras import config as keras_config

    runtime = metadata.get("runtime") or {}
    if runtime.get("backend") != "tensorflow" or keras_config.backend() != "tensorflow":
        raise TrainingError("resume requires the TensorFlow Keras backend recorded for the run")
    recorded = str(runtime.get("tensorflow", ""))
    if recorded.split(".", 1)[0] != tf.__version__.split(".", 1)[0]:
        raise TrainingError(
            f"TensorFlow {tf.__version__} is not compatible with the run's TensorFlow {recorded}"
        )


def _has_recovery_progress(run_dir: Path) -> bool:
    try:
        if recovered_completed_epochs(run_dir) > 0:
            return True
    except TrainingError:
        return True
    history = run_dir / "history.csv"
    return history.exists() and history.stat().st_size > 0


def _preflight(tf, logical_name: str, selected_type: str) -> str:
    with tf.device(logical_name):
        value = tf.reduce_sum(tf.ones((2, 2), dtype=tf.float32))
    observed = str(value.device)
    if selected_type == "GPU" and "GPU" not in observed.upper():
        raise TrainingError(f"GPU was selected but the preflight op ran on {observed}")
    if selected_type == "CPU" and "GPU" in observed.upper():
        raise TrainingError(f"CPU was selected but the preflight op ran on {observed}")
    return observed


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _verify_saved_model(path: Path, metadata: dict) -> None:
    from keras.layers import Conv2D
    from keras.saving import load_model

    loaded = load_model(path)
    classes = int(loaded.output.shape[-1])
    if classes != int(metadata["output_classes"]):
        raise TrainingError(
            f"saved model has {classes} outputs, expected {metadata['output_classes']}"
        )
    if not loaded.built:
        raise TrainingError("saved model is not built")
    expected = metadata.get("convolutional_layers")
    if expected is not None:
        count = sum(isinstance(layer, Conv2D) for layer in loaded.layers)
        if count != int(expected):
            raise TrainingError(f"saved model has {count} convolutions, expected {expected}")
