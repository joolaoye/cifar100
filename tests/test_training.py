"""Training schedule, regularization, run safety, and interruption recovery."""

import ast
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cifar100_study import contracts
from cifar100_study.config import load_config
from cifar100_study.runs import (
    TrainingDivergence,
    TrainingError,
    TrainingInterrupted,
    append_history_row,
    create_run,
    git_state,
    load_history,
    read_metadata,
    read_status,
    reconcile_history,
    restore_checkpoint,
    save_epoch_checkpoint,
    update_status,
    valid_checkpoints,
)
from cifar100_study.trainer import (
    EpochSchedule,
    _resume_official,
    compile_study_model,
    create_training_run,
    execute_run,
    fit_session,
    publish_final_model,
    require_registered_schedule,
    scheduled_learning_rate,
    select_device,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _e0():
    return load_config(CONFIG_DIR / "e0.yaml")


def _schedule(epochs=5, milestones=(2,)) -> EpochSchedule:
    return EpochSchedule(0.05, 0.1, tuple(milestones), epochs)


def _run(tmp_path, epochs=5, milestones=(2,)) -> Path:
    schedule = _schedule(epochs, milestones)
    return create_run(
        tmp_path,
        {
            "experiment": "e0",
            "data_source": "synthetic",
            "schedule": {
                "initial_learning_rate": schedule.initial_learning_rate,
                "decay_factor": schedule.decay_factor,
                "milestones": list(schedule.milestones),
                "epochs": schedule.epochs,
            },
            "batch_size": 8,
        },
        (CONFIG_DIR / "e0.yaml").read_text(encoding="utf-8"),
    )


def _row(epoch: int, learning_rate: float, seconds: float = 0.25) -> dict:
    return {
        "epoch": epoch,
        "loss": 1.0 + epoch / 10.0,
        "accuracy": 0.1 * epoch,
        "learning_rate": learning_rate,
        "epoch_seconds": seconds,
    }


def _write_backup(run_dir: Path, epoch: int) -> None:
    backup = run_dir / "backup"
    backup.mkdir(exist_ok=True)
    (backup / "latest.weights.h5").write_bytes(b"not-loaded-in-this-test")
    (backup / "training_metadata.json").write_text(
        json.dumps({"epoch": epoch, "batch": 0}),
        encoding="utf-8",
    )


def _tiny_training(tmp_path, epochs=1, milestones=()):
    from keras import Input, Model, layers

    config = _e0()
    schedule = _schedule(epochs, milestones)
    images = np.linspace(-1.0, 1.0, 32 * 4, dtype=np.float32).reshape(32, 4)
    labels = np.zeros((32, 2), dtype=np.float32)
    labels[np.arange(32), np.arange(32) % 2] = 1.0
    inputs = Input((4,), name="images")
    hidden = layers.Dense(3, activation="relu", name="hidden")(inputs)
    outputs = layers.Dense(2, activation="softmax", name="probs")(hidden)
    model = Model(inputs, outputs, name="tiny_test")
    compile_study_model(model, config.training)
    device = {"requested": "cpu", "selected_type": "CPU", "selected_name": "cpu"}
    run_dir = create_training_run(
        tmp_path,
        config=config,
        config_text=(CONFIG_DIR / "e0.yaml").read_text(encoding="utf-8"),
        schedule=schedule,
        data_source="synthetic",
        batch_size=8,
        output_classes=2,
        device=device,
        cache_key_hex=None,
        source_fingerprint=None,
        trainable_parameters=None,
        convolutional_layers=0,
        git_root=ROOT,
    )
    return model, images, labels, run_dir, schedule, device


def _new_tiny():
    from keras import Input, Model, layers

    inputs = Input((4,), name="images")
    hidden = layers.Dense(3, activation="relu", name="hidden")(inputs)
    outputs = layers.Dense(2, activation="softmax", name="probs")(hidden)
    model = Model(inputs, outputs, name="tiny_test")
    compile_study_model(model, _e0().training)
    return model


def _iterations(model) -> int:
    value = model.optimizer.iterations
    if hasattr(value, "numpy"):
        value = value.numpy()
    return int(value)


def _momentum(model) -> dict:
    slots = {}
    for variable in model.optimizer.variables:
        if "momentum" in variable.name:
            slots[variable.name] = np.asarray(variable.numpy())
    return slots


def _generations(run_dir: Path) -> dict[int, Path]:
    found = {}
    for child in (run_dir / "backup").iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        state = json.loads((child / "state.json").read_text(encoding="utf-8"))
        found[int(state["epoch"])] = child
    return found


def _assert_has_checkpoint(run_dir: Path) -> None:
    assert (run_dir / "backup" / "manifest.json").is_file()
    assert list((run_dir / "backup").glob("g-*/model.weights.h5"))


def test_learning_rate_boundaries_for_e0_and_e1():
    e0 = (0.05, 0.1, (200, 250, 300))
    e1 = (0.05, 0.1, (400, 500, 600))
    expected_e0 = {
        0: 0.05,
        199: 0.05,
        200: 0.005,
        249: 0.005,
        250: 0.0005,
        299: 0.0005,
        300: 0.00005,
        349: 0.00005,
    }
    expected_e1 = {
        0: 0.05,
        399: 0.05,
        400: 0.005,
        499: 0.005,
        500: 0.0005,
        599: 0.0005,
        600: 0.00005,
        699: 0.00005,
    }
    for epoch, rate in expected_e0.items():
        assert scheduled_learning_rate(
            epoch, initial_learning_rate=e0[0], decay_factor=e0[1], milestones=e0[2]
        ) == pytest.approx(rate)
    for epoch, rate in expected_e1.items():
        assert scheduled_learning_rate(
            epoch, initial_learning_rate=e1[0], decay_factor=e1[1], milestones=e1[2]
        ) == pytest.approx(rate)
    for name in ("e0", "e1", "e2"):
        config = load_config(CONFIG_DIR / f"{name}.yaml")
        schedule = require_registered_schedule(config)
        assert schedule.epochs == contracts.EXPERIMENT_SPECS[name].epochs
        assert schedule.milestones == contracts.EXPERIMENT_SPECS[name].lr_milestones
    tweaked = replace(_e0(), training=replace(_e0().training, epochs=351))
    with pytest.raises(TrainingError, match="unregistered|refuses schedule"):
        require_registered_schedule(tweaked)


def test_regularizer_gradient_matches_coupled_weight_decay():
    import tensorflow as tf
    from keras import regularizers

    weights = tf.Variable([[1.0, -2.0], [0.5, 0.25]], dtype=tf.float64)
    with tf.GradientTape() as tape:
        penalty = regularizers.L2(contracts.KERAS_KERNEL_L2)(weights)
    gradient = tape.gradient(penalty, weights)
    np.testing.assert_allclose(gradient.numpy(), contracts.L2_COEFFICIENT * weights.numpy())


def test_compiled_loss_includes_each_kernel_penalty_once():
    import tensorflow as tf
    from keras import Input, Model, layers, losses, regularizers

    inputs = Input((4, 4, 1))
    conv = layers.Conv2D(
        1,
        1,
        use_bias=True,
        activation="linear",
        kernel_regularizer=regularizers.L2(contracts.KERAS_KERNEL_L2),
        kernel_initializer="ones",
        bias_initializer="zeros",
        name="regularized",
    )(inputs)
    pooled = layers.GlobalAveragePooling2D()(conv)
    outputs = layers.Dense(2, activation="softmax", use_bias=False, kernel_initializer="zeros")(pooled)
    model = Model(inputs, outputs)
    compile_study_model(model, _e0().training)
    optimizer = model.optimizer
    assert type(optimizer).__name__ == "SGD"
    assert optimizer.momentum == contracts.MOMENTUM
    assert optimizer.nesterov is False
    assert optimizer.weight_decay is None
    assert optimizer.clipnorm is None
    assert optimizer.use_ema is False
    assert optimizer.gradient_accumulation_steps is None
    assert type(model.loss).__name__ == "CategoricalCrossentropy"
    user_metrics = list(model._compile_metrics._user_metrics)
    assert [type(metric).__name__ for metric in user_metrics] == ["CategoricalAccuracy"]
    assert [metric.name for metric in user_metrics] == ["accuracy"]
    regularized = model.get_layer("regularized")
    assert regularized.bias_regularizer is None
    assert regularized.activity_regularizer is None
    images = tf.ones((4, 4, 4, 1))
    labels = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    predictions = model(images, training=False)
    penalties = list(model.losses)
    assert len(penalties) == 1
    kernel_size = int(tf.size(regularized.kernel))
    assert float(penalties[0]) == pytest.approx(contracts.KERAS_KERNEL_L2 * kernel_size)
    cross_entropy = float(tf.reduce_mean(losses.categorical_crossentropy(labels, predictions)))
    total = float(model.compute_loss(images, labels, predictions))
    assert total == pytest.approx(cross_entropy + float(penalties[0]))
    assert len(list(model.losses)) == 1


def test_sparse_labels_are_rejected(tmp_path):
    model, images, _labels, run_dir, schedule, device = _tiny_training(tmp_path)
    sparse = np.arange(images.shape[0], dtype=np.int32)
    with pytest.raises(TrainingError, match="one-hot"):
        fit_session(model, images, sparse, run_dir, schedule, batch_size=8, device=device)


def test_run_directories_are_unique_and_existing_runs_stay(tmp_path):
    first = _run(tmp_path)
    second = _run(tmp_path)
    assert first != second
    assert first.is_dir() and second.is_dir()
    (first / "marker.txt").write_text("keep", encoding="utf-8")
    _run(tmp_path)
    assert (first / "marker.txt").read_text(encoding="utf-8") == "keep"
    for name in ("trainer.py", "runs.py", "training_callbacks.py", "train.py"):
        path = ROOT / "src" / "cifar100_study" / name
        if name == "train.py":
            path = ROOT / "scripts" / "train.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        source = path.read_text(encoding="utf-8")
        assert "rmtree" not in source
        assert "save_best_only" not in source
        assert "EarlyStopping" not in source
        assert "load_test_data" not in source
        assert "validation_split=" not in source
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"rmtree", "removedirs"}
            for node in ast.walk(tree)
        )
    callbacks = (ROOT / "src" / "cifar100_study" / "training_callbacks.py").read_text(encoding="utf-8")
    assert "TerminateOnNaN" in callbacks
    assert "save_epoch_checkpoint" in callbacks
    assert "BackupAndRestore" not in callbacks


def test_git_state_allows_a_repository_without_commits(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    state = git_state(tmp_path)
    assert state["has_commit"] is False
    assert state["commit"] is None
    assert state["dirty"] is False
    assert state["detail"] == "repository has no commits"


def test_history_reconciliation_drops_only_a_durable_suffix(tmp_path):
    run_dir = _run(tmp_path)
    schedule = _schedule()
    append_history_row(run_dir, _row(1, 0.05, 1.5))
    append_history_row(run_dir, _row(2, 0.05, 2.25))
    append_history_row(run_dir, _row(3, 0.005, 0.5))
    _write_backup(run_dir, 2)
    rows = reconcile_history(run_dir, 2, schedule.rate)
    assert [row["epoch"] for row in rows] == [1, 2]
    assert [row["epoch"] for row in load_history(run_dir)] == [1, 2]
    assert sum(row["epoch_seconds"] for row in rows) == pytest.approx(3.75)
    text = (run_dir / "history.csv").read_text(encoding="utf-8")
    text += "4,1.4,0.4,not-a-number\n"
    (run_dir / "history.csv").write_text(text, encoding="utf-8")
    assert [row["epoch"] for row in load_history(run_dir)] == [1, 2]
    assert "not-a-number" not in (run_dir / "history.csv").read_text(encoding="utf-8")
    append_history_row(run_dir, _row(3, 0.005, 0.5))
    assert [row["epoch"] for row in load_history(run_dir)] == [1, 2, 3]


def test_append_keeps_a_valid_row_that_has_no_terminating_newline(tmp_path):
    run_dir = _run(tmp_path)
    schedule = _schedule()
    append_history_row(run_dir, _row(1, 0.05, 1.5))
    original = load_history(run_dir)[0]
    path = run_dir / "history.csv"
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    path.write_text(text[:-1], encoding="utf-8")
    reconciled = reconcile_history(run_dir, 1, schedule.rate)
    assert reconciled == [original]
    append_history_row(run_dir, _row(2, 0.05, 2.25))
    loaded = load_history(run_dir)
    assert [row["epoch"] for row in loaded] == [1, 2]
    assert loaded[0] == original
    assert loaded[1]["loss"] == pytest.approx(1.2)
    assert loaded[1]["accuracy"] == pytest.approx(0.2)
    assert loaded[1]["learning_rate"] == pytest.approx(0.05)
    assert loaded[1]["epoch_seconds"] == pytest.approx(2.25)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_retention_keeps_a_newer_generation_missing_from_the_manifest(tmp_path):
    model, _images, _labels, run_dir, _schedule_value, _device = _tiny_training(tmp_path, epochs=2)
    fit_session(
        model,
        _images,
        _labels,
        run_dir,
        _schedule_value,
        batch_size=8,
        device=_device,
    )
    generations = _generations(run_dir)
    assert set(generations) == {1, 2}
    published = run_dir / "backup" / "g-00000003-published"
    shutil.copytree(generations[2], published)
    state_path = published / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["epoch"] = 3
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads((run_dir / "backup" / "manifest.json").read_text(encoding="utf-8"))
    assert published.name not in manifest["generations"]
    save_epoch_checkpoint(model, run_dir, 4)
    retained = _generations(run_dir)
    assert set(retained) == {3, 4}
    manifest = json.loads((run_dir / "backup" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generations"] == [retained[4].name, retained[3].name]


def test_history_rejects_gaps_duplicates_shortfalls_and_bad_rates(tmp_path):
    schedule = _schedule()
    behind = _run(tmp_path / "behind")
    append_history_row(behind, _row(1, 0.05))
    _write_backup(behind, 2)
    with pytest.raises(TrainingError, match="invent"):
        reconcile_history(behind, 2, schedule.rate)

    missing_backup = _run(tmp_path / "missing")
    append_history_row(missing_backup, _row(1, 0.05))
    with pytest.raises(TrainingError, match="backup is missing"):
        reconcile_history(missing_backup, 0, schedule.rate)

    gap = _run(tmp_path / "gap")
    (gap / "history.csv").write_text(
        "epoch,loss,accuracy,learning_rate,epoch_seconds\n"
        "1,1.0,0.1,0.05,0.100000\n"
        "2,1.0,0.1,0.05,0.100000\n"
        "4,1.0,0.1,0.005,0.100000\n",
        encoding="utf-8",
    )
    with pytest.raises(TrainingError, match="numbering"):
        load_history(gap)

    duplicate = _run(tmp_path / "duplicate")
    (duplicate / "history.csv").write_text(
        "epoch,loss,accuracy,learning_rate,epoch_seconds\n"
        "1,1.0,0.1,0.05,0.100000\n"
        "1,1.0,0.1,0.05,0.100000\n",
        encoding="utf-8",
    )
    with pytest.raises(TrainingError, match="numbering"):
        load_history(duplicate)

    bad_rate = _run(tmp_path / "rate")
    append_history_row(bad_rate, _row(1, 0.01))
    with pytest.raises(TrainingError, match="learning rate"):
        reconcile_history(bad_rate, 1, schedule.rate)


def test_resume_rejects_completed_mismatched_and_synthetic_runs(tmp_path):
    model, images, labels, run_dir, schedule, device = _tiny_training(tmp_path / "done", epochs=1)
    fit_session(model, images, labels, run_dir, schedule, batch_size=8, device=device)
    publish_final_model(model, run_dir, restore=False)
    with pytest.raises(TrainingError, match="already completed"):
        fit_session(model, images, labels, run_dir, schedule, batch_size=8, device=device)

    synthetic = _run(tmp_path / "synthetic")
    with pytest.raises(TrainingError, match="synthetic"):
        _resume_official(
            synthetic,
            CONFIG_DIR / "e0.yaml",
            tmp_path / "cache",
            device,
            ROOT,
        )

    official = create_run(
        tmp_path / "official",
        {
            "experiment": "e0",
            "data_source": "official",
            "schedule": {
                "initial_learning_rate": 0.05,
                "decay_factor": 0.1,
                "milestones": [200, 250, 300],
                "epochs": 350,
            },
            "batch_size": 256,
        },
        (CONFIG_DIR / "e0.yaml").read_text(encoding="utf-8"),
    )
    with pytest.raises(TrainingError, match="does not match"):
        _resume_official(official, CONFIG_DIR / "e1.yaml", tmp_path / "cache", device, ROOT)


def test_failed_publication_keeps_the_backup_and_can_be_retried(tmp_path):
    model, images, labels, run_dir, schedule, device = _tiny_training(tmp_path, epochs=1)
    called = {}
    original = model.fit

    def spy(*args, **kwargs):
        called["kwargs"] = kwargs
        return original(*args, **kwargs)

    model.fit = spy
    fit_session(model, images, labels, run_dir, schedule, batch_size=8, device=device)
    assert called["kwargs"]["validation_data"] is None
    assert called["kwargs"]["shuffle"] is True
    assert "validation_split" not in called["kwargs"]
    _assert_has_checkpoint(run_dir)

    def broken_save(*args, **kwargs):
        raise OSError("disk full")

    model.save = broken_save
    with pytest.raises(TrainingError, match="publication failed"):
        publish_final_model(model, run_dir, restore=False)
    assert read_status(run_dir)["status"] == "interrupted"
    assert not (run_dir / "model.keras").exists()
    _assert_has_checkpoint(run_dir)
    del model.save
    publish_final_model(model, run_dir, restore=False)
    assert read_status(run_dir)["status"] == "completed"
    assert (run_dir / "model.keras").is_file()
    _assert_has_checkpoint(run_dir)


def test_publication_can_restore_a_finished_backup_into_a_fresh_model(tmp_path):
    trained, images, labels, run_dir, schedule, device = _tiny_training(tmp_path, epochs=1)
    fit_session(trained, images, labels, run_dir, schedule, batch_size=8, device=device)
    assert read_status(run_dir)["status"] == "running"
    expected = [np.asarray(weight.numpy()) for weight in trained.weights]
    from keras import Input, Model, layers

    inputs = Input((4,), name="images")
    hidden = layers.Dense(3, activation="relu", name="hidden")(inputs)
    outputs = layers.Dense(2, activation="softmax", name="probs")(hidden)
    fresh = Model(inputs, outputs, name="tiny_test")
    compile_study_model(fresh, _e0().training)
    for weight in fresh.weights:
        weight.assign(np.ones(tuple(weight.shape), dtype=np.float32))
    publish_final_model(fresh, run_dir, restore=True)
    from keras.saving import load_model

    loaded = load_model(run_dir / "model.keras")
    for actual, target in zip(loaded.weights, expected):
        np.testing.assert_allclose(np.asarray(actual.numpy()), target)
    assert read_status(run_dir)["status"] == "completed"
    _assert_has_checkpoint(run_dir)


def test_short_training_records_the_milestone_learning_rate(tmp_path):
    model, images, labels, run_dir, schedule, device = _tiny_training(tmp_path, epochs=3, milestones=(2,))
    fit_session(model, images, labels, run_dir, schedule, batch_size=8, device=device)
    rows = load_history(run_dir)
    assert [row["epoch"] for row in rows] == [1, 2, 3]
    assert [row["learning_rate"] for row in rows] == pytest.approx([0.05, 0.05, 0.005])
    assert int(model.optimizer.iterations) == 12
    assert read_status(run_dir)["epoch_seconds_total"] == pytest.approx(sum(row["epoch_seconds"] for row in rows))
    metadata = read_metadata(run_dir)
    assert metadata["git"] == git_state(ROOT)
    assert metadata["reproducibility"]


def test_nonfinite_loss_fails_the_run_without_recording_the_epoch(tmp_path):
    from keras import Input, Model, callbacks, layers

    _, images, labels, run_dir, schedule, device = _tiny_training(tmp_path, epochs=3)

    class FiniteGate(layers.Layer):
        """Scale logits elementwise. A fused matmul can drop a NaN kernel."""

        def __init__(self):
            super().__init__(name="finite_gate")
            self.scale = self.add_weight(
                name="scale",
                shape=(),
                initializer="ones",
                trainable=False,
            )

        def call(self, inputs):
            return inputs * self.scale

    def build():
        inputs = Input((4,), name="images")
        hidden = layers.Dense(3, activation="relu", name="hidden")(inputs)
        logits = layers.Dense(2, name="probs")(hidden)
        outputs = layers.Activation("softmax")(FiniteGate()(logits))
        model = Model(inputs, outputs, name="tiny_test")
        compile_study_model(model, _e0().training)
        return model

    model = build()

    class DivergeOnSecondEpoch(callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if int(epoch) != 1:
                return
            self.model.get_layer("finite_gate").scale.assign(np.float32("nan"))

    with pytest.raises(TrainingDivergence, match="epoch 2"):
        execute_run(
            model,
            images,
            labels,
            run_dir,
            schedule,
            batch_size=8,
            device=device,
            extra_callbacks=[DivergeOnSecondEpoch()],
        )
    status = read_status(run_dir)
    assert status["status"] == "failed"
    assert "epoch 2" in status["interruption"]
    assert not (run_dir / "model.keras").exists()
    rows = load_history(run_dir)
    assert [row["epoch"] for row in rows] == [1]
    assert all(math.isfinite(row["loss"]) for row in rows)
    checkpoints = valid_checkpoints(run_dir)
    assert [item["epoch"] for item in checkpoints] == [1]
    assert checkpoints[0]["optimizer_iterations"] == 4
    restored = build()
    assert restore_checkpoint(restored, run_dir) == 1
    assert int(restored.optimizer.iterations) == 4
    with pytest.raises(TrainingError, match="failed run cannot publish"):
        publish_final_model(restored, run_dir, restore=True)


def test_interruption_and_resume_across_processes(tmp_path):
    images = np.linspace(-1.0, 1.0, 32 * 4, dtype=np.float32).reshape(32, 4)
    labels = np.zeros((32, 2), dtype=np.float32)
    labels[np.arange(32), np.arange(32) % 2] = 1.0
    data_path = tmp_path / "synthetic.npz"
    np.savez(data_path, images=images, labels=labels)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    probe = ROOT / "tests" / "resume_probe.py"

    def run(phase: str, code_ok: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(probe), phase, str(tmp_path), str(data_path)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    started = run("start", 2)
    assert started.returncode == 2, started.stderr
    run_dir = Path((tmp_path / "run_path.txt").read_text(encoding="utf-8"))
    before = (run_dir / "history.csv").read_text(encoding="utf-8")
    assert read_status(run_dir)["status"] == "interrupted"
    assert [row["epoch"] for row in load_history(run_dir)] == [1, 2, 3]
    assert [row["learning_rate"] for row in load_history(run_dir)] == pytest.approx([0.05, 0.05, 0.005])
    first_seconds = sum(row["epoch_seconds"] for row in load_history(run_dir))

    resumed = run("resume", 0)
    assert resumed.returncode == 0, resumed.stderr
    rows = load_history(run_dir)
    assert [row["epoch"] for row in rows] == [1, 2, 3, 4, 5]
    assert len({row["epoch"] for row in rows}) == 5
    assert [row["learning_rate"] for row in rows] == pytest.approx([0.05, 0.05, 0.005, 0.005, 0.005])
    after = (run_dir / "history.csv").read_text(encoding="utf-8")
    assert after.startswith(before)
    assert read_status(run_dir)["status"] == "completed"
    assert read_status(run_dir)["epoch_seconds_total"] == pytest.approx(sum(row["epoch_seconds"] for row in rows))
    assert read_status(run_dir)["epoch_seconds_total"] > first_seconds
    assert (run_dir / "model.keras").is_file()
    _assert_has_checkpoint(run_dir)
    saved = json.loads((run_dir / "optimizer_snapshot.json").read_text(encoding="utf-8"))
    restored = json.loads((run_dir / "restored_snapshot.json").read_text(encoding="utf-8"))
    assert set(saved) == set(restored)
    momentum_norm = 0.0
    for name, value in saved.items():
        np.testing.assert_allclose(restored[name], value, rtol=1e-6, atol=1e-6)
        if "momentum" in name:
            momentum_norm += float(np.linalg.norm(value))
    assert momentum_norm > 0.0
    assert float(np.asarray(saved["iteration"])) == pytest.approx(12.0)
    from keras.saving import load_model

    loaded = load_model(run_dir / "model.keras")
    assert int(loaded.optimizer.iterations) == 20
    assert tuple(loaded.output.shape) == (None, 2)

    continuous = run("continuous", 0)
    assert continuous.returncode == 0, continuous.stderr
    continuous_rates = json.loads((tmp_path / "continuous_rates.json").read_text(encoding="utf-8"))
    assert continuous_rates == pytest.approx([row["learning_rate"] for row in rows])


def test_device_selection_reports_the_device_that_ran_the_preflight():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    def run(script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    cpu = run(
        "from cifar100_study.trainer import select_device\n"
        "cpu = select_device('cpu')\n"
        "assert cpu['selected_type'] == 'CPU', cpu\n"
        "assert 'GPU' not in cpu['preflight_device'].upper(), cpu\n"
    )
    assert cpu.returncode == 0, cpu.stderr
    detected = run(
        "from cifar100_study.trainer import TrainingError, select_device\n"
        "auto = select_device('auto')\n"
        "assert auto['selected_type'] in {'CPU', 'GPU'}, auto\n"
        "if auto['gpu_devices']:\n"
        "    gpu = select_device('gpu')\n"
        "    assert gpu['selected_type'] == 'GPU', gpu\n"
        "    assert 'GPU' in gpu['preflight_device'].upper(), gpu\n"
        "    assert 'GPU' in auto['preflight_device'].upper(), auto\n"
        "else:\n"
        "    assert 'GPU' not in auto['preflight_device'].upper(), auto\n"
        "    try:\n"
        "        select_device('gpu')\n"
        "    except TrainingError as exc:\n"
        "        assert 'gpu' in str(exc).lower()\n"
        "    else:\n"
        "        raise SystemExit('gpu request should fail when no GPU is visible')\n"
    )
    assert detected.returncode == 0, detected.stderr


def test_training_does_not_request_test_arrays(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("load_test_data was called")

    monkeypatch.setattr("cifar100_study.data.load_test_data", forbidden)
    model, images, labels, run_dir, schedule, device = _tiny_training(tmp_path, epochs=1)
    fit_session(model, images, labels, run_dir, schedule, batch_size=8, device=device)
    assert math.isfinite(load_history(run_dir)[0]["loss"])


def test_production_cache_resolution_rejects_a_synthetic_cache(tmp_path):
    from cifar100_study.data import prepare_cache, preprocessing_identity
    from cifar100_study.trainer import resolve_official_cache

    config = _e0()
    identity = preprocessing_identity(config)
    identity = dict(identity)
    identity["train_size"] = 4
    identity["test_size"] = 2
    identity["image_shape"] = [2, 2, 3]
    images = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)
    test = images[:2]
    labels = np.arange(4, dtype=np.int64) % 100
    test_labels = labels[:2]
    prepare_cache(identity, images, labels, test, test_labels, tmp_path)
    with pytest.raises(TrainingError, match="synthetic|no official"):
        resolve_official_cache(config, tmp_path)


def test_torn_checkpoint_weights_do_not_receive_extra_updates(tmp_path):
    model, images, labels, run_dir, schedule, device = _tiny_training(
        tmp_path, epochs=3, milestones=(2,)
    )
    with pytest.raises(TrainingInterrupted, match="stopped after 2"):
        fit_session(
            model,
            images,
            labels,
            run_dir,
            schedule,
            batch_size=8,
            device=device,
            stop_after_completed_epochs=2,
        )
    assert _iterations(model) == 8
    generations = _generations(run_dir)
    assert set(generations) == {1, 2}
    reference = _new_tiny()
    reference.optimizer.build(reference.trainable_variables)
    reference.load_weights(generations[1] / "model.weights.h5")
    expected_momentum = _momentum(reference)
    assert expected_momentum
    assert _iterations(reference) == 4
    kept = load_history(run_dir)[0]

    model.fit(images, labels, epochs=1, batch_size=8, shuffle=False, verbose=0)
    assert _iterations(model) == 12
    model.save_weights(generations[2] / "model.weights.h5")
    staging = run_dir / "backup" / ".staging-torn"
    staging.mkdir()
    model.save_weights(staging / "model.weights.h5")

    fresh = _new_tiny()
    assert restore_checkpoint(fresh, run_dir) == 1
    assert _iterations(fresh) == 4
    for name, value in expected_momentum.items():
        np.testing.assert_allclose(_momentum(fresh)[name], value)
    fit_session(fresh, images, labels, run_dir, schedule, batch_size=8, device=device)
    assert _iterations(fresh) == 12
    rows = load_history(run_dir)
    assert [row["epoch"] for row in rows] == [1, 2, 3]
    assert len({row["epoch"] for row in rows}) == 3
    assert rows[0] == kept
    assert [row["learning_rate"] for row in rows] == pytest.approx([0.05, 0.05, 0.005])
    assert read_status(run_dir)["status"] != "failed"
    assert read_status(run_dir)["completed_epochs"] == 3


def test_corrupt_newest_checkpoint_falls_back_to_the_previous_generation(tmp_path):
    trained, images, labels, run_dir, schedule, device = _tiny_training(
        tmp_path, epochs=3, milestones=(2,)
    )
    fit_session(trained, images, labels, run_dir, schedule, batch_size=8, device=device)
    assert _iterations(trained) == 12
    assert set(_generations(run_dir)) == {2, 3}
    original = load_history(run_dir)
    (_generations(run_dir)[3] / "model.weights.h5").write_bytes(b"corrupt")

    fresh = _new_tiny()
    with pytest.raises(TrainingError, match="refusing to publish"):
        publish_final_model(fresh, run_dir, restore=True)
    assert _iterations(fresh) == 8
    assert not (run_dir / "model.keras").exists()
    assert read_status(run_dir)["status"] not in {"failed", "completed"}

    fit_session(fresh, images, labels, run_dir, schedule, batch_size=8, device=device)
    assert _iterations(fresh) == 12
    rows = load_history(run_dir)
    assert [row["epoch"] for row in rows] == [1, 2, 3]
    assert rows[0] == original[0]
    assert rows[1] == original[1]
    assert [row["learning_rate"] for row in rows] == pytest.approx([0.05, 0.05, 0.005])
    assert set(_generations(run_dir)) == {2, 3}
    publish_final_model(fresh, run_dir, restore=False)
    from keras.saving import load_model

    loaded = load_model(run_dir / "model.keras")
    assert int(loaded.optimizer.iterations) == 12
    for actual, target in zip(loaded.weights, fresh.weights):
        np.testing.assert_allclose(np.asarray(actual.numpy()), np.asarray(target.numpy()))
    assert read_status(run_dir)["status"] == "completed"
    assert read_status(run_dir)["completed_epochs"] == 3


def test_resume_without_a_checkpoint_does_not_restart(tmp_path):
    model, images, labels, run_dir, schedule, device = _tiny_training(tmp_path, epochs=3)
    with pytest.raises(TrainingInterrupted, match="stopped after 2"):
        fit_session(
            model,
            images,
            labels,
            run_dir,
            schedule,
            batch_size=8,
            device=device,
            stop_after_completed_epochs=2,
        )
    history = (run_dir / "history.csv").read_text(encoding="utf-8")
    weights = [np.asarray(weight.numpy()) for weight in model.weights]
    iterations = _iterations(model)
    shutil.rmtree(run_dir / "backup")
    with pytest.raises(TrainingError, match="refusing to restart"):
        fit_session(model, images, labels, run_dir, schedule, batch_size=8, device=device)
    with pytest.raises(TrainingError, match="refusing to restart"):
        fit_session(model, images, labels, run_dir, schedule, batch_size=8, device=device)
    assert (run_dir / "history.csv").read_text(encoding="utf-8") == history
    assert _iterations(model) == iterations
    for actual, expected in zip(model.weights, weights):
        np.testing.assert_array_equal(np.asarray(actual.numpy()), expected)
    status = read_status(run_dir)
    assert status["status"] not in {"failed", "completed"}
    assert status["completed_epochs"] == 2
    assert [row["epoch"] for row in load_history(run_dir)] == [1, 2]
