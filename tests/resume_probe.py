"""Subprocess probe for interruption and uninterrupted synthetic training.

This file is not a pytest module. It exercises the production fit, history,
backup, and publication helpers on a tiny model.
"""

import json
import sys
from pathlib import Path

import numpy as np

from cifar100_study.config import load_config
from cifar100_study.runs import TrainingInterrupted, load_history
from cifar100_study.trainer import (
    EpochSchedule,
    compile_study_model,
    create_training_run,
    fit_session,
    publish_final_model,
    seed_runtime,
    select_device,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = EpochSchedule(0.05, 0.1, (2,), 5)


def build_model():
    from keras import Input, Model, layers

    inputs = Input((4,), name="images")
    hidden = layers.Dense(3, activation="relu", name="hidden")(inputs)
    outputs = layers.Dense(2, activation="softmax", name="probs")(hidden)
    return Model(inputs, outputs, name="tiny_probe")


def optimizer_state(model) -> dict:
    return {
        variable.name: np.asarray(variable.numpy()).tolist()
        for variable in model.optimizer.variables
    }


def make_run(run_root: Path, device: dict):
    config = load_config(ROOT / "configs" / "e0.yaml")
    seed_runtime(config.training.seed)
    model = build_model()
    compile_study_model(model, config.training)
    run_dir = create_training_run(
        run_root,
        config=config,
        config_text=(ROOT / "configs" / "e0.yaml").read_text(encoding="utf-8"),
        schedule=SCHEDULE,
        data_source="synthetic",
        batch_size=8,
        output_classes=2,
        device=device,
        cache_key_hex=None,
        source_fingerprint=None,
        trainable_parameters=int(sum(np.asarray(weight.numpy()).size for weight in model.weights)),
        convolutional_layers=0,
        git_root=ROOT,
    )
    return model, run_dir


def main() -> int:
    phase, run_root_text, data_text = sys.argv[1:]
    run_root = Path(run_root_text)
    payload = np.load(data_text)
    images = payload["images"]
    labels = payload["labels"]
    device = select_device("cpu")
    if phase == "start":
        model, run_dir = make_run(run_root, device)
        (run_root / "run_path.txt").write_text(str(run_dir), encoding="utf-8")
        try:
            fit_session(
                model,
                images,
                labels,
                run_dir,
                SCHEDULE,
                batch_size=8,
                device=device,
                stop_after_completed_epochs=3,
            )
        except TrainingInterrupted:
            (run_dir / "optimizer_snapshot.json").write_text(
                json.dumps(optimizer_state(model)),
                encoding="utf-8",
            )
            return 2
        raise SystemExit("the first process was expected to stop after three epochs")
    if phase == "resume":
        run_dir = Path((run_root / "run_path.txt").read_text(encoding="utf-8"))
        config = load_config(ROOT / "configs" / "e0.yaml")
        seed_runtime(config.training.seed)
        model = build_model()
        compile_study_model(model, config.training)

        from keras import callbacks

        class CaptureRestoredState(callbacks.Callback):
            def on_train_begin(self, logs=None):
                (run_dir / "restored_snapshot.json").write_text(
                    json.dumps(optimizer_state(self.model)),
                    encoding="utf-8",
                )

        fit_session(
            model,
            images,
            labels,
            run_dir,
            SCHEDULE,
            batch_size=8,
            device=device,
            extra_callbacks=[CaptureRestoredState()],
        )
        publish_final_model(model, run_dir, restore=False)
        return 0
    if phase == "continuous":
        model, run_dir = make_run(run_root / "continuous", device)
        fit_session(model, images, labels, run_dir, SCHEDULE, batch_size=8, device=device)
        publish_final_model(model, run_dir, restore=False)
        rates = [row["learning_rate"] for row in load_history(run_dir)]
        (run_root / "continuous_rates.json").write_text(json.dumps(rates), encoding="utf-8")
        return 0
    raise SystemExit(f"unknown phase {phase}")


if __name__ == "__main__":
    raise SystemExit(main())
