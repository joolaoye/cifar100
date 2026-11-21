"""Epoch learning rate, history, and checkpoint callbacks.

Callback order is part of the recovery contract. The learning rate is set in
``on_epoch_begin`` from the zero-based epoch index. A non-finite batch loss
raises before history or a checkpoint is written, so that partial epoch is
not recorded. History is appended and fsynced in ``on_epoch_end`` before that
epoch's checkpoint generation is published. A crash between those two steps
leaves history ahead of the last consistent generation, which resume
truncates. ``TerminateOnNaN`` remains a backstop and does not choose a
checkpoint. An optional stop callback runs after the checkpoint so a normal
interruption leaves the epoch durable.
"""

import math
import time
from pathlib import Path


def make_training_callbacks(
    run_dir: Path,
    schedule_rate,
    *,
    stop_after_completed_epochs: int | None = None,
    extra_callbacks: list | None = None,
) -> list:
    """Build the production callback list. Keras is imported here, not at import."""
    from keras import callbacks

    from cifar100_study.runs import (
        TrainingDivergence,
        TrainingError,
        TrainingInterrupted,
        append_history_row,
        note_completed_epoch,
        save_epoch_checkpoint,
    )

    class RejectNonFiniteLoss(callbacks.Callback):
        """Fail the epoch before it can be stored as a completed checkpoint."""

        def __init__(self):
            super().__init__()
            self._epoch = None

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch = int(epoch)

        def on_train_batch_end(self, batch, logs=None):
            loss = (logs or {}).get("loss")
            if loss is not None and _loss_is_finite(loss):
                return
            if loss is None:
                return
            affected = self._epoch + 1 if self._epoch is not None else "unknown"
            raise TrainingDivergence(
                f"non-finite loss stopped epoch {affected} at batch {int(batch)} "
                "before that epoch could be recorded"
            )

        def on_epoch_end(self, epoch, logs=None):
            if getattr(self.model, "stop_training", False):
                raise TrainingDivergence(
                    f"non-finite loss stopped epoch {int(epoch) + 1} "
                    "before that epoch could be recorded"
                )

    class MilestoneLearningRate(callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.model.optimizer.learning_rate = float(schedule_rate(int(epoch)))

    class EpochHistory(callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._started = None

        def on_epoch_begin(self, epoch, logs=None):
            self._started = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            if self._started is None:
                raise TrainingError("epoch timing started without an epoch boundary")
            elapsed = time.perf_counter() - self._started
            rate = self.model.optimizer.learning_rate
            actual = float(rate.numpy()) if hasattr(rate, "numpy") else float(rate)
            expected = float(schedule_rate(int(epoch)))
            if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-7):
                raise TrainingError(
                    f"epoch index {epoch} used learning rate {actual}, schedule requires {expected}"
                )
            logs = logs or {}
            if "loss" not in logs or "accuracy" not in logs:
                raise TrainingError("training logs are missing loss or accuracy")
            append_history_row(
                run_dir,
                {
                    "epoch": int(epoch) + 1,
                    "loss": float(logs["loss"]),
                    "accuracy": float(logs["accuracy"]),
                    "learning_rate": actual,
                    "epoch_seconds": elapsed,
                },
            )
            note_completed_epoch(run_dir)

    class EpochCheckpoint(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            save_epoch_checkpoint(self.model, Path(run_dir), int(epoch) + 1)

    class StopAfterCompletedEpoch(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch + 1 >= stop_after_completed_epochs:
                raise TrainingInterrupted(f"stopped after {epoch + 1} completed epochs")

    ordered = [
        MilestoneLearningRate(),
        RejectNonFiniteLoss(),
        EpochHistory(),
        EpochCheckpoint(),
    ]
    if extra_callbacks:
        ordered.extend(extra_callbacks)
    ordered.append(callbacks.TerminateOnNaN())
    if stop_after_completed_epochs is not None:
        ordered.append(StopAfterCompletedEpoch())
    return ordered


def _loss_is_finite(loss) -> bool:
    if hasattr(loss, "numpy"):
        loss = loss.numpy()
    try:
        value = float(loss)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)
