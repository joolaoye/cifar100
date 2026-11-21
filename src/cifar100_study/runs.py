"""Run directories, history, and recovery bookkeeping.

Consistency model
-----------------
Each completed epoch is saved as one checkpoint generation under ``backup/``.
The generation directory is published only after both ``model.weights.h5`` and
``state.json`` are on disk. ``state.json`` records the completed epoch, the
optimizer iteration count, the run id, and the SHA-256 of that weights file.
A weights file is never selected from an epoch number that belongs to a
different file.

The manifest lists the two newest verified generations. A generation that was
published before the manifest was updated is included. Ranking uses the
completed epoch, then publication time, then the generation name. Resume tries
the newest generation whose hash, run id, and iteration count agree, and
otherwise the previous generation. A staging directory left by a crash is not
a generation. History is reconciled to the generation that is actually restored.

History is written before the checkpoint. Rows beyond the restored epoch are
removed when they form a valid suffix. A valid final row is given a terminating
newline before another row is appended. A malformed trailing CSV row is deleted
from the file before any later append. An interior malformed row is rejected.
History that is shorter than the restored epoch, has a gap or duplicate, or
disagrees with the learning-rate schedule is rejected. Missing metrics are
never invented.

If the run records completed epochs and no generation verifies, resume raises
instead of starting again at epoch zero. That failure leaves the run
resumable; it is not marked permanently failed.

``metadata.json`` is written once. ``status.json`` holds mutable status and
the sum of completed-epoch durations, excluding time between sessions. This
module never deletes a run directory.
"""

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

HISTORY_COLUMNS = ("epoch", "loss", "accuracy", "learning_rate", "epoch_seconds")
RUN_STATUS_VALUES = ("created", "running", "interrupted", "completed", "failed")
_LR_REL_TOL = 1e-5
_LR_ABS_TOL = 1e-7


class TrainingError(ValueError):
    """The training run cannot be started, resumed, or recorded safely."""


class TrainingInterrupted(TrainingError):
    """Training stopped at an epoch boundary before the scheduled budget."""


class TrainingDivergence(TrainingError):
    """A non-finite loss stopped an epoch before it could be recorded."""


def create_run(run_root: Path, metadata: dict, config_text: str) -> Path:
    """Create a new run directory exclusively. Never replaces an existing one."""
    experiment = metadata.get("experiment")
    if type(experiment) is not str or not experiment:
        raise TrainingError("a run needs an experiment id")
    if metadata.get("data_source") not in {"official", "synthetic"}:
        raise TrainingError("data_source must be 'official' or 'synthetic'")
    schedule = metadata.get("schedule")
    if type(schedule) is not dict:
        raise TrainingError("a run needs its learning-rate schedule in metadata")
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    config_bytes = config_text.encode("utf-8")
    payload["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    payload["created_at"] = _now()
    for _ in range(3):
        run_id = f"{experiment}-{uuid.uuid4().hex[:12]}"
        path = root / run_id
        try:
            path.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        payload["run_id"] = run_id
        _atomic_json(path / "metadata.json", payload)
        _atomic_json(
            path / "status.json",
            {
                "status": "created",
                "completed_epochs": 0,
                "epoch_seconds_total": 0.0,
                "sessions": [],
                "started_at": None,
                "finished_at": None,
                "interruption": None,
                "timing_note": (
                    "epoch_seconds_total sums completed-epoch durations and "
                    "excludes time between sessions"
                ),
            },
        )
        (path / "config.yaml").write_bytes(config_bytes)
        return path
    raise TrainingError("could not allocate a unique run directory")


def read_metadata(run_dir: Path) -> dict:
    return _read_json(Path(run_dir) / "metadata.json", "run metadata")


def read_status(run_dir: Path) -> dict:
    status = _read_json(Path(run_dir) / "status.json", "run status")
    if status.get("status") not in RUN_STATUS_VALUES:
        raise TrainingError(f"run status {status.get('status')!r} is not recognized")
    return status


def update_status(run_dir: Path, **changes) -> dict:
    status = read_status(run_dir)
    status.update(changes)
    if status.get("status") not in RUN_STATUS_VALUES:
        raise TrainingError(f"refusing to record status {status.get('status')!r}")
    _atomic_json(Path(run_dir) / "status.json", status)
    return status


def open_session(run_dir: Path, device: dict | None) -> None:
    status = read_status(run_dir)
    if status["status"] in {"completed", "failed"}:
        raise TrainingError(f"cannot train a run whose status is {status['status']}")
    sessions = list(status.get("sessions") or [])
    sessions.append({"started_at": _now(), "ended_at": None, "device": device})
    changes = {
        "status": "running",
        "sessions": sessions,
        "interruption": None,
    }
    if status.get("started_at") is None:
        changes["started_at"] = sessions[-1]["started_at"]
    update_status(run_dir, **changes)


def close_session(
    run_dir: Path,
    status_name: str,
    interruption: str | None = None,
    *,
    final_model_sha256: str | None = None,
    history_sha256: str | None = None,
) -> None:
    status = read_status(run_dir)
    sessions = list(status.get("sessions") or [])
    if sessions and sessions[-1].get("ended_at") is None:
        sessions[-1]["ended_at"] = _now()
    try:
        rows = load_history(run_dir)
        completed = len(rows)
        epoch_seconds = sum(row["epoch_seconds"] for row in rows)
    except TrainingError:
        completed = status.get("completed_epochs", 0)
        epoch_seconds = status.get("epoch_seconds_total", 0.0)
    changes = {
        "status": status_name,
        "sessions": sessions,
        "completed_epochs": completed,
        "epoch_seconds_total": epoch_seconds,
        "interruption": interruption,
        "finished_at": _now() if status_name in {"completed", "failed"} else status.get("finished_at"),
    }
    if status_name == "completed":
        if not _is_sha256(final_model_sha256) or not _is_sha256(history_sha256):
            raise TrainingError(
                "completion requires the SHA-256 of the published model.keras and history.csv"
            )
        changes["final_model_sha256"] = final_model_sha256
        changes["history_sha256"] = history_sha256
    update_status(run_dir, **changes)


def note_completed_epoch(run_dir: Path) -> None:
    rows = load_history(run_dir)
    update_status(
        run_dir,
        completed_epochs=len(rows),
        epoch_seconds_total=sum(row["epoch_seconds"] for row in rows),
        last_learning_rate=rows[-1]["learning_rate"] if rows else None,
    )


def append_history_row(run_dir: Path, row: dict) -> None:
    """Append one completed epoch and fsync it before the caller returns."""
    path = Path(run_dir) / "history.csv"
    existing = load_history(run_dir)
    expected = len(existing) + 1
    if int(row["epoch"]) != expected:
        raise TrainingError(
            f"refusing to append history epoch {row['epoch']}; the next epoch is {expected}"
        )
    line = _format_row(row)
    new_file = not path.exists() or path.stat().st_size == 0
    if not new_file:
        _ensure_trailing_newline(path)
    with path.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write(",".join(HISTORY_COLUMNS) + "\n")
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_history(run_dir: Path) -> list[dict]:
    """Return completed rows and delete a malformed trailing row from the file."""
    path = Path(run_dir) / "history.csv"
    if not path.exists() or path.stat().st_size == 0:
        return []
    text = path.read_text(encoding="utf-8")
    missing_newline = not text.endswith("\n")
    parsed_rows = list(csv.reader(io.StringIO(text)))
    if not parsed_rows:
        return []
    if tuple(parsed_rows[0]) != HISTORY_COLUMNS:
        raise TrainingError(
            "history.csv header must be " + ",".join(HISTORY_COLUMNS)
        )
    body = parsed_rows[1:]
    dropped_trailing = False
    while body:
        cells = body[-1]
        if cells in ([], [""]) or _parse_history_cells(cells) is None:
            body.pop()
            dropped_trailing = True
            continue
        break
    rows: list[dict] = []
    for index, cells in enumerate(body):
        if cells in ([], [""]):
            raise TrainingError(f"malformed history row {index + 2}: {cells!r}")
        row = _parse_history_cells(cells)
        if row is None:
            raise TrainingError(f"malformed history row {index + 2}: {cells!r}")
        rows.append(row)
    for index, row in enumerate(rows, start=1):
        if row["epoch"] != index:
            raise TrainingError(
                f"history epoch numbering is inconsistent at position {index}: "
                f"got epoch {row['epoch']}"
            )
    if dropped_trailing:
        _write_history(run_dir, rows)
    elif missing_newline:
        _ensure_trailing_newline(path)
    return rows


def reconcile_history(run_dir: Path, recovered_epochs: int, rate_for_index) -> list[dict]:
    """Align history with the last durably recovered epoch.

    ``rate_for_index`` takes the zero-based epoch index and returns the
    learning rate that epoch must have used.
    """
    if type(recovered_epochs) is not int or recovered_epochs < 0:
        raise TrainingError(f"recovered epoch count is invalid: {recovered_epochs!r}")
    rows = load_history(run_dir)
    _require_scheduled_rates(rows, rate_for_index)
    if len(rows) > recovered_epochs:
        if recovered_epochs == 0 and not valid_checkpoints(run_dir):
            raise TrainingError(
                "history.csv records completed epochs but the recovery backup is missing"
            )
        rows = rows[:recovered_epochs]
        _write_history(run_dir, rows)
    elif len(rows) < recovered_epochs:
        raise TrainingError(
            f"history.csv has {len(rows)} completed epochs but the backup recovered "
            f"{recovered_epochs}; refusing to invent metrics"
        )
    return rows


def save_epoch_checkpoint(model, run_dir: Path, completed_epochs: int) -> None:
    """Publish one consistent epoch generation, then keep the latest two."""
    if type(completed_epochs) is not int or completed_epochs < 1:
        raise TrainingError(f"checkpoint epoch must be a positive integer, got {completed_epochs!r}")
    metadata = read_metadata(run_dir)
    run_id = metadata.get("run_id")
    if type(run_id) is not str or not run_id:
        raise TrainingError("run metadata has no run_id")
    backup = Path(run_dir) / "backup"
    backup.mkdir(exist_ok=True)
    staging = backup / f".staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        weights_path = staging / "model.weights.h5"
        model.save_weights(weights_path)
        _fsync_file(weights_path)
        state = {
            "batch": 0,
            "epoch": completed_epochs,
            "optimizer_iterations": _optimizer_iterations(model),
            "run_id": run_id,
            "weights_sha256": _sha256_file(weights_path),
        }
        state_path = staging / "state.json"
        state_path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
        _fsync_file(state_path)
        generation = f"g-{completed_epochs:08d}-{uuid.uuid4().hex[:8]}"
        os.rename(staging, backup / generation)
    except Exception:
        if staging.exists():
            _delete_directory(staging)
        raise
    _update_manifest(backup, generation, run_id)
    _prune_checkpoints(backup)


def valid_checkpoints(run_dir: Path) -> list[dict]:
    """Return verified generations, newest completed epoch first."""
    backup = Path(run_dir) / "backup"
    if not backup.is_dir():
        return []
    metadata = read_metadata(run_dir)
    run_id = metadata.get("run_id")
    if type(run_id) is not str:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for name in _read_manifest(backup):
        if name not in seen:
            names.append(name)
            seen.add(name)
    if backup.is_dir():
        for child in sorted(backup.iterdir()):
            if child.is_dir() and not child.name.startswith(".") and child.name not in seen:
                names.append(child.name)
    verified = []
    for name in names:
        checkpoint = _verify_generation(backup / name, run_id)
        if checkpoint is not None:
            verified.append(checkpoint)
    verified.sort(key=_checkpoint_rank, reverse=True)
    return verified


def restore_checkpoint(model, run_dir: Path) -> int:
    """Load the newest consistent generation. Zero means the run has not checkpointed."""
    if not getattr(model, "compiled", False) or getattr(model, "optimizer", None) is None:
        raise TrainingError("compile the model before restoring optimizer state")
    candidates = valid_checkpoints(run_dir)
    for candidate in candidates:
        try:
            if not model.optimizer.built:
                model.optimizer.build(model.trainable_variables)
            model.load_weights(candidate["weights_path"])
            loaded = _optimizer_iterations(model)
        except Exception:
            continue
        if loaded != candidate["optimizer_iterations"]:
            continue
        return int(candidate["epoch"])
    if candidates or _run_records_progress(run_dir):
        raise TrainingError(
            "no valid checkpoint is available for this run; refusing to restart from epoch zero"
        )
    return 0


def recovered_completed_epochs(run_dir: Path) -> int:
    """Return the newest verified epoch, or zero when nothing has been checkpointed."""
    checkpoints = valid_checkpoints(run_dir)
    if not checkpoints:
        return 0
    return int(checkpoints[0]["epoch"])


def git_state(root: Path) -> dict:
    """Record HEAD when it exists. A repository with no commits is not an error."""

    def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )

    inside = _run(["git", "rev-parse", "--is-inside-work-tree"])
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return {
            "commit": None,
            "has_commit": False,
            "dirty": None,
            "detail": "not a git work tree",
        }
    status = _run(["git", "status", "--porcelain"])
    dirty = None if status.returncode != 0 else bool(status.stdout.strip())
    head = _run(["git", "rev-parse", "--verify", "HEAD"])
    if head.returncode != 0:
        return {
            "commit": None,
            "has_commit": False,
            "dirty": dirty,
            "detail": "repository has no commits",
        }
    return {
        "commit": head.stdout.strip(),
        "has_commit": True,
        "dirty": dirty,
        "detail": None,
    }


def _require_scheduled_rates(rows: list[dict], rate_for_index) -> None:
    for row in rows:
        expected = float(rate_for_index(row["epoch"] - 1))
        if not math.isclose(row["learning_rate"], expected, rel_tol=_LR_REL_TOL, abs_tol=_LR_ABS_TOL):
            raise TrainingError(
                f"history epoch {row['epoch']} learning rate {row['learning_rate']} "
                f"does not match the schedule value {expected}"
            )


def _write_history(run_dir: Path, rows: list[dict]) -> None:
    lines = [",".join(HISTORY_COLUMNS)]
    lines.extend(_format_row(row) for row in rows)
    text = "\n".join(lines) + "\n"
    path = Path(run_dir) / "history.csv"
    temporary = path.with_suffix(".csv.tmp")
    temporary.write_text(text, encoding="utf-8")
    _fsync_file(temporary)
    os.replace(temporary, path)


def _ensure_trailing_newline(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _format_row(row: dict) -> str:
    return ",".join(
        (
            str(int(row["epoch"])),
            format(float(row["loss"]), ".10g"),
            format(float(row["accuracy"]), ".10g"),
            format(float(row["learning_rate"]), ".10g"),
            format(float(row["epoch_seconds"]), ".6f"),
        )
    )


def _parse_history_cells(cells: list[str]) -> dict | None:
    if len(cells) != len(HISTORY_COLUMNS):
        return None
    try:
        epoch_text = cells[0].strip()
        epoch = int(epoch_text)
        if epoch_text != str(epoch):
            return None
        loss = float(cells[1])
        accuracy = float(cells[2])
        learning_rate = float(cells[3])
        epoch_seconds = float(cells[4])
    except ValueError:
        return None
    values = (loss, accuracy, learning_rate, epoch_seconds)
    if epoch < 1 or epoch_seconds < 0 or any(not math.isfinite(value) for value in values):
        return None
    return {
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy,
        "learning_rate": learning_rate,
        "epoch_seconds": epoch_seconds,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _fsync_file(temporary)
    os.replace(temporary, path)


def _update_manifest(backup: Path, generation: str, run_id: str) -> None:
    verified = []
    for child in backup.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            checkpoint = _verify_generation(child, run_id)
            if checkpoint is not None:
                verified.append(checkpoint)
    if not any(item["generation"] == generation for item in verified):
        raise TrainingError("the checkpoint generation just published did not verify")
    verified.sort(key=_checkpoint_rank, reverse=True)
    _atomic_json(
        backup / "manifest.json",
        {"generations": [item["generation"] for item in verified[:2]]},
    )


def _checkpoint_rank(checkpoint: dict) -> tuple:
    """Newest completed epoch first; ties use publication time, then name."""
    return (checkpoint["epoch"], checkpoint["published_at"], checkpoint["generation"])


def _read_manifest(backup: Path) -> list[str]:
    path = backup / "manifest.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    generations = payload.get("generations") if type(payload) is dict else None
    if type(generations) is not list:
        return []
    return [name for name in generations if type(name) is str and name and not name.startswith(".")]


def _verify_generation(directory: Path, run_id: str) -> dict | None:
    state_path = directory / "state.json"
    weights_path = directory / "model.weights.h5"
    if not state_path.is_file() or not weights_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if type(state) is not dict:
        return None
    epoch = state.get("epoch")
    iterations = state.get("optimizer_iterations")
    digest = state.get("weights_sha256")
    if (
        state.get("run_id") != run_id
        or state.get("batch") != 0
        or type(epoch) is not int
        or epoch < 1
        or type(iterations) is not int
        or iterations < 0
        or type(digest) is not str
    ):
        return None
    if _sha256_file(weights_path) != digest:
        return None
    return {
        "epoch": epoch,
        "optimizer_iterations": iterations,
        "weights_path": weights_path,
        "generation": directory.name,
        "published_at": state_path.stat().st_mtime_ns,
    }


def _prune_checkpoints(backup: Path) -> None:
    keep = set(_read_manifest(backup))
    for child in list(backup.iterdir()):
        if child.is_dir() and not child.name.startswith(".") and child.name not in keep:
            _delete_directory(child)


def _run_records_progress(run_dir: Path) -> bool:
    status = read_status(run_dir)
    if int(status.get("completed_epochs") or 0) > 0:
        return True
    history = Path(run_dir) / "history.csv"
    if not history.exists() or history.stat().st_size == 0:
        return False
    try:
        return len(load_history(run_dir)) > 0
    except TrainingError:
        return True


def _optimizer_iterations(model) -> int:
    value = model.optimizer.iterations
    if hasattr(value, "numpy"):
        value = value.numpy()
    return int(value)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _delete_directory(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    if path.exists():
        path.rmdir()


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise TrainingError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainingError(f"{label} is unreadable: {path}") from exc
    if type(payload) is not dict:
        raise TrainingError(f"{label} must be a JSON object")
    return payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
