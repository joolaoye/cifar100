"""Final evaluation, example figures, training curves, and the comparison table."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from cifar100_study import contracts
from cifar100_study.config import load_config
from cifar100_study.evaluation import (
    EvaluationError,
    class_name,
    confusion_counts,
    evaluate_run,
    read_completed_history,
    score_probabilities,
    select_examples,
)
from cifar100_study.reporting import ReportingError, comparison_rows, training_series, write_report
from cifar100_study.runs import append_history_row, create_run, read_status, update_status
from cifar100_study.trainer import EpochSchedule, create_training_run

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
CONFIRM = "confirm-final-test"


def test_top1_counts_are_exact_on_a_known_example():
    probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]], dtype=np.float32)
    labels = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    scored = score_probabilities(probabilities, labels)
    assert scored["test_count"] == 3
    assert scored["correct_predictions"] == 2
    assert scored["top1_accuracy"] == pytest.approx(2 / 3)
    assert scored["predicted"].tolist() == [0, 1, 1]
    assert scored["truth"].tolist() == [0, 1, 0]
    counts = confusion_counts(scored["truth"], scored["predicted"], 2)
    assert counts.tolist() == [[1, 1], [0, 1]]
    assert int(np.trace(counts)) == 2


def test_example_figure_titles_stay_clear_of_images_and_labels(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    from cifar100_study.evaluation import _write_example_figure

    display = np.zeros((8, 16, 16, 3), dtype=np.uint8)
    display[..., 1] = 90
    truth = np.asarray([47, 58, 41, 1, 20, 54, 22, 63], dtype=np.int64)
    predicted = np.asarray([52, 58, 84, 1, 12, 36, 7, 15], dtype=np.int64)
    scored = {"truth": truth, "predicted": predicted}
    selected = {"correct": [1, 3], "incorrect": [0, 2, 4, 5]}
    recorded: dict[str, list] = {}
    original = Figure.savefig

    def capture(self, *args, **kwargs):
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        recorded["titles"] = []
        recorded["axes"] = []
        for axis in self.axes:
            if axis.get_title():
                recorded["titles"].append(axis.title.get_window_extent(renderer).frozen())
                recorded["axes"].append(axis.get_window_extent(renderer).frozen())
        recorded["suptitle"] = self.texts[0].get_window_extent(renderer).frozen()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    _write_example_figure(tmp_path / "examples.png", display, scored, selected, 100)
    assert len(recorded["titles"]) == 6
    for title_box, axis_box in zip(recorded["titles"], recorded["axes"], strict=True):
        assert title_box.y0 >= axis_box.y1 + 2
        assert not title_box.overlaps(axis_box)
    for title_box in recorded["titles"]:
        assert not title_box.overlaps(recorded["suptitle"])
        for axis_box in recorded["axes"]:
            assert not title_box.overlaps(axis_box)
    for index, title_box in enumerate(recorded["titles"]):
        for other in recorded["titles"][index + 1 :]:
            assert not title_box.overlaps(other)


def test_example_selection_keeps_test_order_and_short_groups():
    correct = np.asarray([True, True, False, True, False])
    selected = select_examples(correct, limit=4)
    assert selected == {"correct": [0, 1, 3], "incorrect": [2, 4]}
    assert class_name(0, 100) == "apple"
    assert class_name(3, 2) == "class 3"


def test_evaluation_refuses_missing_authorization_and_incomplete_runs(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("test data was loaded")

    monkeypatch.setattr("cifar100_study.data.load_test_data", forbidden)
    monkeypatch.setattr("cifar100_study.evaluation._load_saved_model", forbidden)
    run_dir = _completed_run(tmp_path, "e0", epochs=1, seconds=[1.25])
    with pytest.raises(EvaluationError, match="confirm-final-test"):
        evaluate_run(run_dir, confirm_final_test=None, test_images=_images(), test_labels=_labels([0, 1, 0, 1]), display_images=_display())
    update_status(run_dir, status="running")
    with pytest.raises(EvaluationError, match="completed"):
        evaluate_run(
            run_dir,
            confirm_final_test=CONFIRM,
            test_images=_images(),
            test_labels=_labels([0, 1, 0, 1]),
            display_images=_display(),
        )


def test_configuration_and_cache_identity_mismatches_are_rejected(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("test data was loaded")

    monkeypatch.setattr("cifar100_study.data.load_test_data", forbidden)
    run_dir = _completed_run(tmp_path / "config", "e0", epochs=1, seconds=[1.25])
    snapshot = run_dir / "config.yaml"
    snapshot.write_bytes(snapshot.read_bytes() + b"\n")
    with pytest.raises(EvaluationError, match="config.yaml"):
        evaluate_run(
            run_dir,
            confirm_final_test=CONFIRM,
            test_images=_images(),
            test_labels=_labels([0, 1, 0, 1]),
            display_images=_display(),
        )

    official = _official_run_with_wrong_cache(tmp_path / "cache-case")
    with pytest.raises(EvaluationError, match="preprocessing identity"):
        evaluate_run(official, confirm_final_test=CONFIRM, cache_root=tmp_path / "cache-case")


def test_completed_run_scores_reuses_and_keeps_example_correspondence(tmp_path, monkeypatch):
    class_ids = [0, 1, 0, 1]
    run_dir = _completed_run(tmp_path, "e0", epochs=1, seconds=[1.25])
    history_before = (run_dir / "history.csv").read_bytes()
    model_before = (run_dir / "model.keras").read_bytes()
    predicted = {"calls": 0}
    original = __import__(
        "cifar100_study.evaluation", fromlist=["_predict_probabilities"]
    )._predict_probabilities

    def counting(model, images, batch_size):
        predicted["calls"] += 1
        return original(model, images, batch_size)

    monkeypatch.setattr("cifar100_study.evaluation._predict_probabilities", counting)
    display = _display()
    summary = evaluate_run(
        run_dir,
        confirm_final_test=CONFIRM,
        test_images=_images(),
        test_labels=_labels(class_ids),
        display_images=display,
    )
    assert predicted["calls"] == 1
    assert summary["inference"] is True
    assert summary["reused"] is False
    assert summary["test_count"] == 4
    assert summary["correct_predictions"] == 2
    assert summary["top1_accuracy"] == pytest.approx(0.5)
    assert (run_dir / "history.csv").read_bytes() == history_before
    assert (run_dir / "model.keras").read_bytes() == model_before
    _assert_examples_match_display(run_dir, display)

    monkeypatch.setattr(
        "cifar100_study.evaluation._predict_probabilities",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inference was repeated")),
    )
    reused = evaluate_run(
        run_dir,
        confirm_final_test=CONFIRM,
        test_images=_images(),
        test_labels=_labels(class_ids),
        display_images=display,
    )
    assert reused["reused"] is True
    assert reused["inference"] is False
    assert reused["top1_accuracy"] == summary["top1_accuracy"]
    assert predicted["calls"] == 1

    (run_dir / "evaluation" / "confusion.csv").unlink()
    with pytest.raises(EvaluationError, match="partial or inconsistent"):
        evaluate_run(
            run_dir,
            confirm_final_test=CONFIRM,
            test_images=_images(),
            test_labels=_labels(class_ids),
            display_images=display,
        )
    assert (run_dir / "evaluation" / "predictions.csv").is_file()


def test_training_curves_use_recorded_metrics_only(tmp_path):
    run_dir = _completed_run(tmp_path, "e0", epochs=2, seconds=[1.5, 2.25])
    series = training_series(run_dir)
    history = read_completed_history(run_dir, 2)
    assert series["epoch"] == [1, 2]
    assert series["loss"] == [row["loss"] for row in history]
    assert series["accuracy"] == [row["accuracy"] for row in history]
    assert "val_loss" not in series and "val_accuracy" not in series
    text = (run_dir / "history.csv").read_text(encoding="utf-8")
    text = text.replace(
        "epoch,loss,accuracy,learning_rate,epoch_seconds",
        "epoch,loss,accuracy,learning_rate,epoch_seconds,val_accuracy",
    )
    (run_dir / "history.csv").write_text(text, encoding="utf-8")
    with pytest.raises(EvaluationError, match="header"):
        training_series(run_dir)


def test_comparison_differences_and_time_totals(tmp_path, monkeypatch):
    runs = {
        "e0": _scored_run(tmp_path, "e0", [0, 0, 0, 0], [1.5]),
        "e1": _scored_run(tmp_path, "e1", [0, 1, 0, 1], [2.0, 0.75]),
        "e2": _scored_run(tmp_path, "e2", [1, 1, 1, 1], [0.25]),
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("report touched test data or ran inference")

    monkeypatch.setattr("cifar100_study.data.load_test_data", forbidden)
    monkeypatch.setattr("cifar100_study.evaluation._predict_probabilities", forbidden)
    rows = comparison_rows([runs["e2"], runs["e0"], runs["e1"]], allow_synthetic=True)
    by_name = {row["experiment"]: row for row in rows}
    assert [row["experiment"] for row in rows] == ["e0", "e1", "e2"]
    assert by_name["e0"]["top1_accuracy"] == pytest.approx(1.0)
    assert by_name["e1"]["top1_accuracy"] == pytest.approx(0.5)
    assert by_name["e2"]["top1_accuracy"] == pytest.approx(0.0)
    assert by_name["e0"]["accuracy_difference_vs_e0_percentage_points"] == pytest.approx(0.0)
    assert by_name["e1"]["accuracy_difference_vs_e0_percentage_points"] == pytest.approx(-50.0)
    assert by_name["e2"]["accuracy_difference_vs_e0_percentage_points"] == pytest.approx(-100.0)
    assert by_name["e0"]["epoch_seconds_total"] == pytest.approx(1.5)
    assert by_name["e1"]["epoch_seconds_total"] == pytest.approx(2.75)
    assert by_name["e2"]["epoch_seconds_total"] == pytest.approx(0.25)
    assert by_name["e0"]["convolutional_layers"] == 0
    assert by_name["e2"]["convolutional_layers"] == 0
    assert by_name["e2"]["macs_per_image"] == 0
    assert by_name["e0"]["trainable_parameters"] == 8
    assert by_name["e0"]["epochs"] == 1
    assert by_name["e1"]["epochs"] == 2
    written = write_report(list(runs.values()), tmp_path / "report", allow_synthetic=True)
    assert (tmp_path / "report" / "comparison.csv").is_file()
    assert (tmp_path / "report" / "comparison.md").is_file()
    assert (tmp_path / "report" / "comparison.tex").is_file()
    assert "multiply-accumulates" in (tmp_path / "report" / "comparison.md").read_text(encoding="utf-8")
    assert "not training FLOPs" in (tmp_path / "report" / "comparison.md").read_text(encoding="utf-8")
    assert "\\begin{tabular}" in (tmp_path / "report" / "comparison.tex").read_text(encoding="utf-8")
    for experiment in ("e0", "e1", "e2"):
        assert (tmp_path / "report" / f"{experiment}-training-loss.png").stat().st_size > 0
        assert (tmp_path / "report" / f"{experiment}-training-accuracy.png").stat().st_size > 0
    assert [curve["experiment"] for curve in written["curves"]] == ["e0", "e1", "e2"]
    with pytest.raises(ReportingError, match="E0, E1, and E2"):
        comparison_rows([runs["e0"], runs["e1"]])
    (runs["e2"] / "evaluation" / "evaluation.json").unlink()
    with pytest.raises(ReportingError, match="missing or inconsistent"):
        comparison_rows(list(runs.values()), allow_synthetic=True)


def test_comparison_requires_one_controlled_official_study(tmp_path):
    synthetic = {
        "e0": _scored_run(tmp_path / "synthetic", "e0", [0, 0, 0, 0], [1.0]),
        "e1": _scored_run(tmp_path / "synthetic", "e1", [0, 1, 0, 1], [1.0]),
        "e2": _scored_run(tmp_path / "synthetic", "e2", [1, 1, 1, 1], [1.0]),
    }
    with pytest.raises(ReportingError, match="official"):
        comparison_rows([synthetic["e0"], synthetic["e1"], synthetic["e2"]])

    official_e2 = _official_completed(tmp_path / "mixed", "e2", cache_key="cache-a", fingerprint="finger-a")
    with pytest.raises(ReportingError, match="mixes synthetic and official"):
        comparison_rows(
            [synthetic["e0"], synthetic["e1"], official_e2],
            allow_synthetic=True,
        )

    shared = {
        "e0": _official_completed(tmp_path / "cache", "e0", cache_key="cache-a", fingerprint="finger-a"),
        "e1": _official_completed(tmp_path / "cache", "e1", cache_key="cache-a", fingerprint="finger-a"),
        "e2": _official_completed(tmp_path / "cache", "e2", cache_key="cache-b", fingerprint="finger-a"),
    }
    with pytest.raises(ReportingError, match="cache key"):
        comparison_rows([shared["e0"], shared["e1"], shared["e2"]])

    seeded = {
        "e0": _official_completed(tmp_path / "seed", "e0", cache_key="cache-a", fingerprint="finger-a"),
        "e1": _official_completed(tmp_path / "seed", "e1", cache_key="cache-a", fingerprint="finger-a"),
        "e2": _official_completed(tmp_path / "seed", "e2", cache_key="cache-a", fingerprint="finger-a"),
    }
    _set_metadata_seed(seeded["e0"], 43)
    with pytest.raises(EvaluationError, match="seed"):
        comparison_rows([seeded["e0"], seeded["e1"], seeded["e2"]])

    _set_metadata_batch(synthetic["e2"], 16)
    with pytest.raises(ReportingError, match="batch size"):
        comparison_rows([synthetic["e0"], synthetic["e1"], synthetic["e2"]], allow_synthetic=True)


def test_replaced_checkpoint_is_rejected_before_test_access(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("test data was loaded")

    monkeypatch.setattr("cifar100_study.data.load_test_data", forbidden)
    monkeypatch.setattr(
        "cifar100_study.evaluation._predict_probabilities",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inference ran")),
    )
    run_dir = _completed_run(tmp_path, "e0", epochs=1, seconds=[1.25])
    _save_different_same_shaped_model(run_dir / "model.keras")
    with pytest.raises(EvaluationError, match="final model recorded when training completed"):
        evaluate_run(
            run_dir,
            confirm_final_test=CONFIRM,
            test_images=_images(),
            test_labels=_labels([0, 1, 0, 1]),
            display_images=_display(),
        )


def test_figure_failure_finishes_without_a_second_prediction(tmp_path, monkeypatch):
    predicted = {"calls": 0}
    original = __import__(
        "cifar100_study.evaluation", fromlist=["_predict_probabilities"]
    )._predict_probabilities

    def counting(model, images, batch_size):
        predicted["calls"] += 1
        return original(model, images, batch_size)

    monkeypatch.setattr("cifar100_study.evaluation._predict_probabilities", counting)
    figure = {"calls": 0}
    original_figure = __import__(
        "cifar100_study.evaluation", fromlist=["_write_example_figure"]
    )._write_example_figure

    def fail_once(*args, **kwargs):
        figure["calls"] += 1
        if figure["calls"] == 1:
            raise RuntimeError("figure publication failed")
        return original_figure(*args, **kwargs)

    monkeypatch.setattr("cifar100_study.evaluation._write_example_figure", fail_once)
    run_dir = _completed_run(tmp_path, "e0", epochs=1, seconds=[1.25])
    kwargs = {
        "confirm_final_test": CONFIRM,
        "test_images": _images(),
        "test_labels": _labels([0, 1, 0, 1]),
        "display_images": _display(),
    }
    with pytest.raises(RuntimeError, match="figure publication failed"):
        evaluate_run(run_dir, **kwargs)
    assert predicted["calls"] == 1
    assert (run_dir / "evaluation" / "predictions.csv").is_file()
    assert not (run_dir / "evaluation" / "evaluation.json").exists()
    finished = evaluate_run(run_dir, **kwargs)
    assert predicted["calls"] == 1
    assert finished["inference"] is False
    assert finished["reused"] is False
    assert (run_dir / "evaluation" / "evaluation.json").is_file()
    assert finished["correct_predictions"] == 2


def test_unrecoverable_predictions_require_a_recorded_reason(tmp_path, monkeypatch):
    predicted = {"calls": 0}
    original = __import__(
        "cifar100_study.evaluation", fromlist=["_predict_probabilities"]
    )._predict_probabilities

    def fail_first(model, images, batch_size):
        predicted["calls"] += 1
        if predicted["calls"] == 1:
            raise RuntimeError("inference interrupted")
        return original(model, images, batch_size)

    monkeypatch.setattr("cifar100_study.evaluation._predict_probabilities", fail_first)
    run_dir = _completed_run(tmp_path, "e0", epochs=1, seconds=[1.25])
    kwargs = {
        "confirm_final_test": CONFIRM,
        "test_images": _images(),
        "test_labels": _labels([0, 1, 0, 1]),
        "display_images": _display(),
    }
    with pytest.raises(RuntimeError, match="inference interrupted"):
        evaluate_run(run_dir, **kwargs)
    assert (run_dir / "evaluation" / "attempt.json").is_file()
    with pytest.raises(EvaluationError, match="explicit reason"):
        evaluate_run(run_dir, **kwargs)
    assert predicted["calls"] == 1
    reason = "predictions were unreadable after the interrupted inference"
    summary = evaluate_run(run_dir, repeat_inference_reason=reason, **kwargs)
    assert predicted["calls"] == 2
    assert summary["inference"] is True
    import json

    attempt = json.loads((run_dir / "evaluation" / "attempt.json").read_text(encoding="utf-8"))
    assert attempt["repeat_authorizations"][0]["reason"] == reason
    reused = evaluate_run(run_dir, repeat_inference_reason="select a different checkpoint", **kwargs)
    assert reused["reused"] is True
    assert predicted["calls"] == 2


def test_interrupted_summary_recovers_with_one_inference(tmp_path, monkeypatch):
    import json

    predicted = {"calls": 0}
    original_predict = __import__(
        "cifar100_study.evaluation", fromlist=["_predict_probabilities"]
    )._predict_probabilities
    original_write = __import__(
        "cifar100_study.evaluation", fromlist=["_write_json"]
    )._write_json

    def counting(model, images, batch_size):
        predicted["calls"] += 1
        return original_predict(model, images, batch_size)

    def interrupt_summary(path, payload):
        if path.name == "evaluation.json":
            path.write_text("{", encoding="utf-8")
            raise RuntimeError("summary write interrupted")
        return original_write(path, payload)

    monkeypatch.setattr("cifar100_study.evaluation._predict_probabilities", counting)
    monkeypatch.setattr("cifar100_study.evaluation._write_json", interrupt_summary)
    run_dir = _completed_run(tmp_path, "e0", epochs=1, seconds=[1.25])
    kwargs = {
        "confirm_final_test": CONFIRM,
        "test_images": _images(),
        "test_labels": _labels([0, 1, 0, 1]),
        "display_images": _display(),
    }
    with pytest.raises(RuntimeError, match="summary write interrupted"):
        evaluate_run(run_dir, **kwargs)
    predictions = (run_dir / "evaluation" / "predictions.csv").read_bytes()
    assert predictions
    assert (run_dir / "evaluation" / "evaluation.json").read_text(encoding="utf-8") == "{"
    assert predicted["calls"] == 1
    monkeypatch.setattr("cifar100_study.evaluation._write_json", original_write)
    finished = evaluate_run(run_dir, **kwargs)
    assert predicted["calls"] == 1
    assert finished["inference"] is False
    assert finished["reused"] is False
    assert (run_dir / "evaluation" / "predictions.csv").read_bytes() == predictions
    assert json.loads((run_dir / "evaluation" / "evaluation.json").read_text(encoding="utf-8"))["correct_predictions"] == 2

    summary_path = run_dir / "evaluation" / "evaluation.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["macs_per_image"] = 816_860_160
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="MAC"):
        evaluate_run(run_dir, **kwargs)
    assert predicted["calls"] == 1
    assert (run_dir / "evaluation" / "predictions.csv").read_bytes() == predictions


def test_metadata_seed_is_checked_on_every_entrypoint(tmp_path):
    kwargs = {
        "confirm_final_test": CONFIRM,
        "test_images": _images(),
        "test_labels": _labels([0, 0, 0, 0]),
        "display_images": _display(),
    }
    initial = _completed_run(tmp_path / "initial", "e0", epochs=1, seconds=[1.0])
    config_bytes = (initial / "config.yaml").read_bytes()
    _set_metadata_seed(initial, 43)
    assert (initial / "config.yaml").read_bytes() == config_bytes
    with pytest.raises(EvaluationError, match="seed"):
        evaluate_run(initial, **kwargs)

    reused = _scored_run(tmp_path / "reuse", "e0", [0, 0, 0, 0], [1.0])
    _set_metadata_seed(reused, 43)
    with pytest.raises(EvaluationError, match="seed"):
        evaluate_run(reused, **kwargs)
    with pytest.raises(EvaluationError, match="seed"):
        training_series(reused)

    runs = {
        "e0": _scored_run(tmp_path / "report", "e0", [0, 0, 0, 0], [1.0]),
        "e1": _scored_run(tmp_path / "report", "e1", [0, 1, 0, 1], [1.0]),
        "e2": _scored_run(tmp_path / "report", "e2", [1, 1, 1, 1], [1.0]),
    }
    _set_metadata_seed(runs["e1"], 43)
    with pytest.raises(EvaluationError, match="seed"):
        comparison_rows([runs["e2"], runs["e0"], runs["e1"]])


def test_seed_change_and_altered_duration_are_rechecked(tmp_path):
    import json

    seeded = _scored_run(tmp_path / "seed", "e0", [0, 0, 0, 0], [1.5])
    snapshot = seeded / "config.yaml"
    snapshot.write_text(snapshot.read_text(encoding="utf-8").replace("seed: 42", "seed: 43", 1), encoding="utf-8")
    metadata_path = seeded / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="seed"):
        evaluate_run(
            seeded,
            confirm_final_test=CONFIRM,
            test_images=_images(),
            test_labels=_labels([0, 0, 0, 0]),
            display_images=_display(),
        )
    with pytest.raises(EvaluationError, match="seed"):
        training_series(seeded)

    timed = _scored_run(tmp_path / "time", "e1", [0, 1, 0, 1], [1.25, 0.5])
    history_path = timed / "history.csv"
    history_path.write_text(
        history_path.read_text(encoding="utf-8").replace("1.250000", "9.250000", 1),
        encoding="utf-8",
    )
    update_status(timed, epoch_seconds_total=9.75)
    with pytest.raises(EvaluationError, match="history.csv"):
        evaluate_run(
            timed,
            confirm_final_test=CONFIRM,
            test_images=_images(),
            test_labels=_labels([0, 1, 0, 1]),
            display_images=_display(),
        )
    with pytest.raises(EvaluationError, match="history.csv"):
        training_series(timed)


def test_transposed_confusion_and_altered_macs_are_rejected(tmp_path):
    import json

    from cifar100_study.evaluation import load_evaluation_summary

    confused = _scored_run(tmp_path / "confusion", "e0", [0, 1, 0, 1], [1.0])
    confusion_path = confused / "evaluation" / "confusion.csv"
    rows = [line.split(",") for line in confusion_path.read_text(encoding="utf-8").splitlines()]
    transposed = "\n".join(",".join(row) for row in zip(*rows)) + "\n"
    assert transposed != confusion_path.read_text(encoding="utf-8")
    confusion_path.write_text(transposed, encoding="utf-8")
    summary_path = confused / "evaluation" / "evaluation.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["confusion_sha256"] = hashlib.sha256(confusion_path.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="every cell"):
        load_evaluation_summary(confused)

    altered = _scored_run(tmp_path / "macs", "e2", [1, 1, 1, 1], [0.25])
    summary_path = altered / "evaluation" / "evaluation.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["macs_per_image"] = 816_860_160
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    companions = {
        "e0": _scored_run(tmp_path / "macs", "e0", [0, 0, 0, 0], [1.0]),
        "e1": _scored_run(tmp_path / "macs", "e1", [0, 1, 0, 1], [1.0]),
        "e2": altered,
    }
    with pytest.raises(EvaluationError, match="MAC"):
        load_evaluation_summary(altered)
    with pytest.raises(ReportingError, match="MAC"):
        comparison_rows([companions["e0"], companions["e1"], companions["e2"]], allow_synthetic=True)


def _completed_run(tmp_path: Path, experiment: str, epochs: int, seconds: list[float]) -> Path:
    config_path = CONFIG_DIR / f"{experiment}.yaml"
    config = load_config(config_path)
    model, parameters, depth, macs = _zero_model()
    schedule = EpochSchedule(0.05, 0.1, (), epochs)
    run_dir = create_training_run(
        tmp_path,
        config=config,
        config_text=config_path.read_text(encoding="utf-8"),
        schedule=schedule,
        data_source="synthetic",
        batch_size=8,
        output_classes=2,
        device={"requested": "cpu", "selected_type": "CPU", "selected_name": f"{experiment}-cpu"},
        cache_key_hex=None,
        source_fingerprint=None,
        trainable_parameters=parameters,
        convolutional_layers=depth,
        git_root=ROOT,
    )
    model.save(run_dir / "model.keras")
    for epoch, elapsed in enumerate(seconds, start=1):
        append_history_row(
            run_dir,
            {
                "epoch": epoch,
                "loss": 1.0 + epoch / 10.0,
                "accuracy": 0.25 * epoch,
                "learning_rate": 0.05,
                "epoch_seconds": elapsed,
            },
        )
    update_status(
        run_dir,
        status="completed",
        completed_epochs=epochs,
        epoch_seconds_total=sum(seconds),
        final_model_sha256=hashlib.sha256((run_dir / "model.keras").read_bytes()).hexdigest(),
        history_sha256=hashlib.sha256((run_dir / "history.csv").read_bytes()).hexdigest(),
    )
    assert read_status(run_dir)["status"] == "completed"
    return run_dir


def _scored_run(tmp_path: Path, experiment: str, class_ids: list[int], seconds: list[float]) -> Path:
    run_dir = _completed_run(tmp_path, experiment, len(seconds), seconds)
    evaluate_run(
        run_dir,
        confirm_final_test=CONFIRM,
        test_images=_images(),
        test_labels=_labels(class_ids),
        display_images=_display(),
    )
    return run_dir


def _official_run_with_wrong_cache(tmp_path: Path) -> Path:
    from cifar100_study.data import prepare_cache, preprocessing_identity

    config = load_config(CONFIG_DIR / "e0.yaml")
    identity = dict(preprocessing_identity(config))
    identity["train_size"] = 4
    identity["test_size"] = 2
    identity["image_shape"] = [2, 2, 3]
    images = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)
    labels = np.arange(4, dtype=np.int64) % 100
    result = prepare_cache(identity, images, labels, images[:2], labels[:2], tmp_path)
    metadata = {
        "study": config.study,
        "experiment": "e0",
        "data_source": "official",
        "seed": config.training.seed,
        "batch_size": config.training.batch_size,
        "output_classes": 100,
        "architecture": config.model.architecture,
        "trainable_parameters": 1_387_108,
        "convolutional_layers": 9,
        "schedule": {
            "initial_learning_rate": 0.05,
            "decay_factor": 0.1,
            "milestones": [200, 250, 300],
            "epochs": 350,
        },
        "cache_key": result.path.name,
        "source_fingerprint": result.source_fingerprint,
        "device": {"requested": "cpu", "selected_type": "CPU", "selected_name": "cpu"},
    }
    run_dir = create_run(tmp_path / "runs", metadata, (CONFIG_DIR / "e0.yaml").read_text(encoding="utf-8"))
    (run_dir / "model.keras").write_bytes(b"not-loaded")
    history = ["epoch,loss,accuracy,learning_rate,epoch_seconds"]
    history.extend(f"{epoch},1.0,0.1,0.05,1.000000" for epoch in range(1, 351))
    history_path = run_dir / "history.csv"
    history_path.write_text("\n".join(history) + "\n", encoding="utf-8")
    update_status(
        run_dir,
        status="completed",
        completed_epochs=350,
        epoch_seconds_total=350.0,
        final_model_sha256=hashlib.sha256((run_dir / "model.keras").read_bytes()).hexdigest(),
        history_sha256=hashlib.sha256(history_path.read_bytes()).hexdigest(),
    )
    return run_dir


def _official_completed(tmp_path: Path, experiment: str, *, cache_key: str, fingerprint: str) -> Path:
    from cifar100_study.allcnn import REGISTERED_COMPLEXITY

    config = load_config(CONFIG_DIR / f"{experiment}.yaml")
    spec = contracts.EXPERIMENT_SPECS[experiment]
    depth, parameters, _macs = REGISTERED_COMPLEXITY[experiment]
    schedule = EpochSchedule(
        config.training.initial_learning_rate,
        config.training.lr_decay_factor,
        spec.lr_milestones,
        spec.epochs,
    )
    run_dir = create_training_run(
        tmp_path,
        config=config,
        config_text=(CONFIG_DIR / f"{experiment}.yaml").read_text(encoding="utf-8"),
        schedule=schedule,
        data_source="official",
        batch_size=config.training.batch_size,
        output_classes=config.dataset.num_classes,
        device={"requested": "cpu", "selected_type": "CPU", "selected_name": "cpu"},
        cache_key_hex=cache_key,
        source_fingerprint=fingerprint,
        trainable_parameters=parameters,
        convolutional_layers=depth,
        git_root=ROOT,
    )
    (run_dir / "model.keras").write_bytes(b"official-not-loaded")
    lines = ["epoch,loss,accuracy,learning_rate,epoch_seconds"]
    lines.extend(
        f"{epoch},1.0,0.1,{config.training.initial_learning_rate:.2f},1.000000"
        for epoch in range(1, spec.epochs + 1)
    )
    history = run_dir / "history.csv"
    history.write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_status(
        run_dir,
        status="completed",
        completed_epochs=spec.epochs,
        epoch_seconds_total=float(spec.epochs),
        final_model_sha256=hashlib.sha256((run_dir / "model.keras").read_bytes()).hexdigest(),
        history_sha256=hashlib.sha256(history.read_bytes()).hexdigest(),
    )
    return run_dir


def _set_metadata_batch(run_dir: Path, batch_size: int) -> None:
    import json

    path = run_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["batch_size"] = batch_size
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _set_metadata_seed(run_dir: Path, seed: int) -> None:
    import json

    path = run_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["seed"] == 42
    metadata["seed"] = seed
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_different_same_shaped_model(path: Path) -> None:
    from keras import Input, Model, initializers, layers

    inputs = Input((4,), name="images")
    outputs = layers.Dense(
        2,
        activation="softmax",
        use_bias=False,
        kernel_initializer=initializers.Ones(),
        name="probs",
    )(inputs)
    Model(inputs, outputs, name="replacement").save(path)


def _zero_model():
    from keras import Input, Model, initializers, layers

    from cifar100_study.allcnn import architecture_report

    inputs = Input((4,), name="images")
    outputs = layers.Dense(
        2,
        activation="softmax",
        use_bias=False,
        kernel_initializer=initializers.Zeros(),
        name="probs",
    )(inputs)
    model = Model(inputs, outputs, name="tiny_eval")
    report = architecture_report(model)
    return model, report.trainable_parameters, report.convolutional_layers, report.macs_per_image


def _images() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 4 * 4, dtype=np.float32).reshape(4, 4)


def _labels(class_ids: list[int]) -> np.ndarray:
    labels = np.zeros((len(class_ids), 2), dtype=np.float32)
    labels[np.arange(len(class_ids)), class_ids] = 1.0
    return labels


def _display() -> np.ndarray:
    images = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    for index in range(4):
        images[index, :, :, :] = index + 1
        images[index, 0, 0, 0] = 20 + index
    return images


def _assert_examples_match_display(run_dir: Path, display: np.ndarray) -> None:
    import json

    payload = json.loads((run_dir / "evaluation" / "examples.json").read_text(encoding="utf-8"))
    predictions = (run_dir / "evaluation" / "predictions.csv").read_text(encoding="utf-8").splitlines()[1:]
    parsed = []
    for line in predictions:
        index, truth, predicted, _confidence = line.split(",")
        parsed.append((int(index), int(truth), int(predicted)))
    correct = [index for index, truth, predicted in parsed if truth == predicted]
    incorrect = [index for index, truth, predicted in parsed if truth != predicted]
    assert [row["index"] for row in payload["correct"]] == correct[:4]
    assert [row["index"] for row in payload["incorrect"]] == incorrect[:4]
    for group in ("correct", "incorrect"):
        for row in payload[group]:
            image = np.ascontiguousarray(display[row["index"]])
            assert row["image_sha256"] == hashlib.sha256(image.tobytes()).hexdigest()
            assert parsed[row["index"]][1:] == (row["true_label"], row["predicted_label"])
