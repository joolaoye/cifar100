"""Validation and comparison tests for the locked E0/E1/E2 configs."""

from dataclasses import replace
from pathlib import Path

import subprocess

import pytest
import yaml

from cifar100_study.config import (
    ConfigError,
    assert_registered_differences,
    differing_paths,
    load_config,
    load_study_configs,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load_mutated(tmp_path: Path, name: str, mutate) -> object:
    document = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return load_config(path)


def test_each_config_matches_the_registered_study():
    e0 = load_config(CONFIG_DIR / "e0.yaml")
    e1 = load_config(CONFIG_DIR / "e1.yaml")
    e2 = load_config(CONFIG_DIR / "e2.yaml")

    assert e0.study == e1.study == e2.study == "cifar100-allcnnc"
    assert (e0.experiment, e1.experiment, e2.experiment) == ("e0", "e1", "e2")
    assert e0 == load_config(CONFIG_DIR / "e0.yaml")

    assert e0.dataset.name == "cifar100"
    assert e0.dataset.label_mode == "fine"
    assert e0.dataset.num_classes == 100
    assert e0.dataset.image_shape == (32, 32, 3)
    assert e0.dataset.train_size == 50_000
    assert e0.dataset.test_size == 10_000
    assert e0.dataset.validation_split == 0
    assert e0.dataset.augmentation == "none"
    assert e0.dataset.label_encoding == "one_hot"
    assert e0.preprocessing.rescale_from == (0, 255)
    assert e0.preprocessing.rescale_to == (0, 1)
    assert e0.preprocessing.gcn is True
    assert e0.preprocessing.gcn_min_divisor == 1e-8
    assert e0.preprocessing.zca_whitening is True
    assert e0.preprocessing.zca_form == "full"
    assert e0.preprocessing.zca_regularization == 0.1
    assert e0.preprocessing.covariance_divisor == "N"
    assert e0.preprocessing.fit_on == "train"
    assert e0.preprocessing.dtype == "float32"
    assert e0.model.architecture == "all-cnn-c"
    assert e0.model.input_dropout == 0.2
    assert e0.model.dropout_after_downsampler == 0.5
    assert e0.model.initializer == "he_normal"
    assert e0.training.optimizer == "sgd"
    assert e0.training.momentum == 0.9
    assert e0.training.initial_learning_rate == 0.05
    assert e0.training.lr_decay_factor == 0.1
    assert e0.training.lr_milestone_convention == "zero_indexed_epoch"
    assert e0.training.batch_size == 256
    assert e0.training.l2_coefficient == 0.001
    assert e0.training.loss == e1.training.loss == e2.training.loss == "categorical_crossentropy"
    assert e0.training.seed == 42
    assert e0.training.pretrained_weights is False
    assert e0.training.early_stopping is False
    assert e0.training.checkpoint_selection == "final_scheduled_epoch"
    assert e0.lifecycle.validation_driven_checkpoints is False
    assert e0.lifecycle.run_deletion is False
    assert e0.lifecycle.resumable is True
    assert e0.lifecycle.history_reconciliation is True
    assert e0.lifecycle.final_test_guard == "confirm-final-test"
    assert e0.lifecycle.periodic_test_metrics is False

    assert e0.training.epochs == 350
    assert e0.training.lr_milestones == (200, 250, 300)
    assert e0.model.stage_extra_convolutions == (0, 0, 0)
    assert e1.training.epochs == 700
    assert e1.training.lr_milestones == (400, 500, 600)
    assert e1.model.stage_extra_convolutions == (0, 0, 0)
    assert e2.training.epochs == 350
    assert e2.training.lr_milestones == (200, 250, 300)
    assert e2.model.stage_extra_convolutions == (3, 3, 3)
    assert 9 + sum(e0.model.stage_extra_convolutions) == 9
    assert 9 + sum(e2.model.stage_extra_convolutions) == 18


def test_study_directory_agrees_on_registered_differences():
    e0, e1, e2 = load_study_configs(CONFIG_DIR)
    assert differing_paths(e0, e1) == (
        "experiment",
        "training.epochs",
        "training.lr_milestones",
    )
    assert differing_paths(e0, e2) == (
        "experiment",
        "model.stage_extra_convolutions",
    )
    assert differing_paths(e1, e2) == (
        "experiment",
        "model.stage_extra_convolutions",
        "training.epochs",
        "training.lr_milestones",
    )
    again = differing_paths(e0, e1)
    assert again == tuple(sorted(again))


def test_configs_are_complete_files_without_yaml_merge():
    for name in ("e0.yaml", "e1.yaml", "e2.yaml"):
        text = (CONFIG_DIR / name).read_text(encoding="utf-8")
        assert "<<" not in text
        for key in ("stage_extra_convolutions", "lr_milestones", "epochs", "seed"):
            assert key in text
        assert 'loss: "categorical_crossentropy"' in text
        assert "sparse_categorical_crossentropy" not in text


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("training", "seed", 43, "seed"),
        ("training", "seed", True, "integer"),
        ("training", "lr_milestones", [300, 250, 200], "strictly increasing"),
        ("training", "lr_milestones", [200, 250], "expected 3 integers"),
        ("training", "epochs", 700, "epochs"),
        ("training", "initial_learning_rate", 0.01, "initial_learning_rate"),
        ("training", "lr_decay_factor", 0.5, "lr_decay_factor"),
        ("training", "lr_milestone_convention", "one_indexed_epoch", "lr_milestone_convention"),
        ("training", "batch_size", 128, "batch_size"),
        ("training", "momentum", 0.95, "momentum"),
        ("training", "loss", "sparse_categorical_crossentropy", "loss"),
        ("training", "loss", "binary_crossentropy", "loss"),
        ("training", "loss", "mean_squared_error", "loss"),
        ("training", "pretrained_weights", True, "pretrained_weights"),
        ("training", "early_stopping", True, "early_stopping"),
        ("training", "checkpoint_selection", "best_val_accuracy", "checkpoint_selection"),
        ("training", "nesterov", False, "unknown keys"),
        ("model", "stage_extra_convolutions", [3, 3, 3], "stage_extra_convolutions"),
        ("model", "input_dropout", 0.5, "input_dropout"),
        ("model", "initializer", "glorot_uniform", "initializer"),
        ("dataset", "augmentation", "horizontal_flip", "augmentation"),
        ("dataset", "augmentation", ["flip"], "string"),
        ("dataset", "validation_split", 1, "validation_split"),
        ("dataset", "train_size", 45000, "train_size"),
        ("dataset", "label_mode", "coarse", "label_mode"),
        ("dataset", "num_classes", 20, "num_classes"),
        ("preprocessing", "fit_on", "test", "fit_on"),
        ("preprocessing", "zca_whitening", False, "zca_whitening"),
        ("preprocessing", "gcn", False, "gcn"),
        ("preprocessing", "gcn_min_divisor", 1e-6, "gcn_min_divisor"),
        ("preprocessing", "zca_form", "pca", "zca_form"),
        ("preprocessing", "zca_regularization", 0.2, "zca_regularization"),
        ("preprocessing", "covariance_divisor", "n-1", "covariance_divisor"),
        ("preprocessing", "dtype", "float64", "dtype"),
        ("dataset", "label_encoding", "sparse", "label_encoding"),
        ("lifecycle", "run_deletion", True, "run_deletion"),
        ("lifecycle", "validation_driven_checkpoints", True, "validation_driven_checkpoints"),
        ("lifecycle", "periodic_test_metrics", True, "periodic_test_metrics"),
        (None, "notes", "extra", "unknown keys"),
        (None, "experiment", "e3", "unknown experiment"),
    ],
)
def test_e0_rejects_study_violations(tmp_path, section, key, value, match):
    def mutate(document):
        if section is None:
            document[key] = value
        else:
            document[section][key] = value

    with pytest.raises(ConfigError, match=match):
        _load_mutated(tmp_path, "e0.yaml", mutate)


@pytest.mark.parametrize(
    ("name", "section", "key", "value", "match"),
    [
        ("e1.yaml", "training", "epochs", 350, "epochs"),
        ("e1.yaml", "training", "lr_milestones", [200, 250, 300], "lr_milestones"),
        ("e2.yaml", "model", "stage_extra_convolutions", [0, 0, 0], "stage_extra_convolutions"),
        ("e2.yaml", "model", "stage_extra_convolutions", [3, 3, 0], "stage_extra_convolutions"),
        ("e2.yaml", "training", "epochs", 700, "epochs"),
        ("e1.yaml", "model", "stage_extra_convolutions", [3, 3, 3], "stage_extra_convolutions"),
    ],
)
def test_experiment_specific_locks(tmp_path, name, section, key, value, match):
    def mutate(document):
        document[section][key] = value

    with pytest.raises(ConfigError, match=match):
        _load_mutated(tmp_path, name, mutate)


def test_missing_key_fails(tmp_path):
    def mutate(document):
        del document["training"]["seed"]

    with pytest.raises(ConfigError, match="missing keys: seed"):
        _load_mutated(tmp_path, "e0.yaml", mutate)


def test_duplicate_key_fails(tmp_path):
    text = (CONFIG_DIR / "e0.yaml").read_text(encoding="utf-8")
    path = tmp_path / "e0.yaml"
    path.write_text('study: "other"\n' + text, encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate YAML key"):
        load_config(path)


def test_rejects_non_mapping_python_tags_and_missing_files(tmp_path):
    listing = tmp_path / "list.yaml"
    listing.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(listing)

    tagged = tmp_path / "tagged.yaml"
    tagged.write_text('study: !!python/str "cifar100-allcnnc"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(tagged)

    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")


def test_filename_must_match_experiment_id(tmp_path):
    e0_text = (CONFIG_DIR / "e0.yaml").read_text(encoding="utf-8")
    for name in ("e0.yaml", "e1.yaml", "e2.yaml"):
        (tmp_path / name).write_text(e0_text, encoding="utf-8")
    with pytest.raises(ConfigError, match="expected 'e1'"):
        load_study_configs(tmp_path)


def test_study_directory_requires_every_config(tmp_path):
    text = (CONFIG_DIR / "e0.yaml").read_text(encoding="utf-8")
    (tmp_path / "e0.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot read"):
        load_study_configs(tmp_path)


def test_comparison_rejects_a_shared_illegal_setting():
    e0, e1, e2 = load_study_configs(CONFIG_DIR)
    bad_e0 = replace(e0, training=replace(e0.training, seed=7))
    bad_e1 = replace(e1, training=replace(e1.training, seed=7))
    with pytest.raises(ConfigError, match="seed"):
        assert_registered_differences(bad_e0, bad_e1, e2)


def test_comparison_rejects_an_unregistered_difference():
    e0, e1, e2 = load_study_configs(CONFIG_DIR)
    tweaked = replace(e1, training=replace(e1.training, seed=7))
    with pytest.raises(ConfigError, match="training.seed"):
        assert_registered_differences(e0, tweaked, e2)


def test_comparison_rejects_missing_e2_depth_difference():
    e0, e1, e2 = load_study_configs(CONFIG_DIR)
    flattened = replace(
        e2,
        model=replace(e2.model, stage_extra_convolutions=(0, 0, 0)),
    )
    with pytest.raises(ConfigError, match="stage_extra_convolutions"):
        assert_registered_differences(e0, e1, flattened)


def test_comparison_rejects_unregistered_epoch_budget():
    e0, e1, e2 = load_study_configs(CONFIG_DIR)
    tweaked = replace(e1, training=replace(e1.training, epochs=701))
    with pytest.raises(ConfigError, match="training.epochs"):
        assert_registered_differences(e0, tweaked, e2)


def test_sources_do_not_import_tensorflow_at_module_scope():
    import ast
    import sys

    import cifar100_study
    import cifar100_study.cli
    import cifar100_study.config
    import cifar100_study.contracts

    root = Path(__file__).resolve().parents[1]
    for path in list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            imported = []
            if isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
            imported.extend(alias.name for alias in node.names)
            joined = " ".join(imported)
            assert "tensorflow" not in joined
            assert "keras" not in joined

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import cifar100_study, cifar100_study.cli, cifar100_study.config, "
            "cifar100_study.contracts, cifar100_study.data, cifar100_study.allcnn, "
            "cifar100_study.runs, cifar100_study.training_callbacks, "
            "cifar100_study.trainer, cifar100_study.evaluation, "
            "cifar100_study.reporting, sys\n"
            "bad = [name for name in sys.modules if name == 'keras' or name.startswith('keras.') "
            "or name == 'tensorflow' or name.startswith('tensorflow.')]\n"
            "raise SystemExit(1 if bad else 0)\n",
        ],
        cwd=root,
        check=False,
    )
    assert probe.returncode == 0
    assert cifar100_study.__version__ == "0.0.0"
