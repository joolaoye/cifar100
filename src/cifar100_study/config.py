"""Strict loader for the locked E0, E1, and E2 configs.

Unknown keys, missing keys, type mismatches, and any departure from
``cifar100_study.contracts`` fail. This module applies no defaults and does
not normalize values that violate the study.
"""

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import yaml

from cifar100_study import contracts

_EXPERIMENT_IDS = ("e0", "e1", "e2")
_E1_SCIENTIFIC_DIFFERENCES = frozenset({"training.epochs", "training.lr_milestones"})
_E2_SCIENTIFIC_DIFFERENCES = frozenset({"model.stage_extra_convolutions"})
_E2_VS_E1_SCIENTIFIC_DIFFERENCES = _E1_SCIENTIFIC_DIFFERENCES | _E2_SCIENTIFIC_DIFFERENCES


class ConfigError(ValueError):
    """A config is unreadable or disagrees with the locked study."""


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str
    label_mode: str
    num_classes: int
    image_shape: tuple[int, int, int]
    train_size: int
    test_size: int
    validation_split: int
    augmentation: str
    label_encoding: str


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    rescale_from: tuple[int, int]
    rescale_to: tuple[int, int]
    gcn: bool
    gcn_min_divisor: float
    zca_whitening: bool
    zca_form: str
    zca_regularization: float
    covariance_divisor: str
    fit_on: str
    dtype: str


@dataclass(frozen=True, slots=True)
class ModelConfig:
    architecture: str
    stage_extra_convolutions: tuple[int, int, int]
    input_dropout: float
    dropout_after_downsampler: float
    initializer: str


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    optimizer: str
    momentum: float
    initial_learning_rate: float
    lr_decay_factor: float
    lr_milestones: tuple[int, int, int]
    lr_milestone_convention: str
    epochs: int
    batch_size: int
    l2_coefficient: float
    loss: str
    seed: int
    pretrained_weights: bool
    early_stopping: bool
    checkpoint_selection: str


@dataclass(frozen=True, slots=True)
class LifecycleConfig:
    validation_driven_checkpoints: bool
    run_deletion: bool
    resumable: bool
    history_reconciliation: bool
    final_test_guard: str
    periodic_test_metrics: bool


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    study: str
    experiment: str
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    lifecycle: LifecycleConfig


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.Node) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if type(key) is not str:
            raise ConfigError(f"mapping keys must be strings, got {key!r}")
        if key in mapping:
            raise ConfigError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _expect_str(value: object, path: str) -> str:
    if type(value) is not str:
        raise ConfigError(f"{path}: expected a string")
    return value


def _expect_int(value: object, path: str) -> int:
    if type(value) is not int:
        raise ConfigError(f"{path}: expected an integer")
    return value


def _expect_float(value: object, path: str) -> float:
    if type(value) is not float:
        raise ConfigError(f"{path}: expected a float")
    return value


def _expect_bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{path}: expected a boolean")
    return value


def _expect_int_tuple(value: object, length: int, path: str) -> tuple[int, ...]:
    if type(value) is not list:
        raise ConfigError(f"{path}: expected a list of {length} integers")
    if len(value) != length:
        raise ConfigError(f"{path}: expected {length} integers, got {len(value)}")
    return tuple(_expect_int(item, f"{path}[{index}]") for index, item in enumerate(value))


def _require_mapping(value: object, path: str) -> dict:
    if type(value) is not dict:
        raise ConfigError(f"{path}: expected a mapping")
    return value


def _take(raw: dict, path: str, parsers: dict) -> dict:
    unknown = sorted(set(raw) - set(parsers))
    missing = sorted(set(parsers) - set(raw))
    if unknown:
        raise ConfigError(f"{path}: unknown keys: {', '.join(unknown)}")
    if missing:
        raise ConfigError(f"{path}: missing keys: {', '.join(missing)}")
    return {key: parser(raw[key], f"{path}.{key}") for key, parser in parsers.items()}


def _parse_dataset(value: object) -> DatasetConfig:
    raw = _take(
        _require_mapping(value, "dataset"),
        "dataset",
        {
            "name": _expect_str,
            "label_mode": _expect_str,
            "num_classes": _expect_int,
            "image_shape": lambda item, path: _expect_int_tuple(item, 3, path),
            "train_size": _expect_int,
            "test_size": _expect_int,
            "validation_split": _expect_int,
            "augmentation": _expect_str,
            "label_encoding": _expect_str,
        },
    )
    return DatasetConfig(**raw)


def _parse_preprocessing(value: object) -> PreprocessingConfig:
    raw = _take(
        _require_mapping(value, "preprocessing"),
        "preprocessing",
        {
            "rescale_from": lambda item, path: _expect_int_tuple(item, 2, path),
            "rescale_to": lambda item, path: _expect_int_tuple(item, 2, path),
            "gcn": _expect_bool,
            "gcn_min_divisor": _expect_float,
            "zca_whitening": _expect_bool,
            "zca_form": _expect_str,
            "zca_regularization": _expect_float,
            "covariance_divisor": _expect_str,
            "fit_on": _expect_str,
            "dtype": _expect_str,
        },
    )
    return PreprocessingConfig(**raw)


def _parse_model(value: object) -> ModelConfig:
    raw = _take(
        _require_mapping(value, "model"),
        "model",
        {
            "architecture": _expect_str,
            "stage_extra_convolutions": lambda item, path: _expect_int_tuple(item, 3, path),
            "input_dropout": _expect_float,
            "dropout_after_downsampler": _expect_float,
            "initializer": _expect_str,
        },
    )
    return ModelConfig(**raw)


def _parse_training(value: object) -> TrainingConfig:
    raw = _take(
        _require_mapping(value, "training"),
        "training",
        {
            "optimizer": _expect_str,
            "momentum": _expect_float,
            "initial_learning_rate": _expect_float,
            "lr_decay_factor": _expect_float,
            "lr_milestones": lambda item, path: _expect_int_tuple(item, 3, path),
            "lr_milestone_convention": _expect_str,
            "epochs": _expect_int,
            "batch_size": _expect_int,
            "l2_coefficient": _expect_float,
            "loss": _expect_str,
            "seed": _expect_int,
            "pretrained_weights": _expect_bool,
            "early_stopping": _expect_bool,
            "checkpoint_selection": _expect_str,
        },
    )
    return TrainingConfig(**raw)


def _parse_lifecycle(value: object) -> LifecycleConfig:
    raw = _take(
        _require_mapping(value, "lifecycle"),
        "lifecycle",
        {
            "validation_driven_checkpoints": _expect_bool,
            "run_deletion": _expect_bool,
            "resumable": _expect_bool,
            "history_reconciliation": _expect_bool,
            "final_test_guard": _expect_str,
            "periodic_test_metrics": _expect_bool,
        },
    )
    return LifecycleConfig(**raw)


def _parse_document(document: object) -> ExperimentConfig:
    raw = _require_mapping(document, "top level")
    parsed = _take(
        raw,
        "top level",
        {
            "study": _expect_str,
            "experiment": _expect_str,
            "dataset": lambda item, _path: _parse_dataset(item),
            "preprocessing": lambda item, _path: _parse_preprocessing(item),
            "model": lambda item, _path: _parse_model(item),
            "training": lambda item, _path: _parse_training(item),
            "lifecycle": lambda item, _path: _parse_lifecycle(item),
        },
    )
    return ExperimentConfig(**parsed)


def expected_config(experiment: str) -> ExperimentConfig:
    """Return the only legal config for ``experiment``."""
    if experiment not in contracts.EXPERIMENT_SPECS:
        raise ConfigError(f"experiment: unknown experiment {experiment!r}")
    spec = contracts.EXPERIMENT_SPECS[experiment]
    return ExperimentConfig(
        study=contracts.STUDY_ID,
        experiment=experiment,
        dataset=DatasetConfig(
            name=contracts.DATASET_NAME,
            label_mode=contracts.LABEL_MODE,
            num_classes=contracts.NUM_CLASSES,
            image_shape=contracts.IMAGE_SHAPE,
            train_size=contracts.TRAIN_SIZE,
            test_size=contracts.TEST_SIZE,
            validation_split=contracts.VALIDATION_SPLIT,
            augmentation=contracts.AUGMENTATION,
            label_encoding=contracts.LABEL_ENCODING,
        ),
        preprocessing=PreprocessingConfig(
            rescale_from=contracts.RESCALE_FROM,
            rescale_to=contracts.RESCALE_TO,
            gcn=contracts.GCN,
            gcn_min_divisor=contracts.GCN_MIN_DIVISOR,
            zca_whitening=contracts.ZCA_WHITENING,
            zca_form=contracts.ZCA_FORM,
            zca_regularization=contracts.ZCA_REGULARIZATION,
            covariance_divisor=contracts.COVARIANCE_DIVISOR,
            fit_on=contracts.FIT_ON,
            dtype=contracts.DTYPE,
        ),
        model=ModelConfig(
            architecture=contracts.ARCHITECTURE,
            stage_extra_convolutions=spec.stage_extra_convolutions,
            input_dropout=contracts.INPUT_DROPOUT,
            dropout_after_downsampler=contracts.DROPOUT_AFTER_DOWNSAMPLER,
            initializer=contracts.INITIALIZER,
        ),
        training=TrainingConfig(
            optimizer=contracts.OPTIMIZER,
            momentum=contracts.MOMENTUM,
            initial_learning_rate=contracts.INITIAL_LEARNING_RATE,
            lr_decay_factor=contracts.LR_DECAY_FACTOR,
            lr_milestones=spec.lr_milestones,
            lr_milestone_convention=contracts.LR_MILESTONE_CONVENTION,
            epochs=spec.epochs,
            batch_size=contracts.BATCH_SIZE,
            l2_coefficient=contracts.L2_COEFFICIENT,
            loss=contracts.LOSS,
            seed=contracts.SEED,
            pretrained_weights=contracts.PRETRAINED_WEIGHTS,
            early_stopping=contracts.EARLY_STOPPING,
            checkpoint_selection=contracts.CHECKPOINT_SELECTION,
        ),
        lifecycle=LifecycleConfig(
            validation_driven_checkpoints=contracts.VALIDATION_DRIVEN_CHECKPOINTS,
            run_deletion=contracts.RUN_DELETION_ALLOWED,
            resumable=contracts.RESUMABLE,
            history_reconciliation=contracts.HISTORY_RECONCILIATION,
            final_test_guard=contracts.FINAL_TEST_GUARD,
            periodic_test_metrics=contracts.PERIODIC_TEST_METRICS,
        ),
    )


def _lookup(value: object, path: str) -> object:
    current = value
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _is_dataclass_instance(value: object) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def differing_paths(left: object, right: object) -> tuple[str, ...]:
    """Return the sorted dotted paths whose values differ.

    Sequences such as milestones and stage extras are one path each. The result
    is deterministic.
    """
    paths: list[str] = []
    _collect_differences(left, right, "", paths)
    return tuple(sorted(paths))


def _collect_differences(left: object, right: object, prefix: str, paths: list[str]) -> None:
    if _is_dataclass_instance(left) and _is_dataclass_instance(right):
        left_fields = {field.name for field in fields(left)}
        right_fields = {field.name for field in fields(right)}
        for name in sorted(left_fields | right_fields):
            child = f"{prefix}.{name}" if prefix else name
            if name not in left_fields or name not in right_fields:
                paths.append(child)
                continue
            _collect_differences(getattr(left, name), getattr(right, name), child, paths)
        return
    if type(left) is not type(right) or left != right:
        paths.append(prefix or "<root>")


def assert_locked(config: ExperimentConfig) -> None:
    """Reject a config that is not exactly the pre-registered experiment."""
    milestones = config.training.lr_milestones
    if any(milestones[index] >= milestones[index + 1] for index in range(len(milestones) - 1)):
        raise ConfigError(
            "training.lr_milestones: milestones must be strictly increasing, "
            f"got {list(milestones)}"
        )
    if config.experiment not in contracts.EXPERIMENT_SPECS:
        raise ConfigError(f"experiment: unknown experiment {config.experiment!r}")
    expected = expected_config(config.experiment)
    paths = differing_paths(config, expected)
    if not paths:
        return
    details = [
        f"{path}: expected {_lookup(expected, path)!r}, got {_lookup(config, path)!r}"
        for path in paths
    ]
    raise ConfigError("config departs from the locked study: " + "; ".join(details))


def _scientific_paths(left: ExperimentConfig, right: ExperimentConfig) -> tuple[str, ...]:
    return tuple(path for path in differing_paths(left, right) if path != "experiment")


def _assert_path_set(label: str, actual: tuple[str, ...], expected: frozenset[str]) -> None:
    actual_set = set(actual)
    if actual_set == expected:
        return
    unexpected = sorted(actual_set - expected)
    missing = sorted(expected - actual_set)
    raise ConfigError(
        f"{label}: unexpected fields {unexpected}; missing fields {missing}; "
        f"actual {list(actual)}"
    )


def _assert_triple(
    label: str,
    actual: tuple[object, object, object],
    expected: tuple[object, object, object],
) -> None:
    if actual != expected:
        raise ConfigError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_registered_differences(
    e0: ExperimentConfig,
    e1: ExperimentConfig,
    e2: ExperimentConfig,
) -> None:
    """Validate each config, then check the pre-registered differences.

    ``experiment`` is an identity field. It is required to differ and is not a
    scientific difference. E1's scientific differences from E0 are the epoch
    budget and learning-rate milestones. E2's scientific difference from E0 is
    ``model.stage_extra_convolutions``.
    """
    for config in (e0, e1, e2):
        assert_locked(config)
    identities = (e0.experiment, e1.experiment, e2.experiment)
    if identities != _EXPERIMENT_IDS:
        raise ConfigError(f"experiment: expected {_EXPERIMENT_IDS}, got {identities}")
    studies = (e0.study, e1.study, e2.study)
    if studies != (contracts.STUDY_ID, contracts.STUDY_ID, contracts.STUDY_ID):
        raise ConfigError(f"study: expected {contracts.STUDY_ID!r} for every config, got {studies}")
    if "experiment" not in differing_paths(e0, e1) or "experiment" not in differing_paths(e0, e2):
        raise ConfigError("experiment: E0, E1, and E2 must identify distinct experiments")

    _assert_path_set("E1 vs E0", _scientific_paths(e0, e1), _E1_SCIENTIFIC_DIFFERENCES)
    _assert_path_set("E2 vs E0", _scientific_paths(e0, e2), _E2_SCIENTIFIC_DIFFERENCES)
    _assert_path_set("E2 vs E1", _scientific_paths(e1, e2), _E2_VS_E1_SCIENTIFIC_DIFFERENCES)

    e0_spec = contracts.EXPERIMENT_SPECS["e0"]
    e1_spec = contracts.EXPERIMENT_SPECS["e1"]
    e2_spec = contracts.EXPERIMENT_SPECS["e2"]
    _assert_triple(
        "training.epochs",
        (e0.training.epochs, e1.training.epochs, e2.training.epochs),
        (e0_spec.epochs, e1_spec.epochs, e2_spec.epochs),
    )
    _assert_triple(
        "training.lr_milestones",
        (
            e0.training.lr_milestones,
            e1.training.lr_milestones,
            e2.training.lr_milestones,
        ),
        (e0_spec.lr_milestones, e1_spec.lr_milestones, e2_spec.lr_milestones),
    )
    _assert_triple(
        "model.stage_extra_convolutions",
        (
            e0.model.stage_extra_convolutions,
            e1.model.stage_extra_convolutions,
            e2.model.stage_extra_convolutions,
        ),
        (
            e0_spec.stage_extra_convolutions,
            e1_spec.stage_extra_convolutions,
            e2_spec.stage_extra_convolutions,
        ),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load one complete config and reject anything off the locked study."""
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{config_path}: cannot read config ({exc.strerror})") from exc
    try:
        # _UniqueKeyLoader subclasses SafeLoader and rejects duplicate keys.
        document = yaml.load(text, Loader=_UniqueKeyLoader)
    except ConfigError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path}: invalid YAML: {exc}") from exc
    try:
        config = _parse_document(document)
        assert_locked(config)
    except ConfigError as exc:
        raise ConfigError(f"{config_path}: {exc}") from exc
    return config


def load_study_configs(
    directory: str | Path,
) -> tuple[ExperimentConfig, ExperimentConfig, ExperimentConfig]:
    """Load ``e0.yaml``, ``e1.yaml``, and ``e2.yaml`` and compare them."""
    config_dir = Path(directory)
    loaded: list[ExperimentConfig] = []
    for experiment_id in _EXPERIMENT_IDS:
        path = config_dir / f"{experiment_id}.yaml"
        config = load_config(path)
        if config.experiment != experiment_id:
            raise ConfigError(
                f"{path}: experiment id is {config.experiment!r}, expected {experiment_id!r}"
            )
        loaded.append(config)
    e0, e1, e2 = loaded
    assert_registered_differences(e0, e1, e2)
    return e0, e1, e2
