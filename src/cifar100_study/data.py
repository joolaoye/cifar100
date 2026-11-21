"""CIFAR-100 loading, one-hot labels, and the shared preprocessing cache.

Official loading uses ``keras.datasets.cifar100.load_data(label_mode="fine")``
and imports Keras only inside that function. ZCA statistics are fit on the
training partition. The cached test partition is written with those training
parameters and is loaded only through ``load_test_data`` with the
``confirm-final-test`` token.
"""

import hashlib
import json
import os
import shutil
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cifar100_study import contracts
from cifar100_study.config import ConfigError, ExperimentConfig, load_config, load_study_configs
from cifar100_study.preprocess import (
    CHUNK_SIZE,
    PreprocessError,
    fit_zca_images,
    transform_images,
)

TRAIN_IMAGES = "train_images.npy"
TRAIN_LABELS = "train_labels.npy"
TEST_IMAGES = "test_images.npy"
TEST_LABELS = "test_labels.npy"
ZCA_MEAN = "zca_mean.npy"
ZCA_MATRIX = "zca_matrix.npy"
MANIFEST_NAME = "manifest.json"
ARTIFACT_NAMES = (
    TRAIN_IMAGES,
    TRAIN_LABELS,
    TEST_IMAGES,
    TEST_LABELS,
    ZCA_MEAN,
    ZCA_MATRIX,
)
TRAINING_ARTIFACTS = (TRAIN_IMAGES, TRAIN_LABELS)
TEST_ARTIFACTS = (TEST_IMAGES, TEST_LABELS)
MANIFEST_KEYS = (
    "artifacts",
    "format_version",
    "identity",
    "library_versions",
    "preprocessing_version",
    "source_fingerprint",
)


class DataError(ValueError):
    """Dataset validation, cache integrity, or test-access guard failed."""


@dataclass(frozen=True, slots=True)
class PrepareResult:
    """A published preprocessing cache."""

    path: Path
    reused: bool
    train_images_shape: tuple[int, ...]
    train_labels_shape: tuple[int, ...]
    test_images_shape: tuple[int, ...]
    test_labels_shape: tuple[int, ...]
    source_fingerprint: str


def fetch_fine_split(load_data: Any) -> Any:
    """Call a Keras-style loader with the locked fine-label mode."""
    return load_data(label_mode="fine")


def load_official_cifar100() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load the official CIFAR-100 fine split. This is the only download path."""
    from keras.datasets import cifar100

    return fetch_fine_split(cifar100.load_data)


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    """Return a 1-D int64 label vector, accepting Keras ``(N, 1)`` labels."""
    if not isinstance(labels, np.ndarray):
        raise DataError(f"labels must be a numpy array, got {type(labels).__name__}")
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        raise DataError(f"labels must have shape (N,) or (N, 1), got {labels.shape}")
    if labels.dtype == np.bool_ or not np.issubdtype(labels.dtype, np.integer):
        raise DataError(f"labels must be integers, got dtype {labels.dtype}")
    return np.ascontiguousarray(labels, dtype=np.int64)


def validate_raw_images(
    images: np.ndarray,
    image_shape: tuple[int, int, int],
    expected_count: int,
    partition: str,
) -> None:
    """Reject a partition whose raw array is not uint8 NHWC of the locked shape."""
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8 or images.ndim != 4:
        raise DataError(
            f"{partition} images must be uint8 with shape (N, H, W, C), "
            f"got {getattr(images, 'dtype', None)} {getattr(images, 'shape', None)}"
        )
    if images.shape[0] != expected_count or tuple(images.shape[1:]) != tuple(image_shape):
        raise DataError(
            f"{partition} images must have shape {(expected_count, *image_shape)}, "
            f"got {images.shape}"
        )


def validate_class_counts(
    labels: np.ndarray,
    num_classes: int,
    expected_per_class: int,
    partition: str,
) -> None:
    """Require every fine class, which rejects a coarse 20-class label vector."""
    canonical = canonical_labels(labels)
    if canonical.shape[0] != expected_per_class * num_classes:
        raise DataError(
            f"{partition} labels have length {canonical.shape[0]}, "
            f"expected {expected_per_class * num_classes}"
        )
    if np.any(canonical < 0) or np.any(canonical >= num_classes):
        raise DataError(f"{partition} labels fall outside 0..{num_classes - 1}")
    counts = np.bincount(canonical, minlength=num_classes)
    if counts.shape != (num_classes,) or not np.all(counts == expected_per_class):
        raise DataError(
            f"{partition} class counts are not {expected_per_class} for each of "
            f"{num_classes} fine classes"
        )


def encode_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Encode integer fine labels as float32 one-hot rows."""
    if num_classes < 1:
        raise DataError(f"num_classes must be positive, got {num_classes}")
    canonical = canonical_labels(labels)
    if np.any(canonical < 0) or np.any(canonical >= num_classes):
        raise DataError(f"labels fall outside 0..{num_classes - 1}")
    encoded = np.zeros((canonical.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(canonical.shape[0]), canonical] = np.float32(1.0)
    _assert_one_hot(canonical, encoded, num_classes)
    return encoded


def preprocessing_identity(config: ExperimentConfig) -> dict[str, Any]:
    """Identity shared by E0, E1, and E2. It excludes schedule and depth."""
    return {
        "augmentation": config.dataset.augmentation,
        "covariance_divisor": config.preprocessing.covariance_divisor,
        "dataset_name": config.dataset.name,
        "dtype": config.preprocessing.dtype,
        "fit_on": config.preprocessing.fit_on,
        "gcn": config.preprocessing.gcn,
        "gcn_min_divisor": config.preprocessing.gcn_min_divisor,
        "image_shape": list(config.dataset.image_shape),
        "label_encoding": config.dataset.label_encoding,
        "label_mode": config.dataset.label_mode,
        "num_classes": config.dataset.num_classes,
        "preprocessing_version": contracts.PREPROCESSING_VERSION,
        "rescale_from": list(config.preprocessing.rescale_from),
        "rescale_to": list(config.preprocessing.rescale_to),
        "test_size": config.dataset.test_size,
        "train_size": config.dataset.train_size,
        "validation_split": config.dataset.validation_split,
        "zca_form": config.preprocessing.zca_form,
        "zca_regularization": config.preprocessing.zca_regularization,
        "zca_whitening": config.preprocessing.zca_whitening,
    }


def source_fingerprint(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
) -> str:
    """SHA-256 of the raw partitions in a fixed order."""
    hasher = hashlib.sha256()
    pairs = (
        ("train_images", np.ascontiguousarray(train_images)),
        ("train_labels", canonical_labels(train_labels)),
        ("test_images", np.ascontiguousarray(test_images)),
        ("test_labels", canonical_labels(test_labels)),
    )
    for name, array in pairs:
        hasher.update(name.encode("utf-8"))
        hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def cache_key(identity: Mapping[str, Any], fingerprint: str) -> str:
    """Key from dataset identity, preprocessing settings, and source bytes."""
    payload = {"identity": dict(identity), "source_fingerprint": fingerprint}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def prepare_cache(
    identity: Mapping[str, Any],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    cache_root: str | Path,
    *,
    chunk_size: int = CHUNK_SIZE,
    require_class_balance: bool = False,
) -> PrepareResult:
    """Fit ZCA on training images and publish a validated cache.

    An existing valid cache is reused. A corrupt cache at the same key is an
    error. Files are written in a temporary directory and renamed only after
    the manifest and hashes validate, so an interruption does not publish a
    partial cache.
    """
    _require_locked_preprocessing(identity)
    image_shape = tuple(identity["image_shape"])
    num_classes = int(identity["num_classes"])
    train_count = int(identity["train_size"])
    test_count = int(identity["test_size"])
    validate_raw_images(train_images, image_shape, train_count, "training")
    validate_raw_images(test_images, image_shape, test_count, "test")
    canonical_train = canonical_labels(train_labels)
    canonical_test = canonical_labels(test_labels)
    if canonical_train.shape != (train_count,) or canonical_test.shape != (test_count,):
        raise DataError(
            f"label counts {(canonical_train.shape[0], canonical_test.shape[0])} "
            f"do not match {(train_count, test_count)}"
        )
    _validate_label_range(canonical_train, num_classes, "training")
    _validate_label_range(canonical_test, num_classes, "test")
    if require_class_balance:
        validate_class_counts(canonical_train, num_classes, train_count // num_classes, "training")
        validate_class_counts(canonical_test, num_classes, test_count // num_classes, "test")

    fingerprint = source_fingerprint(train_images, canonical_train, test_images, canonical_test)
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / cache_key(identity, fingerprint)
    if final.exists():
        _validate_cache(final, identity, fingerprint)
        return _result_from_manifest(final, fingerprint, reused=True)

    temporary = root / f".partial-{final.name}-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(parents=False)
        mean, whitening = fit_zca_images(
            train_images,
            float(identity["gcn_min_divisor"]),
            float(identity["zca_regularization"]),
            chunk_size,
        )
        transformed_train = transform_images(
            train_images,
            mean,
            whitening,
            float(identity["gcn_min_divisor"]),
            chunk_size,
        )
        transformed_test = transform_images(
            test_images,
            mean,
            whitening,
            float(identity["gcn_min_divisor"]),
            chunk_size,
        )
        encoded_train = encode_one_hot(canonical_train, num_classes)
        encoded_test = encode_one_hot(canonical_test, num_classes)
        arrays = {
            TRAIN_IMAGES: transformed_train,
            TRAIN_LABELS: encoded_train,
            TEST_IMAGES: transformed_test,
            TEST_LABELS: encoded_test,
            ZCA_MEAN: mean,
            ZCA_MATRIX: whitening,
        }
        _write_arrays(temporary, arrays)
        manifest = _manifest(identity, fingerprint, temporary, arrays)
        _write_manifest(temporary / MANIFEST_NAME, manifest)
        _validate_cache(temporary, identity, fingerprint, check_key=False)
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _result_from_manifest(final, fingerprint, reused=False)


def prepare_official(
    config: ExperimentConfig,
    cache_root: str | Path,
    *,
    loader: Any = None,
    chunk_size: int = CHUNK_SIZE,
) -> PrepareResult:
    """Prepare the official fine split described by a locked experiment config."""
    if loader is None:
        loader = load_official_cifar100
    (train_images, train_labels), (test_images, test_labels) = loader()
    return prepare_cache(
        preprocessing_identity(config),
        train_images,
        train_labels,
        test_images,
        test_labels,
        cache_root,
        chunk_size=chunk_size,
        require_class_balance=True,
    )


def prepare_command(config_path: str | Path, cache_root: str | Path) -> PrepareResult:
    """Validate the study configs, then prepare the shared cache."""
    path = Path(config_path)
    config = load_config(path)
    load_study_configs(path.parent)
    try:
        return prepare_official(config, cache_root)
    except (DataError, PreprocessError, ConfigError):
        raise
    except Exception as exc:
        raise DataError(f"preparation failed: {exc}") from exc


def artifact_schemas(identity: Mapping[str, Any]) -> dict[str, tuple[tuple[int, ...], str]]:
    """Return array schemas implied by a trusted identity.

    Counts and the image shape come from that identity. Class count, image
    dtype, label dtype, and ZCA parameter dtype come from the locked contract.
    Nothing is read from manifest artifact declarations.
    """
    _require_locked_preprocessing(identity)
    num_classes = identity.get("num_classes")
    if num_classes != contracts.NUM_CLASSES:
        raise DataError(
            f"num_classes must be {contracts.NUM_CLASSES}, got {num_classes!r}"
        )
    if _matches_official_dimensions(identity):
        image_shape = contracts.IMAGE_SHAPE
        train_size = contracts.TRAIN_SIZE
        test_size = contracts.TEST_SIZE
    else:
        image_shape = _require_image_shape(identity)
        train_size = _require_count(identity, "train_size")
        test_size = _require_count(identity, "test_size")
    features = image_shape[0] * image_shape[1] * image_shape[2]
    return {
        TRAIN_IMAGES: ((train_size, *image_shape), contracts.DTYPE),
        TRAIN_LABELS: ((train_size, contracts.NUM_CLASSES), "float32"),
        TEST_IMAGES: ((test_size, *image_shape), contracts.DTYPE),
        TEST_LABELS: ((test_size, contracts.NUM_CLASSES), "float32"),
        ZCA_MEAN: ((features,), "float64"),
        ZCA_MATRIX: ((features, features), "float64"),
    }


def load_training_data(
    cache_dir: str | Path,
    *,
    trusted_identity: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached training images and labels without opening test arrays.

    Official caches are checked against the CIFAR-100 contract. A synthetic
    fixture must pass the same trusted identity that was used to build it.
    Test artifact declarations are checked from the manifest only.
    """
    directory = Path(cache_dir)
    manifest = _read_manifest(directory)
    _assert_published_cache(directory, manifest)
    schemas = artifact_schemas(_identity_for_schema(manifest, trusted_identity))
    _require_declared_schemas(manifest, schemas)
    images = _load_checked_array(directory, TRAIN_IMAGES, manifest, schemas[TRAIN_IMAGES])
    labels = _load_checked_array(directory, TRAIN_LABELS, manifest, schemas[TRAIN_LABELS])
    return images, labels


def load_test_data(
    cache_dir: str | Path,
    *,
    confirm_final_test: str | None = None,
    trusted_identity: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached test arrays after an explicit confirmation token."""
    if confirm_final_test != contracts.FINAL_TEST_GUARD:
        raise DataError(
            "loading cached official test arrays requires the token "
            f"{contracts.FINAL_TEST_GUARD!r}"
        )
    directory = Path(cache_dir)
    manifest = _read_manifest(directory)
    _assert_published_cache(directory, manifest)
    schemas = artifact_schemas(_identity_for_schema(manifest, trusted_identity))
    _require_declared_schemas(manifest, schemas)
    images = _load_checked_array(directory, TEST_IMAGES, manifest, schemas[TEST_IMAGES])
    labels = _load_checked_array(directory, TEST_LABELS, manifest, schemas[TEST_LABELS])
    return images, labels


def _require_locked_preprocessing(identity: Mapping[str, Any]) -> None:
    checks = {
        "preprocessing_version": contracts.PREPROCESSING_VERSION,
        "augmentation": contracts.AUGMENTATION,
        "validation_split": contracts.VALIDATION_SPLIT,
        "label_encoding": contracts.LABEL_ENCODING,
        "label_mode": contracts.LABEL_MODE,
        "gcn": contracts.GCN,
        "gcn_min_divisor": contracts.GCN_MIN_DIVISOR,
        "zca_whitening": contracts.ZCA_WHITENING,
        "zca_form": contracts.ZCA_FORM,
        "zca_regularization": contracts.ZCA_REGULARIZATION,
        "covariance_divisor": contracts.COVARIANCE_DIVISOR,
        "fit_on": contracts.FIT_ON,
        "dtype": contracts.DTYPE,
        "rescale_from": list(contracts.RESCALE_FROM),
        "rescale_to": list(contracts.RESCALE_TO),
    }
    for key, expected in checks.items():
        if key not in identity or identity[key] != expected:
            raise DataError(
                f"preprocessing identity {key} must be {expected!r}, got {identity.get(key)!r}"
            )


def _validate_label_range(labels: np.ndarray, num_classes: int, partition: str) -> None:
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise DataError(f"{partition} labels fall outside 0..{num_classes - 1}")


def _assert_one_hot(labels: np.ndarray, encoded: np.ndarray, num_classes: int) -> None:
    if encoded.dtype != np.float32 or encoded.shape != (labels.shape[0], num_classes):
        raise DataError(f"one-hot labels must be float32 {(labels.shape[0], num_classes)}")
    if not np.all(encoded.sum(axis=1) == 1.0) or not np.all(np.count_nonzero(encoded, axis=1) == 1):
        raise DataError("one-hot rows must contain exactly one active class")
    if not np.array_equal(encoded.argmax(axis=1), labels):
        raise DataError("one-hot rows do not match the original class ids")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _library_versions() -> dict[str, str]:
    versions = {"numpy": np.__version__, "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"}
    if "keras" in sys.modules:
        versions["keras"] = sys.modules["keras"].__version__
    if "tensorflow" in sys.modules:
        versions["tensorflow"] = sys.modules["tensorflow"].__version__
    return versions


def _write_arrays(directory: Path, arrays: Mapping[str, np.ndarray]) -> None:
    for name, array in arrays.items():
        np.save(directory / name, np.ascontiguousarray(array), allow_pickle=False)


def _manifest(
    identity: Mapping[str, Any],
    fingerprint: str,
    directory: Path,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    artifacts = {}
    for name, array in arrays.items():
        artifacts[name] = {
            "dtype": str(array.dtype),
            "sha256": _sha256_file(directory / name),
            "shape": list(array.shape),
        }
    return {
        "artifacts": artifacts,
        "format_version": 1,
        "identity": dict(identity),
        "library_versions": _library_versions(),
        "preprocessing_version": contracts.PREPROCESSING_VERSION,
        "source_fingerprint": fingerprint,
    }


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise DataError(f"{directory} has no {MANIFEST_NAME}; the cache is incomplete")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or tuple(sorted(manifest)) != MANIFEST_KEYS:
        raise DataError(f"{path} does not have the required manifest fields {MANIFEST_KEYS}")
    if manifest["format_version"] != 1:
        raise DataError(f"{path} has unsupported format_version {manifest['format_version']!r}")
    if manifest["preprocessing_version"] != contracts.PREPROCESSING_VERSION:
        raise DataError(
            f"{path} preprocessing_version is {manifest['preprocessing_version']!r}, "
            f"expected {contracts.PREPROCESSING_VERSION!r}"
        )
    return manifest


def _assert_published_cache(directory: Path, manifest: Mapping[str, Any]) -> None:
    expected_name = cache_key(manifest["identity"], manifest["source_fingerprint"])
    if directory.name != expected_name:
        raise DataError(
            f"{directory} is not the published cache for its manifest "
            f"(expected directory name {expected_name})"
        )
    _require_locked_preprocessing(manifest["identity"])


def _validate_cache(
    directory: Path,
    identity: Mapping[str, Any],
    fingerprint: str,
    *,
    check_key: bool = True,
) -> dict[str, Any]:
    if not directory.is_dir():
        raise DataError(f"cache directory does not exist: {directory}")
    names = {path.name for path in directory.iterdir()}
    expected = set(ARTIFACT_NAMES) | {MANIFEST_NAME}
    if names != expected:
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        raise DataError(f"{directory} cache files are incomplete or unexpected: missing {missing}, extra {extra}")
    manifest = _read_manifest(directory)
    if check_key:
        _assert_published_cache(directory, manifest)
    if manifest["source_fingerprint"] != fingerprint:
        raise DataError(f"{directory} source fingerprint does not match the raw partitions")
    if _canonical_json(manifest["identity"]) != _canonical_json(dict(identity)):
        raise DataError(f"{directory} preprocessing identity does not match this preparation")
    schemas = artifact_schemas(identity)
    _require_declared_schemas(manifest, schemas)
    for name in ARTIFACT_NAMES:
        _load_checked_array(directory, name, manifest, schemas[name])
    return manifest


def _matches_official_dimensions(identity: Mapping[str, Any]) -> bool:
    return (
        identity.get("train_size") == contracts.TRAIN_SIZE
        and identity.get("test_size") == contracts.TEST_SIZE
        and identity.get("image_shape") == list(contracts.IMAGE_SHAPE)
    )


def _require_official_identity(identity: Mapping[str, Any]) -> None:
    expected = {
        "dataset_name": contracts.DATASET_NAME,
        "label_mode": contracts.LABEL_MODE,
        "num_classes": contracts.NUM_CLASSES,
        "image_shape": list(contracts.IMAGE_SHAPE),
        "train_size": contracts.TRAIN_SIZE,
        "test_size": contracts.TEST_SIZE,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise DataError(
                f"official cache identity {key} must be {value!r}, got {identity.get(key)!r}"
            )


def _identity_for_schema(
    manifest: Mapping[str, Any],
    trusted_identity: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    manifest_identity = manifest["identity"]
    if not isinstance(manifest_identity, dict):
        raise DataError("cache manifest identity must be an object")
    if trusted_identity is None:
        _require_official_identity(manifest_identity)
        return manifest_identity
    if _canonical_json(manifest_identity) != _canonical_json(dict(trusted_identity)):
        raise DataError("manifest identity does not match the trusted fixture identity")
    return trusted_identity


def _require_count(identity: Mapping[str, Any], key: str) -> int:
    value = identity.get(key)
    if type(value) is not int or value < 1:
        raise DataError(f"{key} must be a positive integer, got {value!r}")
    return value


def _require_image_shape(identity: Mapping[str, Any]) -> tuple[int, int, int]:
    value = identity.get("image_shape")
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(type(item) is not int or item < 1 for item in value)
    ):
        raise DataError(f"image_shape must be three positive integers, got {value!r}")
    return (value[0], value[1], value[2])


def _require_declared_schemas(
    manifest: Mapping[str, Any],
    schemas: Mapping[str, tuple[tuple[int, ...], str]],
) -> None:
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(schemas):
        raise DataError("manifest artifact list does not match the required cache artifacts")
    for name, (shape, dtype) in schemas.items():
        info = artifacts[name]
        if not isinstance(info, dict):
            raise DataError(f"{name} manifest declaration must be an object")
        declared_shape = info.get("shape")
        declared_dtype = info.get("dtype")
        expected_shape = list(shape)
        if (
            not isinstance(declared_shape, list)
            or any(type(dim) is not int for dim in declared_shape)
            or declared_shape != expected_shape
            or type(declared_dtype) is not str
            or declared_dtype != dtype
        ):
            raise DataError(
                f"{name} manifest declares shape {declared_shape} dtype {declared_dtype}, "
                f"authoritative schema requires {expected_shape} {dtype}"
            )


def _load_checked_array(
    directory: Path,
    name: str,
    manifest: Mapping[str, Any],
    schema: tuple[tuple[int, ...], str],
) -> np.ndarray:
    info = manifest["artifacts"][name]
    path = directory / name
    digest = _sha256_file(path)
    if digest != info.get("sha256"):
        raise DataError(f"{path} SHA-256 is {digest}, manifest expected {info.get('sha256')}")
    array = np.load(path, allow_pickle=False)
    expected_shape, expected_dtype = schema
    if tuple(array.shape) != tuple(expected_shape) or str(array.dtype) != expected_dtype:
        raise DataError(
            f"{path} has shape {tuple(array.shape)} dtype {array.dtype}, "
            f"authoritative schema requires {tuple(expected_shape)} {expected_dtype}"
        )
    return array


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _result_from_manifest(directory: Path, fingerprint: str, reused: bool) -> PrepareResult:
    manifest = _read_manifest(directory)
    artifacts = manifest["artifacts"]
    return PrepareResult(
        path=directory,
        reused=reused,
        train_images_shape=tuple(artifacts[TRAIN_IMAGES]["shape"]),
        train_labels_shape=tuple(artifacts[TRAIN_LABELS]["shape"]),
        test_images_shape=tuple(artifacts[TEST_IMAGES]["shape"]),
        test_labels_shape=tuple(artifacts[TEST_LABELS]["shape"]),
        source_fingerprint=fingerprint,
    )
