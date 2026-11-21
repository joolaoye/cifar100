"""Synthetic dataset, label, cache, and leakage tests. No CIFAR-100 download."""

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cifar100_study.config import load_config, load_study_configs
from cifar100_study.data import (
    ARTIFACT_NAMES,
    MANIFEST_NAME,
    DataError,
    artifact_schemas,
    cache_key,
    encode_one_hot,
    fetch_fine_split,
    load_test_data,
    load_training_data,
    prepare_cache,
    prepare_official,
    preprocessing_identity,
    source_fingerprint,
    validate_class_counts,
    validate_raw_images,
)
from cifar100_study.preprocess import fit_zca_images

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
IMAGE_SHAPE = (2, 2, 1)


def _identity(train_size: int, test_size: int) -> dict:
    identity = preprocessing_identity(load_config(CONFIG_DIR / "e0.yaml"))
    identity["train_size"] = train_size
    identity["test_size"] = test_size
    identity["image_shape"] = list(IMAGE_SHAPE)
    return identity


def _partitions(train_size: int = 4, test_size: int = 2):
    train_images = np.arange(train_size * 4, dtype=np.uint8).reshape(train_size, *IMAGE_SHAPE)
    test_images = np.arange(test_size * 4, dtype=np.uint8).reshape(test_size, *IMAGE_SHAPE) + np.uint8(11)
    train_labels = np.array([[0], [3], [99], [7]], dtype=np.uint8)[:train_size]
    test_labels = np.array([[4], [99]], dtype=np.uint8)[:test_size]
    return train_images, train_labels, test_images, test_labels


def _prepare(tmp_path: Path, **overrides):
    train_images, train_labels, test_images, test_labels = _partitions()
    identity = _identity(train_images.shape[0], test_images.shape[0])
    result = prepare_cache(
        identity,
        train_images,
        train_labels,
        test_images,
        test_labels,
        tmp_path,
        chunk_size=2,
        **overrides,
    )
    return result, identity, train_images, train_labels, test_images, test_labels


def test_fine_loader_requests_fine_labels_and_preserves_the_return_value():
    seen = []

    def load_data(label_mode):
        seen.append(label_mode)
        return "official-split"

    assert fetch_fine_split(load_data) == "official-split"
    assert seen == ["fine"]


def test_raw_validation_rejects_bad_shape_dtype_and_label_range():
    images = np.zeros((2, *IMAGE_SHAPE), dtype=np.uint8)
    validate_raw_images(images, IMAGE_SHAPE, 2, "training")
    with pytest.raises(DataError, match="uint8"):
        validate_raw_images(images.astype(np.float32), IMAGE_SHAPE, 2, "training")
    with pytest.raises(DataError, match="shape"):
        validate_raw_images(images[:1], IMAGE_SHAPE, 2, "training")
    with pytest.raises(DataError, match="outside"):
        encode_one_hot(np.array([0, 100]), 100)
    with pytest.raises(DataError, match="outside"):
        encode_one_hot(np.array([-1]), 100)
    with pytest.raises(DataError, match="integers"):
        encode_one_hot(np.array([0.0, 1.0]), 100)
    with pytest.raises(DataError, match="integers"):
        encode_one_hot(np.array([True, False]), 100)


def test_one_hot_rows_match_integer_and_column_labels():
    labels = np.array([[3], [0], [99]], dtype=np.uint8)
    encoded = encode_one_hot(labels, 100)
    assert encoded.shape == (3, 100)
    assert encoded.dtype == np.float32
    assert np.all(encoded.sum(axis=1) == 1.0)
    assert np.all(np.count_nonzero(encoded, axis=1) == 1)
    assert list(encoded.argmax(axis=1)) == [3, 0, 99]


def test_class_counts_reject_a_coarse_style_label_vector():
    coarse = np.repeat(np.arange(20), 5)
    with pytest.raises(DataError, match="fine classes"):
        validate_class_counts(coarse, 100, 1, "training")
    balanced = np.arange(100)
    validate_class_counts(balanced, 100, 1, "training")
    with pytest.raises(DataError, match="length"):
        validate_class_counts(np.arange(20), 100, 500, "training")


def test_preprocessing_identity_is_shared_and_excludes_schedule_and_depth():
    e0, e1, e2 = load_study_configs(CONFIG_DIR)
    first = preprocessing_identity(e0)
    assert first == preprocessing_identity(e1) == preprocessing_identity(e2)
    blob = json.dumps(first)
    for banned in ("epochs", "lr_milestones", "stage_extra", "experiment", "seed"):
        assert banned not in blob
    fingerprint = "a" * 64
    assert cache_key(first, fingerprint) == cache_key(preprocessing_identity(e1), fingerprint)
    assert cache_key(first, fingerprint) == cache_key(preprocessing_identity(e2), fingerprint)


def test_cache_publish_reuse_and_deterministic_bytes(tmp_path):
    result, identity, train_images, train_labels, test_images, test_labels = _prepare(tmp_path / "one")
    assert result.reused is False
    assert result.path.name == cache_key(identity, result.source_fingerprint)
    assert result.train_images_shape == (4, *IMAGE_SHAPE)
    assert result.train_labels_shape == (4, 100)
    assert result.test_images_shape == (2, *IMAGE_SHAPE)
    assert result.test_labels_shape == (2, 100)
    assert set(path.name for path in result.path.iterdir()) == set(ARTIFACT_NAMES) | {MANIFEST_NAME}

    again = prepare_cache(
        identity,
        train_images,
        train_labels,
        test_images,
        test_labels,
        tmp_path / "one",
        chunk_size=2,
    )
    assert again.reused is True
    assert again.path == result.path

    other, *_rest = _prepare(tmp_path / "two")
    for name in (*ARTIFACT_NAMES, MANIFEST_NAME):
        assert (result.path / name).read_bytes() == (other.path / name).read_bytes()

    manifest = json.loads((result.path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["identity"]["label_mode"] == "fine"
    assert manifest["identity"]["zca_regularization"] == 0.1
    assert manifest["identity"]["gcn_min_divisor"] == 1e-8
    assert manifest["identity"]["covariance_divisor"] == "N"
    assert manifest["preprocessing_version"] == "gcn-zca-v1"
    assert "numpy" in manifest["library_versions"]
    assert "python" in manifest["library_versions"]


def test_saved_zca_parameters_depend_only_on_training_images(tmp_path):
    train_images = np.array(
        [
            [[[0], [255]], [[10], [20]]],
            [[[5], [1]], [[90], [2]]],
            [[[30], [40]], [[0], [200]]],
            [[[8], [80]], [[15], [3]]],
        ],
        dtype=np.uint8,
    )
    test_images = np.array(
        [
            [[[255], [0]], [[128], [64]]],
            [[[7], [9]], [[11], [250]]],
        ],
        dtype=np.uint8,
    )
    train_labels = np.array([[0], [3], [99], [7]], dtype=np.uint8)
    test_labels = np.array([[4], [99]], dtype=np.uint8)
    identity = _identity(4, 2)
    result = prepare_cache(
        identity,
        train_images,
        train_labels,
        test_images,
        test_labels,
        tmp_path,
        chunk_size=2,
    )
    saved_mean = np.load(result.path / "zca_mean.npy")
    saved_matrix = np.load(result.path / "zca_matrix.npy")
    mean, matrix = fit_zca_images(train_images, 1e-8, 0.1, chunk_size=2)
    np.testing.assert_allclose(saved_mean, mean)
    np.testing.assert_allclose(saved_matrix, matrix)
    mixed_mean, mixed_matrix = fit_zca_images(
        np.concatenate([train_images, test_images]),
        1e-8,
        0.1,
        chunk_size=2,
    )
    assert not np.allclose(saved_mean, mixed_mean)
    assert not np.allclose(saved_matrix, mixed_matrix)

    changed = test_images.copy()
    changed[0, 0, 0, 0] ^= np.uint8(1)
    other = prepare_cache(
        identity,
        train_images,
        train_labels,
        changed,
        test_labels,
        tmp_path,
        chunk_size=2,
    )
    assert other.path != result.path
    np.testing.assert_allclose(np.load(other.path / "zca_mean.npy"), mean)
    np.testing.assert_allclose(np.load(other.path / "zca_matrix.npy"), matrix)


def test_training_load_does_not_open_test_arrays_and_test_load_is_guarded(tmp_path, monkeypatch):
    result, _identity, _train_images, train_labels, _test_images, test_labels = _prepare(tmp_path)
    opened: list[str] = []
    real_open = Path.open

    def spy_open(self, *args, **kwargs):
        opened.append(self.name)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)
    with pytest.raises(DataError, match="confirm-final-test"):
        load_test_data(result.path)
    with pytest.raises(DataError, match="confirm-final-test"):
        load_test_data(result.path, confirm_final_test="yes")
    assert opened == []

    images, labels = load_training_data(result.path, trusted_identity=_identity)
    assert "test_images.npy" not in opened
    assert "test_labels.npy" not in opened
    assert images.shape == (4, *IMAGE_SHAPE)
    assert images.dtype == np.float32
    assert list(labels.argmax(axis=1)) == [int(value) for value in train_labels.reshape(-1)]

    opened.clear()
    test_images, test_encoded = load_test_data(
        result.path,
        confirm_final_test="confirm-final-test",
        trusted_identity=_identity,
    )
    assert "test_images.npy" in opened
    assert "test_labels.npy" in opened
    assert test_images.shape == (2, *IMAGE_SHAPE)
    assert list(test_encoded.argmax(axis=1)) == [int(value) for value in test_labels.reshape(-1)]


def test_training_load_still_works_when_test_files_are_absent(tmp_path):
    result, identity, _train_images, train_labels, *_rest = _prepare(tmp_path)
    (result.path / "test_images.npy").unlink()
    (result.path / "test_labels.npy").unlink()
    _images, labels = load_training_data(result.path, trusted_identity=identity)
    assert list(labels.argmax(axis=1)) == [int(value) for value in train_labels.reshape(-1)]


def test_corrupt_mismatched_and_incomplete_caches_are_rejected(tmp_path):
    result, identity, train_images, train_labels, test_images, test_labels = _prepare(tmp_path / "cache")
    image_path = result.path / "train_images.npy"
    blob = bytearray(image_path.read_bytes())
    blob[-1] ^= 0xFF
    image_path.write_bytes(blob)
    with pytest.raises(DataError, match="SHA-256"):
        load_training_data(result.path, trusted_identity=identity)
    with pytest.raises(DataError, match="SHA-256"):
        prepare_cache(
            identity,
            train_images,
            train_labels,
            test_images,
            test_labels,
            tmp_path / "cache",
            chunk_size=2,
        )
    assert image_path.read_bytes() == bytes(blob)

    fresh, identity, train_images, train_labels, test_images, test_labels = _prepare(tmp_path / "fresh")
    manifest_path = fresh.path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["zca_regularization"] = 0.2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataError, match="published cache"):
        load_training_data(fresh.path)

    renamed = fresh.path.with_name("not-the-key")
    fresh.path.rename(renamed)
    manifest = json.loads((renamed / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["identity"]["zca_regularization"] = 0.1
    (renamed / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(DataError, match="published cache"):
        load_training_data(renamed)

    published, identity, train_images, train_labels, test_images, test_labels = _prepare(tmp_path / "extra")
    (published.path / "notes.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(DataError, match="extra"):
        prepare_cache(
            identity,
            train_images,
            train_labels,
            test_images,
            test_labels,
            tmp_path / "extra",
            chunk_size=2,
        )
    assert (published.path / "notes.txt").is_file()

    missing, identity, train_images, train_labels, test_images, test_labels = _prepare(tmp_path / "missing")
    (missing.path / "test_labels.npy").unlink()
    with pytest.raises(DataError, match="incomplete"):
        prepare_cache(
            identity,
            train_images,
            train_labels,
            test_images,
            test_labels,
            tmp_path / "missing",
            chunk_size=2,
        )


def test_interrupted_publish_leaves_no_partial_or_valid_cache(tmp_path, monkeypatch):
    def fail_replace(_source, _destination):
        raise OSError("interrupted")

    monkeypatch.setattr("cifar100_study.data.os.replace", fail_replace)
    train_images, train_labels, test_images, test_labels = _partitions()
    with pytest.raises(OSError, match="interrupted"):
        prepare_cache(
            _identity(4, 2),
            train_images,
            train_labels,
            test_images,
            test_labels,
            tmp_path,
            chunk_size=2,
        )
    assert list(tmp_path.iterdir()) == []


def test_official_preparation_rejects_a_short_split_without_writing(tmp_path):
    def loader():
        images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        labels = np.zeros((2, 1), dtype=np.uint8)
        return (images, labels), (images, labels)

    with pytest.raises(DataError, match="50000"):
        prepare_official(load_config(CONFIG_DIR / "e0.yaml"), tmp_path, loader=loader)
    assert list(tmp_path.iterdir()) == []


def test_source_fingerprint_tracks_raw_bytes_and_label_layout():
    train_images, train_labels, test_images, test_labels = _partitions()
    first = source_fingerprint(train_images, train_labels, test_images, test_labels)
    column = train_labels.reshape(-1, 1)
    assert source_fingerprint(train_images, column, test_images, test_labels) == first
    changed = train_images.copy()
    changed.flat[0] ^= np.uint8(1)
    assert source_fingerprint(changed, train_labels, test_images, test_labels) != first


def test_keras_import_is_inside_the_official_loader_only():
    path = ROOT / "src" / "cifar100_study" / "data.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else ""
            names = " ".join(alias.name for alias in node.names)
            assert "keras" not in f"{module} {names}"
            assert "tensorflow" not in f"{module} {names}"
    loader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_official_cifar100")
    imported = [
        node.module
        for node in ast.walk(loader)
        if isinstance(node, ast.ImportFrom)
    ]
    assert imported == ["keras.datasets"]
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import cifar100_study.data, sys\n"
            "bad = [name for name in sys.modules if name == 'keras' or name.startswith('keras.') "
            "or name == 'tensorflow' or name.startswith('tensorflow.')]\n"
            "raise SystemExit(1 if bad else 0)\n",
        ],
        cwd=ROOT,
        check=False,
    )
    assert probe.returncode == 0


def test_prepare_script_imports_data_only_after_argument_parsing():
    path = ROOT / "scripts" / "prepare.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else ""
            assert module != "cifar100_study.data"
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    modules = [node.module for node in main.body if isinstance(node, ast.ImportFrom)]
    assert "cifar100_study.data" in modules
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util\n"
            "from pathlib import Path\n"
            "import sys\n"
            "path = Path('scripts/prepare.py')\n"
            "spec = importlib.util.spec_from_file_location('prepare_script_probe', path)\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "bad = [name for name in sys.modules if name == 'keras' or name.startswith('keras.') "
            "or name == 'tensorflow' or name.startswith('tensorflow.')]\n"
            "raise SystemExit(1 if bad else 0)\n",
        ],
        cwd=ROOT,
        check=False,
    )
    assert probe.returncode == 0


def _retarget_artifact(cache: Path, name: str, array: np.ndarray) -> None:
    """Replace one array and make the manifest declaration agree with it."""
    path = cache / name
    saved = np.ascontiguousarray(array)
    np.save(path, saved, allow_pickle=False)
    manifest_path = cache / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][name] = {
        "dtype": str(saved.dtype),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape": list(saved.shape),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_official_config_schemas_follow_the_contract_rather_than_a_manifest():
    identity = preprocessing_identity(load_config(CONFIG_DIR / "e0.yaml"))
    assert artifact_schemas(identity) == {
        "train_images.npy": ((50_000, 32, 32, 3), "float32"),
        "train_labels.npy": ((50_000, 100), "float32"),
        "test_images.npy": ((10_000, 32, 32, 3), "float32"),
        "test_labels.npy": ((10_000, 100), "float32"),
        "zca_mean.npy": ((3072,), "float64"),
        "zca_matrix.npy": ((3072, 3072), "float64"),
    }
    assert artifact_schemas(identity) == artifact_schemas(
        preprocessing_identity(load_config(CONFIG_DIR / "e1.yaml"))
    )


def test_shortened_training_arrays_are_rejected_when_the_manifest_agrees(tmp_path):
    result, identity, train_images, train_labels, test_images, test_labels = _prepare(tmp_path)
    cached_images = np.load(result.path / "train_images.npy")
    cached_labels = np.load(result.path / "train_labels.npy")
    _retarget_artifact(result.path, "train_images.npy", cached_images[:2])
    _retarget_artifact(result.path, "train_labels.npy", cached_labels[:2])
    shortened = (result.path / "train_images.npy").read_bytes()

    with pytest.raises(DataError, match="authoritative schema"):
        load_training_data(result.path, trusted_identity=identity)
    with pytest.raises(DataError, match="authoritative schema"):
        prepare_cache(
            identity,
            train_images,
            train_labels,
            test_images,
            test_labels,
            tmp_path,
            chunk_size=2,
        )

    assert (result.path / "train_images.npy").read_bytes() == shortened
    assert np.load(result.path / "train_images.npy").shape[0] == 2
    assert not any(path.name.startswith(".partial-") for path in tmp_path.iterdir())


def test_manifest_cannot_authorize_a_20_class_or_float64_label_array(tmp_path):
    result, identity, *_rest = _prepare(tmp_path / "width")
    labels = np.zeros((4, 20), dtype=np.float32)
    labels[:, 0] = 1.0
    _retarget_artifact(result.path, "train_labels.npy", labels)
    with pytest.raises(DataError, match="train_labels.npy"):
        load_training_data(result.path, trusted_identity=identity)

    result, identity, *_rest = _prepare(tmp_path / "dtype")
    labels = np.zeros((4, 100), dtype=np.float64)
    labels[:, 0] = 1.0
    _retarget_artifact(result.path, "train_labels.npy", labels)
    with pytest.raises(DataError, match="float32"):
        load_training_data(result.path, trusted_identity=identity)


def test_manifest_cannot_authorize_a_changed_image_or_zca_schema(tmp_path):
    result, identity, *_rest = _prepare(tmp_path / "image-dtype")
    images = np.load(result.path / "train_images.npy").astype(np.float64)
    _retarget_artifact(result.path, "train_images.npy", images)
    with pytest.raises(DataError, match="train_images.npy"):
        load_training_data(result.path, trusted_identity=identity)

    result, identity, *_rest = _prepare(tmp_path / "image-shape")
    images = np.zeros((4, 2, 2, 3), dtype=np.float32)
    _retarget_artifact(result.path, "train_images.npy", images)
    with pytest.raises(DataError, match="authoritative schema"):
        load_training_data(result.path, trusted_identity=identity)

    result, identity, *_rest = _prepare(tmp_path / "zca-mean")
    mean = np.zeros((3,), dtype=np.float64)
    _retarget_artifact(result.path, "zca_mean.npy", mean)
    with pytest.raises(DataError, match="zca_mean.npy"):
        load_training_data(result.path, trusted_identity=identity)

    result, identity, *_rest = _prepare(tmp_path / "zca-matrix")
    matrix = np.zeros((4, 4), dtype=np.float32)
    _retarget_artifact(result.path, "zca_matrix.npy", matrix)
    with pytest.raises(DataError, match="zca_matrix.npy"):
        load_training_data(result.path, trusted_identity=identity)


def test_bad_test_declarations_are_rejected_without_opening_test_arrays(tmp_path, monkeypatch):
    result, identity, *_rest = _prepare(tmp_path)
    manifest_path = result.path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["test_images.npy"]["shape"] = [2, 8, 8, 3]
    manifest["artifacts"]["test_labels.npy"]["dtype"] = "float64"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    opened: list[str] = []
    real_open = Path.open

    def spy_open(self, *args, **kwargs):
        opened.append(self.name)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_open)
    with pytest.raises(DataError, match="test_images.npy"):
        load_training_data(result.path, trusted_identity=identity)
    assert "test_images.npy" not in opened
    assert "test_labels.npy" not in opened


def test_untrusted_synthetic_cache_is_not_accepted_as_official(tmp_path):
    result, _identity, *_rest = _prepare(tmp_path)
    with pytest.raises(DataError, match="official cache identity"):
        load_training_data(result.path)
