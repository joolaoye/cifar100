"""CLI help stays lightweight, and preparation fails closed."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

def _help_does_not_import_tensorflow(script: str, argv: list[str]) -> None:
    probe = f"""
import sys
sys.argv = {argv!r}
import runpy
try:
    runpy.run_path({script!r}, run_name="__main__")
except SystemExit as exc:
    status = exc.code
else:
    status = 0
bad = [
    name
    for name in sys.modules
    if name == "keras"
    or name.startswith("keras.")
    or name == "tensorflow"
    or name.startswith("tensorflow.")
]
raise SystemExit(0 if status in (0, None) and not bad else 1)
"""
    probed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert probed.returncode == 0, probed.stderr


def test_evaluate_help_requires_confirmation_and_does_not_import_tensorflow():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "--run" in output
    assert "--confirm-final-test" in output
    assert "--cache-root" in output
    assert "tensorflow" not in output.lower()
    _help_does_not_import_tensorflow("scripts/evaluate.py", ["evaluate.py", "--help"])
    refused = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--run", "runs/missing"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "confirm-final-test" in (refused.stdout + refused.stderr)
    assert "tensorflow" not in (refused.stdout + refused.stderr).lower()


def test_report_help_describes_three_runs_and_does_not_import_tensorflow():
    result = subprocess.run(
        [sys.executable, "scripts/report.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "--runs" in output
    assert "--output-dir" in output
    assert "tensorflow" not in output.lower()
    _help_does_not_import_tensorflow("scripts/report.py", ["report.py", "--help"])
    refused = subprocess.run(
        [sys.executable, "scripts/report.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "tensorflow" not in (refused.stdout + refused.stderr).lower()


def test_module_help_lists_the_scripts():
    result = subprocess.run(
        [sys.executable, "-m", "cifar100_study", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "scripts/train.py" in result.stdout
    assert "does not load CIFAR-100" in result.stdout


def test_module_version():
    result = subprocess.run(
        [sys.executable, "-m", "cifar100_study", "--version"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "0.0.0" in result.stdout


def test_prepare_help_describes_the_cache_and_does_not_import_keras():
    result = subprocess.run(
        [sys.executable, "scripts/prepare.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "--config" in output
    assert "--cache-root" in output
    assert "tensorflow" not in output.lower()

    probe = """
import sys
sys.argv = ["prepare.py", "--help"]
import runpy
try:
    runpy.run_path("scripts/prepare.py", run_name="__main__")
except SystemExit as exc:
    status = exc.code
else:
    status = 0
bad = [
    name
    for name in sys.modules
    if name == "keras"
    or name.startswith("keras.")
    or name == "tensorflow"
    or name.startswith("tensorflow.")
]
raise SystemExit(0 if status in (0, None) and not bad else 1)
"""
    probed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert probed.returncode == 0, probed.stderr


def test_train_help_describes_resume_and_does_not_import_tensorflow():
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "--config" in output
    assert "--resume" in output
    assert "--device" in output
    assert "--cache-root" in output
    assert "tensorflow" not in output.lower()

    probe = """
import sys
sys.argv = ["train.py", "--help"]
import runpy
try:
    runpy.run_path("scripts/train.py", run_name="__main__")
except SystemExit as exc:
    status = exc.code
else:
    status = 0
bad = [
    name
    for name in sys.modules
    if name == "keras"
    or name.startswith("keras.")
    or name == "tensorflow"
    or name.startswith("tensorflow.")
]
raise SystemExit(0 if status in (0, None) and not bad else 1)
"""
    probed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert probed.returncode == 0, probed.stderr

    refused = subprocess.run(
        [sys.executable, "scripts/train.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "tensorflow" not in (refused.stdout + refused.stderr).lower()


def test_prepare_rejects_missing_arguments_and_config():
    missing_args = subprocess.run(
        [sys.executable, "scripts/prepare.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_args.returncode != 0

    missing_file = subprocess.run(
        [sys.executable, "scripts/prepare.py", "--config", "configs/missing.yaml"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_file.returncode != 0
    assert "error" in (missing_file.stdout + missing_file.stderr).lower()


def test_prepare_command_exits_nonzero_when_the_split_is_short(tmp_path, monkeypatch):
    import importlib.util

    import numpy as np

    spec = importlib.util.spec_from_file_location("prepare_cli", ROOT / "scripts" / "prepare.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_loader():
        images = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        labels = np.zeros((1, 1), dtype=np.uint8)
        return (images, labels), (images, labels)

    monkeypatch.setattr("cifar100_study.data.load_official_cifar100", fake_loader)
    with pytest.raises(SystemExit) as caught:
        module.main(["--config", str(ROOT / "configs" / "e0.yaml"), "--cache-root", str(tmp_path)])
    assert caught.value.code == 1
    assert list(tmp_path.iterdir()) == []
