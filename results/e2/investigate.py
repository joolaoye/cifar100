"""Read-only E2 investigation; prints JSON and never trains or opens test arrays.

From the repository root:
uv run python results/e2/investigate.py --output /tmp/e2-investigation.json
Uses the first 32 cached training images on CPU, with dropout disabled.
The source runs and cache must still be available to reproduce this inspection.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import tensorflow as tf

from cifar100_study.allcnn import architecture_report
from cifar100_study.config import load_config
from cifar100_study.data import load_training_data, preprocessing_identity
from cifar100_study.evaluation import validate_completed_run


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def summarize_history(path):
    with path.open(newline="") as handle:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(handle)]
    metrics = {}
    for key in ("loss", "accuracy", "epoch_seconds"):
        low = min(rows, key=lambda row: row[key])
        high = max(rows, key=lambda row: row[key])
        metrics[key] = {
            "initial": rows[0][key], "final": rows[-1][key],
            "min": low[key], "min_first_epoch": int(low["epoch"]),
            "max": high[key], "max_first_epoch": int(high["epoch"]),
            "mean": statistics.mean(row[key] for row in rows),
        }
    final = rows[-1]
    plateau_start = len(rows)
    while plateau_start > 1 and all(
        rows[plateau_start - 2][key] == final[key] for key in ("loss", "accuracy")
    ):
        plateau_start -= 1
    transitions = []
    for milestone in (200, 250, 300):
        transitions.append({
            "zero_based_epoch": milestone,
            "last_before": rows[milestone - 1], "first_after": rows[milestone],
            "accuracy_mean_previous_10": statistics.mean(
                row["accuracy"] for row in rows[milestone - 10:milestone]),
            "accuracy_mean_next_10": statistics.mean(
                row["accuracy"] for row in rows[milestone:milestone + 10]),
        })
    return {
        "rows": len(rows),
        "contiguous_epochs": [int(row["epoch"]) for row in rows] == list(range(1, 351)),
        "all_values_finite": all(math.isfinite(v) for row in rows for v in row.values()),
        "metrics": metrics, "final_joint_plateau_first_epoch": plateau_start,
        "lr_transitions": transitions,
        "lr_schedule_matches_all_rows": all(math.isclose(
            row["learning_rate"], 0.05 * 0.1 ** sum(i >= m for m in (200, 250, 300)),
            rel_tol=1e-6) for i, row in enumerate(rows)),
        "largest_absolute_adjacent_accuracy_change": max(
            abs(b["accuracy"] - a["accuracy"]) for a, b in zip(rows, rows[1:])),
    }


def main(output_path):
    tf.config.set_visible_devices([], "GPU")
    root = Path(__file__).resolve().parents[2]
    runs = {"e0": root / "runs/e0-047fef8bbd29", "e2": root / "runs/e2-556139f16e87"}
    cache = root / "cache/cf1e2dce56d7e390fb1ce4624493380124b35efa86cae6fad4eb2e0a9ec4f2b3"
    manifest = json.loads((cache / "manifest.json").read_text())
    # This loader validates training hashes and schemas without opening test arrays.
    images, labels = load_training_data(cache)
    assert np.all((labels == 0) | (labels == 1)) and np.all(labels.sum(axis=1) == 1)
    counts = np.bincount(labels.argmax(axis=1), minlength=100)
    output = {
        "inspection_date": "2026-11-22", "diagnostic_device": "CPU",
        "dependency_lock_sha256_at_inspection": digest(root / "uv.lock"),
        "training_or_official_test_evaluation_performed": False,
        "activation_sample": {"partition": "training", "indices": list(range(32)),
                              "training_mode": False, "dropout": "disabled"},
        "cache_manifest": manifest,
        "training_labels": {"shape": list(labels.shape), "dtype": str(labels.dtype),
                            "one_hot_valid": True, "class_counts": counts.tolist()},
        "runs": {},
    }
    for experiment, directory in runs.items():
        valid = validate_completed_run(directory)
        model = tf.keras.models.load_model(directory / "model.keras")
        report = architecture_report(model)
        convs = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        probe = tf.keras.Model(model.input, [layer.output for layer in convs])
        activations = probe(np.asarray(images[:32]), training=False)
        inspected = []
        for layer, activation in zip(convs, activations):
            kernel, bias = layer.get_weights()
            values = activation.numpy()
            inspected.append({
                "name": layer.name, "kernel_shape": list(kernel.shape),
                "activation": layer.activation.__name__,
                "kernel_l2_coefficient": layer.kernel_regularizer.l2,
                "kernel_norm_float64": float(np.linalg.norm(kernel.astype(np.float64))),
                "kernel_max_abs": float(np.max(np.abs(kernel))),
                "bias_min": float(bias.min()), "bias_max": float(bias.max()),
                "weights_finite": bool(np.isfinite(kernel).all() and np.isfinite(bias).all()),
                "sample_activation_finite": bool(np.isfinite(values).all()),
                "sample_activation_zero_fraction": float(np.mean(values == 0)),
                "sample_activation_min": float(values.min()),
                "sample_activation_max": float(values.max()),
                "sample_max_range_across_images": float(np.max(np.ptp(values, axis=0))),
            })
        final = convs[-1]
        preactivation = tf.nn.conv2d(activations[-2], final.kernel, strides=[1, 1, 1, 1],
                                      padding="SAME") + final.bias
        probabilities = tf.nn.softmax(tf.reduce_mean(activations[-1], axis=(1, 2))).numpy()
        optimizer = model.optimizer
        output["runs"][experiment] = {
            "metadata": valid["metadata"], "status": valid["status"],
            "config_equals_locked_bytes": (directory / "config.yaml").read_bytes() ==
                                          (root / f"configs/{experiment}.yaml").read_bytes(),
            "preprocessing_identity_matches_cache": preprocessing_identity(
                load_config(directory / "config.yaml")) == manifest["identity"],
            "history": summarize_history(directory / "history.csv"),
            "source_file_sha256": {str(path.relative_to(directory)): digest(path)
                                   for path in sorted(directory.rglob("*")) if path.is_file()},
            "architecture": {"convolutions": report.convolutional_layers,
                             "parameters": report.trainable_parameters,
                             "macs_per_image": report.macs_per_image,
                             "output_shape": list(report.output_shape)},
            "saved_compile_config": model.get_compile_config(),
            "optimizer_iterations": int(optimizer.iterations.numpy()),
            "optimizer_variables_finite": all(bool(np.isfinite(v.numpy()).all())
                                              for v in optimizer.variables),
            "layers": inspected,
            "sample_final_preactivation_min": float(tf.reduce_min(preactivation).numpy()),
            "sample_final_preactivation_max": float(tf.reduce_max(preactivation).numpy()),
            "sample_probability_min": float(probabilities.min()),
            "sample_probability_max": float(probabilities.max()),
            "sample_max_probability_distance_from_uniform": float(np.max(np.abs(probabilities - 0.01))),
            "evaluation_artifacts_exist": (directory / "evaluation").exists(),
        }
    # Write directly: the Metal plugin can emit shutdown messages on stdout.
    with output_path.open("x") as handle:
        handle.write(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON file; an existing file is never overwritten.")
    main(parser.parse_args().output)
