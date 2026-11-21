"""Structural tests for the nine- and eighteen-convolution All-CNN-C models."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    InputLayer,
    MaxPooling2D,
    Softmax,
)
from keras.regularizers import L2

from cifar100_study.allcnn import (
    KERAS_KERNEL_L2,
    ModelError,
    architecture_report,
    build_allcnn,
    format_architecture_report,
)
from cifar100_study.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

BASELINE_SPECS = (
    ((3, 3), 96, (1, 1), "same", (32, 32, 3), (32, 32, 96)),
    ((3, 3), 96, (1, 1), "same", (32, 32, 96), (32, 32, 96)),
    ((3, 3), 96, (2, 2), "same", (32, 32, 96), (16, 16, 96)),
    ((3, 3), 192, (1, 1), "same", (16, 16, 96), (16, 16, 192)),
    ((3, 3), 192, (1, 1), "same", (16, 16, 192), (16, 16, 192)),
    ((3, 3), 192, (2, 2), "same", (16, 16, 192), (8, 8, 192)),
    ((3, 3), 192, (1, 1), "valid", (8, 8, 192), (6, 6, 192)),
    ((1, 1), 192, (1, 1), "same", (6, 6, 192), (6, 6, 192)),
    ((1, 1), 100, (1, 1), "same", (6, 6, 192), (6, 6, 100)),
)

EXTENDED_SPECS = (
    ((3, 3), 96, (1, 1), "same", (32, 32, 3), (32, 32, 96)),
    ((3, 3), 96, (1, 1), "same", (32, 32, 96), (32, 32, 96)),
    ((3, 3), 96, (1, 1), "same", (32, 32, 96), (32, 32, 96)),
    ((3, 3), 96, (1, 1), "same", (32, 32, 96), (32, 32, 96)),
    ((3, 3), 96, (1, 1), "same", (32, 32, 96), (32, 32, 96)),
    ((3, 3), 96, (2, 2), "same", (32, 32, 96), (16, 16, 96)),
    ((3, 3), 192, (1, 1), "same", (16, 16, 96), (16, 16, 192)),
    ((3, 3), 192, (1, 1), "same", (16, 16, 192), (16, 16, 192)),
    ((3, 3), 192, (1, 1), "same", (16, 16, 192), (16, 16, 192)),
    ((3, 3), 192, (1, 1), "same", (16, 16, 192), (16, 16, 192)),
    ((3, 3), 192, (1, 1), "same", (16, 16, 192), (16, 16, 192)),
    ((3, 3), 192, (2, 2), "same", (16, 16, 192), (8, 8, 192)),
    ((3, 3), 192, (1, 1), "valid", (8, 8, 192), (6, 6, 192)),
    ((3, 3), 192, (1, 1), "same", (6, 6, 192), (6, 6, 192)),
    ((3, 3), 192, (1, 1), "same", (6, 6, 192), (6, 6, 192)),
    ((3, 3), 192, (1, 1), "same", (6, 6, 192), (6, 6, 192)),
    ((1, 1), 192, (1, 1), "same", (6, 6, 192), (6, 6, 192)),
    ((1, 1), 100, (1, 1), "same", (6, 6, 192), (6, 6, 100)),
)


def _config(name: str):
    return load_config(CONFIG_DIR / name)


def _convolutions(model):
    return [layer for layer in model.layers if isinstance(layer, Conv2D)]


def _observed_specs(model):
    return tuple(_layer_spec(layer) for layer in _convolutions(model))


def _layer_spec(layer):
    return (
        tuple(int(dim) for dim in layer.kernel.shape[:2]),
        int(layer.filters),
        tuple(int(dim) for dim in layer.strides),
        str(layer.padding),
        tuple(int(dim) for dim in tuple(layer.input.shape)[1:]),
        tuple(int(dim) for dim in tuple(layer.output.shape)[1:]),
    )


def _is_subsequence(small, big) -> bool:
    position = 0
    for item in big:
        if position < len(small) and item == small[position]:
            position += 1
    return position == len(small)


def _assert_coupled_kernel_l2(convolutions) -> None:
    assert convolutions
    for layer in convolutions:
        regularizer = layer.kernel_regularizer
        assert isinstance(regularizer, L2)
        assert float(regularizer.l2) == KERAS_KERNEL_L2
        assert layer.bias_regularizer is None
        assert layer.activity_regularizer is None


def _independent_macs(model) -> int:
    total = 0
    for layer in _convolutions(model):
        height, width, incoming, outgoing = (int(dim) for dim in layer.kernel.shape)
        _, out_h, out_w, out_c = (int(dim) if dim is not None else None for dim in tuple(layer.output.shape))
        assert out_c == outgoing
        total += out_h * out_w * height * width * incoming * outgoing
    return total


def test_baseline_has_nine_convolutions_and_the_approved_head():
    model = build_allcnn(_config("e0.yaml"))
    convs = _convolutions(model)
    assert _observed_specs(model) == BASELINE_SPECS
    assert len(convs) == 9
    assert sum(tuple(layer.strides) == (2, 2) for layer in convs) == 2
    assert [str(layer.padding) for layer in convs].count("valid") == 1
    assert all(layer.activation.__name__ == "relu" for layer in convs)
    assert all(layer.use_bias for layer in convs)
    _assert_coupled_kernel_l2(convs)
    drops = [layer for layer in model.layers if isinstance(layer, Dropout)]
    assert [layer.rate for layer in drops] == [0.2, 0.5, 0.5]
    stride_positions = [index for index, layer in enumerate(model.layers) if isinstance(layer, Conv2D) and tuple(layer.strides) == (2, 2)]
    for position in stride_positions:
        assert isinstance(model.layers[position + 1], Dropout)
        assert model.layers[position + 1].rate == 0.5
    assert isinstance(model.layers[0], InputLayer)
    assert isinstance(model.layers[1], Dropout) and model.layers[1].rate == 0.2
    assert isinstance(model.layers[-2], GlobalAveragePooling2D)
    assert isinstance(model.layers[-1], Softmax)
    assert tuple(model.output.shape) == (None, 100)
    forbidden = (Dense, MaxPooling2D, BatchNormalization, Add)
    assert not any(isinstance(layer, forbidden) for layer in model.layers)
    report = architecture_report(model)
    assert report.convolutional_layers == 9
    assert report.trainable_parameters == 1_387_108
    assert report.macs_per_image == 271_420_416
    assert report.macs_per_image == _independent_macs(model)
    assert report.output_shape == (None, 100)
    text = format_architecture_report(report)
    assert "convolution_macs_per_image: 271420416" in text
    assert "FLOP" not in text.upper() or "not" in text
    assert model.compiled is False


def test_extended_model_inserts_nine_convolutions_without_replacing_the_baseline():
    model = build_allcnn(_config("e2.yaml"))
    assert _observed_specs(model) == EXTENDED_SPECS
    baseline = build_allcnn(_config("e0.yaml"))
    assert _is_subsequence(_observed_specs(baseline), _observed_specs(model))
    convs = _convolutions(model)
    assert len(convs) == 18
    assert sum(tuple(layer.strides) == (2, 2) for layer in convs) == 2
    valid = [layer for layer in convs if layer.padding == "valid"]
    assert len(valid) == 1
    assert tuple(int(dim) for dim in valid[0].kernel.shape[:2]) == (3, 3)
    drops = [layer for layer in model.layers if isinstance(layer, Dropout)]
    assert [layer.rate for layer in drops] == [0.2, 0.5, 0.5]
    stride_positions = [index for index, layer in enumerate(model.layers) if isinstance(layer, Conv2D) and tuple(layer.strides) == (2, 2)]
    assert len(stride_positions) == 2
    for position in stride_positions:
        assert isinstance(model.layers[position + 1], Dropout)
    valid_index = next(index for index, layer in enumerate(convs) if layer.padding == "valid")
    following = convs[valid_index + 1 : valid_index + 4]
    assert [(_layer_spec(layer)[0], _layer_spec(layer)[1], _layer_spec(layer)[3]) for layer in following] == [
        ((3, 3), 192, "same"),
        ((3, 3), 192, "same"),
        ((3, 3), 192, "same"),
    ]
    assert convs[valid_index + 4].kernel.shape[:2] == (1, 1)
    _assert_coupled_kernel_l2(convs)
    assert not any(isinstance(layer, (Dense, MaxPooling2D, BatchNormalization, Add)) for layer in model.layers)
    report = architecture_report(model)
    assert report.convolutional_layers == 18
    assert report.trainable_parameters == 3_628_036
    assert report.macs_per_image == 816_860_160
    assert report.macs_per_image == _independent_macs(model)
    assert report.output_shape == (None, 100)


def test_e0_and_e1_match_and_ignore_the_training_schedule():
    e0 = _config("e0.yaml")
    e1 = _config("e1.yaml")
    assert e0.training.epochs != e1.training.epochs
    assert e0.training.lr_milestones != e1.training.lr_milestones
    first = build_allcnn(e0)
    second = build_allcnn(e1)
    longer = build_allcnn(replace(e0, training=replace(e0.training, epochs=700, lr_milestones=(400, 500, 600))))
    assert _observed_specs(first) == _observed_specs(second) == BASELINE_SPECS
    for left, right in zip(first.trainable_weights, second.trainable_weights, strict=True):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    for left, right in zip(first.trainable_weights, longer.trainable_weights, strict=True):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    rates = [layer.rate for layer in first.layers if isinstance(layer, Dropout)]
    assert rates == [layer.rate for layer in second.layers if isinstance(layer, Dropout)]


def test_initialization_is_he_normal_reproducible_and_not_copied_across_layers():
    model = build_allcnn(_config("e0.yaml"))
    again = build_allcnn(_config("e0.yaml"))
    convs = _convolutions(model)
    for layer in convs:
        assert layer.kernel_initializer.__class__.__name__ == "HeNormal"
        assert layer.bias_initializer.__class__.__name__ == "Zeros"
        np.testing.assert_array_equal(layer.bias.numpy(), 0)
    for left, right in zip(model.trainable_weights, again.trainable_weights, strict=True):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    same_shape = [layer for layer in convs if tuple(int(dim) for dim in layer.kernel.shape) == (3, 3, 192, 192)]
    assert len(same_shape) >= 2
    assert not np.allclose(same_shape[0].kernel.numpy(), same_shape[1].kernel.numpy())
    kernel = same_shape[0].kernel.numpy()
    fan_in = int(np.prod(kernel.shape[:3]))
    np.testing.assert_allclose(kernel.std(), np.sqrt(2.0 / fan_in), rtol=0.08, atol=0.0)


def test_forward_and_backward_passes_use_dropout_only_while_training():
    import tensorflow as tf

    images = np.linspace(-1.0, 1.0, 4 * 32 * 32 * 3, dtype=np.float32).reshape(4, 32, 32, 3)
    for name in ("e0.yaml", "e2.yaml"):
        model = build_allcnn(_config(name))
        first = model(images, training=False).numpy()
        second = model(images, training=False).numpy()
        training = model(images, training=True).numpy()
        assert first.shape == (4, 100)
        assert np.isfinite(first).all()
        assert np.all(first >= 0.0)
        np.testing.assert_allclose(first.sum(axis=1), np.ones(4), atol=1e-5)
        np.testing.assert_array_equal(first, second)
        assert not np.allclose(first, training)
        inputs = tf.constant(images)
        with tf.GradientTape() as tape:
            probabilities = model(inputs, training=False)
            objective = tf.reduce_sum(probabilities[:, 0])
        gradients = tape.gradient(objective, model.trainable_weights)
        assert all(gradient is not None for gradient in gradients)
        assert any(float(tf.reduce_sum(tf.abs(gradient))) > 0.0 for gradient in gradients)


@pytest.mark.parametrize("extras", [(1, 0, 0), (3, 0, 3), (2, 2, 2)])
def test_builder_rejects_unregistered_layouts_without_yaml(extras):
    config = _config("e0.yaml")
    forged = replace(config, model=replace(config.model, stage_extra_convolutions=extras))
    with pytest.raises(ModelError, match="registered All-CNN-C layout"):
        build_allcnn(forged)


def test_builder_rejects_a_registered_layout_on_the_wrong_experiment():
    baseline = _config("e0.yaml")
    deep = _config("e2.yaml")
    baseline_with_e2_layout = replace(
        baseline,
        model=replace(baseline.model, stage_extra_convolutions=(3, 3, 3)),
    )
    e2_with_baseline_layout = replace(
        deep,
        model=replace(deep.model, stage_extra_convolutions=(0, 0, 0)),
    )
    with pytest.raises(ModelError, match="e0 requires"):
        build_allcnn(baseline_with_e2_layout)
    with pytest.raises(ModelError, match="e2 requires"):
        build_allcnn(e2_with_baseline_layout)


def test_schedule_replacement_does_not_change_parameter_count():
    baseline = architecture_report(build_allcnn(_config("e0.yaml")))
    extended = architecture_report(build_allcnn(_config("e2.yaml")))
    assert baseline.trainable_parameters == pytest.approx(1_387_108)
    assert extended.macs_per_image == pytest.approx(816_860_160)
    assert extended.trainable_parameters > baseline.trainable_parameters
