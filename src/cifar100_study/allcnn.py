"""All-CNN-C architectures for E0, E1, and E2.

E0 and E1 use the nine-convolution baseline. E2 inserts three stride-1
convolutions in each of the three stages and has eighteen convolutions.
Importing this module does not import Keras. Building a model does.

Classification head
-------------------
Every convolution uses ReLU, including the final 1x1 class convolution.
Table 1 of Springenberg et al., *Striving for Simplicity: The All
Convolutional Net*, specifies ReLU for that class-producing convolution.
Global average pooling then reduces the 6x6x100 map to 100 scores, and
softmax turns those scores into class probabilities. The model output shape
is ``(None, 100)``.

Biases are enabled and initialized to zero. The parameter reference for this
study counts those biases.

L2
--
The scientific coefficient is the coupled weight decay 0.001. Keras implements
an L2 penalty as ``coefficient * sum(square(W))``, so its gradient is
``2 * coefficient * W``. Every convolution kernel, including the final
100-class 1x1, uses ``L2(0.0005)``. That is the unique Keras coefficient whose
penalty gradient equals ``0.001 * W``. Biases and activations are not
regularized. The builder does not compile the model and does not set an
optimizer weight-decay term.

Initialization
--------------
``build_allcnn`` seeds Python, NumPy, and Keras once from the experiment
seed, then creates an unseeded He-normal initializer for each convolution.
A per-layer copy of the same fixed seed would give same-shaped layers
identical initial kernels. M0.3 can call this builder after its own
process-level seeding; this function resets the seed from the config so
construction is reproducible on its own.

MACs
----
A convolution MAC total counts one multiply-accumulate for each output
spatial position, output channel, kernel spatial position, and input
channel. Bias addition, ReLU, dropout, global average pooling, and softmax
are excluded. The total is not a FLOP count: treating each multiply and each
add as a separate FLOP would be about twice the MAC total, and this module
does not report that conversion.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

from cifar100_study import contracts
from cifar100_study.config import ExperimentConfig

# See the module note. This equals ``contracts.L2_COEFFICIENT / 2``.
KERAS_KERNEL_L2 = contracts.KERAS_KERNEL_L2
# (convolutional layers, trainable parameters, convolution MACs per image).
# These are the measurements of the registered E0/E1 and E2 graphs.
REGISTERED_COMPLEXITY = MappingProxyType(
    {
        "e0": (9, 1_387_108, 271_420_416),
        "e1": (9, 1_387_108, 271_420_416),
        "e2": (18, 3_628_036, 816_860_160),
    }
)
if KERAS_KERNEL_L2 * 2 != contracts.L2_COEFFICIENT:
    raise RuntimeError("Keras kernel L2 must be half the scientific weight decay")


class ModelError(ValueError):
    """The architecture request is outside the locked All-CNN-C study."""


@dataclass(frozen=True, slots=True)
class ConvInspection:
    """One convolution read back from a built model."""

    name: str
    kernel: tuple[int, int]
    filters: int
    strides: tuple[int, int]
    padding: str
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    macs: int


@dataclass(frozen=True, slots=True)
class ArchitectureReport:
    """Analytical complexity of one built network."""

    convolutional_layers: int
    trainable_parameters: int
    macs_per_image: int
    output_shape: tuple[int | None, ...]
    convolutions: tuple[ConvInspection, ...]


def build_allcnn(config: ExperimentConfig):
    """Build the All-CNN-C variant described by a validated experiment config."""
    extras = _require_registered_layout(config)
    if config.model.architecture != contracts.ARCHITECTURE:
        raise ModelError(
            f"architecture must be {contracts.ARCHITECTURE!r}, got {config.model.architecture!r}"
        )
    if config.model.initializer != contracts.INITIALIZER:
        raise ModelError(
            f"initializer must be {contracts.INITIALIZER!r}, got {config.model.initializer!r}"
        )
    image_shape = config.dataset.image_shape
    if image_shape != contracts.IMAGE_SHAPE:
        raise ModelError(f"All-CNN-C input shape must be {contracts.IMAGE_SHAPE}, got {image_shape}")
    if config.dataset.num_classes != contracts.NUM_CLASSES:
        raise ModelError(
            f"All-CNN-C must end in {contracts.NUM_CLASSES} classes, got {config.dataset.num_classes}"
        )

    from keras import Input, Model, initializers, layers, regularizers, utils

    utils.set_random_seed(config.training.seed)
    plan = _convolution_plan(extras, config.dataset.num_classes)
    features = Input(shape=image_shape, name="images")
    hidden = layers.Dropout(config.model.input_dropout, name="dropout_input")(features)
    downsamplers_seen = 0
    for index, spec in enumerate(plan, start=1):
        hidden = layers.Conv2D(
            filters=spec.filters,
            kernel_size=spec.kernel,
            strides=spec.stride,
            padding=spec.padding,
            activation="relu",
            use_bias=True,
            data_format="channels_last",
            dilation_rate=1,
            groups=1,
            kernel_initializer=initializers.HeNormal(),
            bias_initializer="zeros",
            kernel_regularizer=regularizers.L2(KERAS_KERNEL_L2),
            name=f"conv_{index:02d}",
        )(hidden)
        if spec.dropout_after:
            downsamplers_seen += 1
            hidden = layers.Dropout(
                config.model.dropout_after_downsampler,
                name=f"dropout_downsample_{downsamplers_seen}",
            )(hidden)
    pooled = layers.GlobalAveragePooling2D(name="global_average_pool")(hidden)
    probabilities = layers.Softmax(name="class_probabilities")(pooled)
    return Model(inputs=features, outputs=probabilities, name=f"allcnn_{config.experiment}")


def architecture_report(model) -> ArchitectureReport:
    """Inspect convolution shapes and analytical MACs from a built model."""
    convolutions = tuple(_inspect_convolution(layer) for layer in _convolution_layers(model))
    output_shape = tuple(None if dim is None else int(dim) for dim in model.output.shape)
    return ArchitectureReport(
        convolutional_layers=len(convolutions),
        trainable_parameters=sum(_weight_size(weight) for weight in model.trainable_weights),
        macs_per_image=sum(item.macs for item in convolutions),
        output_shape=output_shape,
        convolutions=convolutions,
    )


def format_architecture_report(report: ArchitectureReport) -> str:
    """Return a plain-text summary. MACs are multiply-accumulates, not FLOPs."""
    lines = [
        f"convolutional_layers: {report.convolutional_layers}",
        f"trainable_parameters: {report.trainable_parameters}",
        f"convolution_macs_per_image: {report.macs_per_image}",
        f"output_shape: {report.output_shape}",
        (
            "mac_convention: H_out * W_out * K_h * K_w * C_in * C_out; "
            "bias, ReLU, dropout, global pooling, and softmax excluded"
        ),
    ]
    for conv in report.convolutions:
        lines.append(
            f"{conv.name}: kernel={conv.kernel[0]}x{conv.kernel[1]} "
            f"filters={conv.filters} stride={conv.strides[0]} padding={conv.padding} "
            f"in={conv.input_shape} out={conv.output_shape} macs={conv.macs}"
        )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class _ConvPlan:
    kernel: int
    filters: int
    stride: int
    padding: str
    dropout_after: bool


def _require_registered_layout(config: ExperimentConfig) -> tuple[int, int, int]:
    """Accept only the layout registered for this experiment.

    The study has two layouts: ``(0, 0, 0)`` for E0 and E1, and ``(3, 3, 3)``
    for E2. A direct call cannot substitute another nonnegative triple, and
    an experiment id cannot be paired with the other experiment's layout.
    """
    extras = tuple(config.model.stage_extra_convolutions)
    registered = {
        spec.stage_extra_convolutions for spec in contracts.EXPERIMENT_SPECS.values()
    }
    if extras not in registered:
        allowed = ", ".join(str(layout) for layout in sorted(registered))
        raise ModelError(
            f"stage_extra_convolutions {extras} is not a registered All-CNN-C layout; "
            f"expected one of {allowed}"
        )
    spec = contracts.EXPERIMENT_SPECS.get(config.experiment)
    if spec is None:
        raise ModelError(f"unknown experiment {config.experiment!r}")
    if extras != spec.stage_extra_convolutions:
        raise ModelError(
            f"{config.experiment} requires stage_extra_convolutions "
            f"{spec.stage_extra_convolutions}, got {extras}"
        )
    return extras


def _convolution_plan(extras: Sequence[int], num_classes: int) -> tuple[_ConvPlan, ...]:
    """Baseline All-CNN-C with extras inserted at the three approved points."""
    before_first, before_second, after_valid = (int(item) for item in extras)
    stem = [_ConvPlan(3, 96, 1, "same", False) for _ in range(2)]
    stem.extend(_ConvPlan(3, 96, 1, "same", False) for _ in range(before_first))
    first_downsample = _ConvPlan(3, 96, 2, "same", True)
    middle = [_ConvPlan(3, 192, 1, "same", False) for _ in range(2)]
    middle.extend(_ConvPlan(3, 192, 1, "same", False) for _ in range(before_second))
    second_downsample = _ConvPlan(3, 192, 2, "same", True)
    tail = [_ConvPlan(3, 192, 1, "valid", False)]
    tail.extend(_ConvPlan(3, 192, 1, "same", False) for _ in range(after_valid))
    head = (
        _ConvPlan(1, 192, 1, "same", False),
        _ConvPlan(1, num_classes, 1, "same", False),
    )
    return tuple([*stem, first_downsample, *middle, second_downsample, *tail, *head])


def _convolution_layers(model) -> list:
    from keras.layers import Conv2D

    return [layer for layer in model.layers if isinstance(layer, Conv2D)]


def _inspect_convolution(layer) -> ConvInspection:
    kernel_shape = tuple(int(dim) for dim in layer.kernel.shape)
    if len(kernel_shape) != 4:
        raise ModelError(f"{layer.name} kernel shape {kernel_shape} is not a 2D convolution")
    height, width, input_channels, filters = kernel_shape
    if getattr(layer, "groups", 1) != 1 or tuple(int(dim) for dim in layer.dilation_rate) != (1, 1):
        raise ModelError(f"{layer.name} is not a standard full convolution")
    input_shape = _spatial_shape(layer.input.shape)
    output_shape = _spatial_shape(layer.output.shape)
    if output_shape[-1] != filters or input_shape[-1] != input_channels:
        raise ModelError(f"{layer.name} channel counts do not match its kernel")
    macs = output_shape[0] * output_shape[1] * height * width * input_channels * filters
    return ConvInspection(
        name=layer.name,
        kernel=(height, width),
        filters=filters,
        strides=tuple(int(dim) for dim in layer.strides),
        padding=str(layer.padding),
        input_shape=input_shape,
        output_shape=output_shape,
        macs=macs,
    )


def _weight_size(weight) -> int:
    size = 1
    for dim in tuple(weight.shape):
        size *= int(dim)
    return size


def _spatial_shape(shape) -> tuple[int, int, int]:
    spatial = tuple(shape)[1:]
    if len(spatial) != 3 or any(dim is None for dim in spatial):
        raise ModelError(f"convolution feature shape must be (None, H, W, C), got {tuple(shape)}")
    return tuple(int(dim) for dim in spatial)
