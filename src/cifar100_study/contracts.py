"""Locked study values and acceptance contracts.

Configuration loading rejects a file that disagrees with these values.
Dataset preparation uses the GCN and ZCA constants in this module. Training
and evaluation code must keep the remaining values.

Learning-rate milestones are zero-indexed epoch numbers. For milestones
``(200, 250, 300)``, epoch index 199 still uses the previous rate and epoch
index 200 uses the decayed rate. The E1 boundary is index 399 / 400.

The approved loss is categorical cross-entropy. Dataset preparation one-hot
encodes the 100 CIFAR-100 fine labels. GCN uses a minimum divisor of 1e-8.
Full ZCA uses regularization 0.1 and divides the training covariance by N.
"""

from dataclasses import dataclass
from types import MappingProxyType

STUDY_ID = "cifar100-allcnnc"

DATASET_NAME = "cifar100"
LABEL_MODE = "fine"
NUM_CLASSES = 100
IMAGE_SHAPE = (32, 32, 3)
TRAIN_SIZE = 50_000
TEST_SIZE = 10_000
VALIDATION_SPLIT = 0
AUGMENTATION = "none"

RESCALE_FROM = (0, 255)
RESCALE_TO = (0, 1)
GCN = True
GCN_MIN_DIVISOR = 1e-8
ZCA_WHITENING = True
ZCA_FORM = "full"
ZCA_REGULARIZATION = 0.1
COVARIANCE_DIVISOR = "N"
FIT_ON = "train"
DTYPE = "float32"
# Increment this when a change alters the numerical output or the stored
# representation of preprocessing or label encoding. That includes GCN, ZCA
# fitting or transformation, numerical conventions, one-hot encoding, and the
# dtype or layout of cached artifacts. Review cache compatibility in the same
# change. A validation-only fix that preserves those outputs keeps the version.
PREPROCESSING_VERSION = "gcn-zca-v1"

ARCHITECTURE = "all-cnn-c"
INPUT_DROPOUT = 0.2
DROPOUT_AFTER_DOWNSAMPLER = 0.5
INITIALIZER = "he_normal"
BASE_CONVOLUTION_COUNT = 9

OPTIMIZER = "sgd"
MOMENTUM = 0.9
INITIAL_LEARNING_RATE = 0.05
LR_DECAY_FACTOR = 0.1
LR_MILESTONE_CONVENTION = "zero_indexed_epoch"
BATCH_SIZE = 256
L2_COEFFICIENT = 0.001
# Keras L2 adds ``coefficient * sum(square(W))``. The gradient of that penalty
# is ``2 * coefficient * W``. Half of the scientific coefficient therefore
# produces the coupled weight-decay gradient ``0.001 * W``. Optimizer
# ``weight_decay`` stays unset; this is not decoupled AdamW-style decay.
KERAS_KERNEL_L2 = 0.0005
LOSS = "categorical_crossentropy"
LABEL_ENCODING = "one_hot"
ONE_HOT_CLASSES = 100

# Dataset preparation one-hot encodes labels. Training must use categorical cross-entropy.
DATA_TRAINING_CONTRACTS = (
    "encode CIFAR-100 fine labels as one-hot vectors with 100 classes",
    "train with categorical_crossentropy",
    "do not use sparse_categorical_crossentropy",
)

SEED = 42
PRETRAINED_WEIGHTS = False
EARLY_STOPPING = False
CHECKPOINT_SELECTION = "final_scheduled_epoch"

VALIDATION_DRIVEN_CHECKPOINTS = False
RUN_DELETION_ALLOWED = False
RESUMABLE = True
HISTORY_RECONCILIATION = True
FINAL_TEST_GUARD = "confirm-final-test"
PERIODIC_TEST_METRICS = False

HISTORY_COLUMNS = ("epoch", "loss", "accuracy", "learning_rate", "epoch_seconds")
RUN_STATUSES = ("running", "completed", "failed", "interrupted")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Pre-registered independent variables for one experiment."""

    epochs: int
    lr_milestones: tuple[int, int, int]
    stage_extra_convolutions: tuple[int, int, int]


EXPERIMENT_SPECS = MappingProxyType(
    {
        "e0": ExperimentSpec(350, (200, 250, 300), (0, 0, 0)),
        "e1": ExperimentSpec(700, (400, 500, 600), (0, 0, 0)),
        "e2": ExperimentSpec(350, (200, 250, 300), (3, 3, 3)),
    }
)


@dataclass(frozen=True, slots=True)
class StageInsertion:
    """Extra convolutions inserted into one All-CNN-C stage."""

    count: int
    kernel: int
    stride: int
    padding: str
    channels: int
    position: str


# E2 inserts these before original layer 3, before original layer 6, and
# immediately after original layer 7 (the valid-padded 3x3), before the 1x1 head.
E2_STAGE_INSERTIONS = (
    StageInsertion(3, 3, 1, "same", 96, "before_first_stride2_downsampler"),
    StageInsertion(3, 3, 1, "same", 192, "before_second_stride2_downsampler"),
    StageInsertion(3, 3, 1, "same", 192, "after_valid_3x3_before_1x1_head"),
)

# Pitfalls from the GTSRB reference harness. Later milestones must not implement them.
LEGACY_HARNESS_PITFALLS = (
    "no validation_data during training",
    "no ModelCheckpoint save_best_only",
    "no EarlyStopping",
    "no val_accuracy checkpoint selection",
    "no recursive deletion of a run directory",
    "no --fresh shortcut",
    "no shared backup directory across experiments",
    "no use of a resume backup as the final test artifact",
    "history.csv must be reconciled if a row is ahead of the restored epoch",
    "E1 must not resume an E0 run",
)

REQUIRED_RUN_BEHAVIORS = (
    "a new run uses a new id",
    "only --resume RUN_ID reopens an existing run",
    "resume checks the config, data fingerprint, and backend",
    "resume restores weights, optimizer state, completed epoch, and learning-rate schedule",
    "history.csv is reconciled to the last durably recovered epoch before appending",
    "the final .keras model is written only after the last scheduled epoch",
    "evaluation requires --confirm-final-test",
    "repeat evaluation is refused unless an auditable override is recorded",
)

# Dropout stays on the original All-CNN-C positions when E2 adds convolutions.
DROPOUT_PLACEMENT = (
    "input dropout 0.2 precedes the first convolution",
    "dropout 0.5 follows only the original stride-2 downsamplers",
)

# Nesterov and per-step optimizer decay are outside the study. The scientific
# L2 coefficient 0.001 is the effective coupled weight decay. Every experiment
# implements it as Keras kernel L2(0.0005) and leaves optimizer weight decay unset.
OPTIMIZER_CONSTRAINTS = (
    "SGD with momentum 0.9",
    "Nesterov momentum is not part of the study",
    "no per-step optimizer decay in addition to the milestone schedule",
    "L2 coefficient 0.001 is the effective coupled weight decay; the Keras kernel L2 coefficient is 0.0005",
)
