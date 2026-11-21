"""Locks for values and lifecycle rules that later milestones must keep."""

from cifar100_study import contracts


def test_registered_schedules_and_depths():
    assert tuple(contracts.EXPERIMENT_SPECS) == ("e0", "e1", "e2")
    assert contracts.EXPERIMENT_SPECS["e0"] == contracts.ExperimentSpec(
        350, (200, 250, 300), (0, 0, 0)
    )
    assert contracts.EXPERIMENT_SPECS["e1"] == contracts.ExperimentSpec(
        700, (400, 500, 600), (0, 0, 0)
    )
    assert contracts.EXPERIMENT_SPECS["e2"] == contracts.ExperimentSpec(
        350, (200, 250, 300), (3, 3, 3)
    )


def test_decay_boundary_is_the_zero_indexed_milestone():
    assert contracts.LR_MILESTONE_CONVENTION == "zero_indexed_epoch"
    assert contracts.LR_DECAY_FACTOR == 0.1
    e0_first = contracts.EXPERIMENT_SPECS["e0"].lr_milestones[0]
    e1_first = contracts.EXPERIMENT_SPECS["e1"].lr_milestones[0]
    assert (e0_first, e0_first - 1) == (200, 199)
    assert (e1_first, e1_first - 1) == (400, 399)


def test_shared_contract_values():
    assert contracts.STUDY_ID == "cifar100-allcnnc"
    assert contracts.SEED == 42
    assert contracts.INITIAL_LEARNING_RATE == 0.05
    assert contracts.BATCH_SIZE == 256
    assert contracts.MOMENTUM == 0.9
    assert contracts.L2_COEFFICIENT == 0.001
    assert contracts.KERAS_KERNEL_L2 == 0.0005
    assert contracts.KERAS_KERNEL_L2 * 2 == contracts.L2_COEFFICIENT
    assert contracts.AUGMENTATION == "none"
    assert contracts.VALIDATION_SPLIT == 0
    assert contracts.TRAIN_SIZE == 50_000
    assert contracts.TEST_SIZE == 10_000
    assert contracts.NUM_CLASSES == 100
    assert contracts.IMAGE_SHAPE == (32, 32, 3)
    assert contracts.LOSS == "categorical_crossentropy"
    assert contracts.LABEL_ENCODING == "one_hot"
    assert contracts.ONE_HOT_CLASSES == 100
    assert contracts.DATA_TRAINING_CONTRACTS == (
        "encode CIFAR-100 fine labels as one-hot vectors with 100 classes",
        "train with categorical_crossentropy",
        "do not use sparse_categorical_crossentropy",
    )
    assert contracts.PRETRAINED_WEIGHTS is False
    assert contracts.FIT_ON == "train"
    assert contracts.GCN is True
    assert contracts.ZCA_WHITENING is True
    assert contracts.DTYPE == "float32"
    assert contracts.GCN_MIN_DIVISOR == 1e-8
    assert contracts.ZCA_FORM == "full"
    assert contracts.ZCA_REGULARIZATION == 0.1
    assert contracts.COVARIANCE_DIVISOR == "N"
    assert contracts.PREPROCESSING_VERSION == "gcn-zca-v1"
    assert contracts.INPUT_DROPOUT == 0.2
    assert contracts.DROPOUT_AFTER_DOWNSAMPLER == 0.5
    assert contracts.EARLY_STOPPING is False
    assert contracts.CHECKPOINT_SELECTION == "final_scheduled_epoch"
    assert contracts.BASE_CONVOLUTION_COUNT == 9


def test_e2_insertions_follow_the_three_stage_positions():
    insertions = contracts.E2_STAGE_INSERTIONS
    assert insertions == (
        contracts.StageInsertion(3, 3, 1, "same", 96, "before_first_stride2_downsampler"),
        contracts.StageInsertion(3, 3, 1, "same", 192, "before_second_stride2_downsampler"),
        contracts.StageInsertion(3, 3, 1, "same", 192, "after_valid_3x3_before_1x1_head"),
    )
    assert sum(item.count for item in insertions) == 9
    assert contracts.BASE_CONVOLUTION_COUNT + sum(item.count for item in insertions) == 18


def test_lifecycle_contracts_block_legacy_harness_pitfalls():
    assert contracts.VALIDATION_DRIVEN_CHECKPOINTS is False
    assert contracts.RUN_DELETION_ALLOWED is False
    assert contracts.RESUMABLE is True
    assert contracts.HISTORY_RECONCILIATION is True
    assert contracts.FINAL_TEST_GUARD == "confirm-final-test"
    assert contracts.PERIODIC_TEST_METRICS is False
    assert contracts.HISTORY_COLUMNS == (
        "epoch",
        "loss",
        "accuracy",
        "learning_rate",
        "epoch_seconds",
    )
    assert contracts.RUN_STATUSES == ("running", "completed", "failed", "interrupted")

    pitfalls = "\n".join(contracts.LEGACY_HARNESS_PITFALLS)
    for token in (
        "validation_data",
        "save_best_only",
        "EarlyStopping",
        "val_accuracy",
        "recursive deletion",
        "--fresh",
        "shared backup",
        "final test artifact",
        "history.csv",
        "E1 must not resume an E0 run",
    ):
        assert token in pitfalls

    required = "\n".join(contracts.REQUIRED_RUN_BEHAVIORS)
    for token in (
        "--resume",
        "data fingerprint",
        "optimizer state",
        "history.csv",
        "last scheduled epoch",
        "--confirm-final-test",
        "auditable override",
    ):
        assert token in required

    optimizer = "\n".join(contracts.OPTIMIZER_CONSTRAINTS)
    assert "Nesterov" in optimizer
    assert "per-step optimizer decay" in optimizer
    assert "0.001" in optimizer
    dropout = "\n".join(contracts.DROPOUT_PLACEMENT)
    assert "input dropout 0.2" in dropout
    assert "original stride-2 downsamplers" in dropout
