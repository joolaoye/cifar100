"""Synthetic checks for GCN and full ZCA. These tests never load CIFAR-100."""

import numpy as np
import pytest

from cifar100_study.preprocess import (
    PreprocessError,
    apply_zca,
    fit_zca,
    fit_zca_images,
    global_contrast_normalize,
    transform_images,
)


def test_gcn_matches_an_independent_four_feature_calculation():
    image = np.array([1.0, 3.0, 5.0, 7.0]).reshape(1, 2, 2, 1)
    centered = np.array([-3.0, -1.0, 1.0, 3.0])
    expected = (centered / np.sqrt(5.0)).reshape(1, 2, 2, 1)
    output = global_contrast_normalize(image, 1e-8)
    np.testing.assert_allclose(output, expected)
    assert output.dtype == np.float64
    assert np.allclose(output.reshape(-1).mean(), 0.0)
    rms = np.sqrt(np.mean(output.reshape(-1) ** 2))
    np.testing.assert_allclose(rms, 1.0)


def test_gcn_is_per_image_and_keeps_sample_and_channel_order():
    first = np.array([[[[1.0, 9.0], [2.0, 8.0]]]], dtype=np.float64)
    second = np.array([[[[4.0, 4.0], [0.0, 12.0]]]], dtype=np.float64)
    batch = np.concatenate([first, second], axis=0)
    output = global_contrast_normalize(batch, 1e-8)
    np.testing.assert_allclose(output[0], global_contrast_normalize(first, 1e-8)[0])
    np.testing.assert_allclose(output[1], global_contrast_normalize(second, 1e-8)[0])

    pixels = np.array([[[[10, 20, 30]]]], dtype=np.float64) / 255.0
    vector = pixels.reshape(-1)
    centered = vector - vector.mean()
    expected = centered / np.sqrt(np.mean(centered**2))
    whitened = global_contrast_normalize(pixels, 1e-8).reshape(-1)
    np.testing.assert_allclose(whitened, expected)
    assert whitened.shape == (3,)


def test_gcn_constant_and_low_variance_images_stay_finite():
    constant = np.full((2, 2, 2, 3), 9.0)
    output = global_contrast_normalize(constant, 1e-8)
    assert output.shape == constant.shape
    assert np.all(output == 0.0)
    assert np.isfinite(output).all()

    flat = np.array([1.0, 1.0, 1.0, 1.0 + 1e-12])
    centered = flat - flat.mean()
    rms = np.sqrt(np.mean(centered**2))
    assert rms < 1e-8
    low = flat.reshape(1, 2, 2, 1)
    output = global_contrast_normalize(low, 1e-8)
    np.testing.assert_allclose(output.reshape(-1), centered / 1e-8)
    assert np.isfinite(output).all()


def test_gcn_rejects_bad_shapes_and_divisors():
    with pytest.raises(PreprocessError, match="4 dimensions"):
        global_contrast_normalize(np.ones((2, 2)), 1e-8)
    with pytest.raises(PreprocessError, match="empty"):
        global_contrast_normalize(np.ones((0, 2, 2, 1)), 1e-8)
    with pytest.raises(PreprocessError, match="positive float"):
        global_contrast_normalize(np.ones((1, 1, 1, 1)), 0.0)


def test_zca_matches_a_hand_calculated_covariance():
    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    mean, whitening = fit_zca(features, 0.1)
    np.testing.assert_allclose(mean, [0.5, 0.5])
    scale = 1.0 / np.sqrt(0.25 + 0.1)
    np.testing.assert_allclose(whitening, np.eye(2) * scale)
    unbiased_scale = 1.0 / np.sqrt((1.0 / 3.0) + 0.1)
    assert not np.isclose(whitening[0, 0], unbiased_scale)


def test_zca_keeps_a_zero_eigenvalue_component():
    features = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [3.0, 3.0],
            [3.0, 3.0],
        ]
    )
    mean, whitening = fit_zca(features, 0.1)
    np.testing.assert_allclose(mean, [2.0, 2.0])
    zero_axis = np.array([1.0, -1.0]) / np.sqrt(2.0)
    signal_axis = np.array([1.0, 1.0]) / np.sqrt(2.0)
    expected = (1.0 / np.sqrt(0.1)) * np.outer(zero_axis, zero_axis)
    expected += (1.0 / np.sqrt(2.1)) * np.outer(signal_axis, signal_axis)
    np.testing.assert_allclose(whitening, expected)
    reduced = (1.0 / np.sqrt(2.1)) * np.outer(signal_axis, signal_axis)
    assert not np.allclose(whitening, reduced)
    assert np.isfinite(whitening).all()
    assert np.allclose(whitening, whitening.T)


def test_apply_zca_uses_training_mean_and_right_multiplication():
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean = np.array([0.5, 0.25])
    matrix = np.array([[0.0, 1.0], [2.0, 0.0]])
    expected = (features - mean) @ matrix
    np.testing.assert_allclose(apply_zca(features, mean, matrix), expected)


def test_image_zca_matches_explicit_scale_gcn_and_fit():
    images = np.array(
        [
            [[[0, 10, 20], [30, 40, 50]], [[60, 70, 80], [90, 100, 110]]],
            [[[5, 15, 25], [35, 45, 55]], [[65, 75, 85], [95, 105, 250]]],
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        ],
        dtype=np.uint8,
    )
    mean, matrix = fit_zca_images(images, 1e-8, 0.1, chunk_size=2)
    scaled = images.astype(np.float64) / 255.0
    normalized = global_contrast_normalize(scaled, 1e-8)
    reference_mean, reference_matrix = fit_zca(normalized.reshape(3, -1), 0.1)
    np.testing.assert_allclose(mean, reference_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(matrix, reference_matrix, rtol=1e-12, atol=1e-12)

    transformed = transform_images(images, mean, matrix, 1e-8, chunk_size=1)
    manual = apply_zca(normalized.reshape(3, -1), mean, matrix)
    np.testing.assert_allclose(transformed, manual.reshape(images.shape).astype(np.float32))
    assert transformed.dtype == np.float32
    assert transformed.shape == images.shape
    assert np.array_equal(transformed[0], transform_images(images[:1], mean, matrix, 1e-8)[0])


def test_changing_only_test_images_does_not_change_the_training_fit():
    train = np.array(
        [
            [[[0, 255, 3], [9, 1, 80]], [[20, 4, 7], [100, 2, 6]]],
            [[[5, 1, 40], [90, 2, 11]], [[8, 70, 15], [3, 200, 9]]],
            [[[30, 0, 12], [40, 18, 250]], [[7, 60, 1], [22, 5, 99]]],
            [[[8, 80, 14], [15, 3, 44]], [[255, 10, 2], [6, 90, 17]]],
        ],
        dtype=np.uint8,
    )
    test = np.array(
        [
            [[[255, 0, 255], [0, 255, 0]], [[10, 20, 30], [40, 50, 60]]],
            [[[7, 9, 11], [13, 17, 19]], [[1, 2, 4], [8, 16, 32]]],
        ],
        dtype=np.uint8,
    )
    normalized = global_contrast_normalize(train.astype(np.float64) / 255.0, 1e-8)
    assert not np.allclose(normalized[0], normalized[1])
    mean, matrix = fit_zca_images(train, 1e-8, 0.1, chunk_size=1)
    altered = test.copy()
    altered += np.uint8(50)
    same_mean, same_matrix = fit_zca_images(train, 1e-8, 0.1, chunk_size=3)
    np.testing.assert_allclose(mean, same_mean)
    np.testing.assert_allclose(matrix, same_matrix)

    contaminated_mean, contaminated_matrix = fit_zca_images(
        np.concatenate([train, altered]),
        1e-8,
        0.1,
        chunk_size=2,
    )
    assert not np.allclose(mean, contaminated_mean)
    assert not np.allclose(matrix, contaminated_matrix)

    train_output = transform_images(train, mean, matrix, 1e-8, chunk_size=2)
    test_output = transform_images(test, mean, matrix, 1e-8, chunk_size=2)
    test_mean, test_matrix = fit_zca_images(test, 1e-8, 0.1, chunk_size=2)
    refit_test = transform_images(test, test_mean, test_matrix, 1e-8, chunk_size=2)
    assert train_output.shape == train.shape
    assert test_output.shape == test.shape
    assert not np.allclose(test_output, refit_test)


def test_chunked_fit_agrees_across_chunk_sizes():
    images = np.arange(6 * 2 * 2 * 1, dtype=np.uint8).reshape(6, 2, 2, 1)
    mean_one, matrix_one = fit_zca_images(images, 1e-8, 0.1, chunk_size=1)
    mean_all, matrix_all = fit_zca_images(images, 1e-8, 0.1, chunk_size=6)
    np.testing.assert_allclose(mean_one, mean_all)
    np.testing.assert_allclose(matrix_one, matrix_all)


def test_zca_rejects_non_uint8_images_and_bad_regularization():
    with pytest.raises(PreprocessError, match="uint8"):
        fit_zca_images(np.ones((2, 2, 2, 1), dtype=np.float32), 1e-8, 0.1)
    with pytest.raises(PreprocessError, match="positive float"):
        fit_zca(np.ones((2, 2)), 0.0)
    with pytest.raises(PreprocessError, match="feature dimension"):
        apply_zca(np.ones((2, 3)), np.zeros(2), np.eye(2))
