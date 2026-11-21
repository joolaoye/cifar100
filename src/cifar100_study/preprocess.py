"""Pixel scaling, per-image GCN, and full ZCA whitening.

Transformations run in this order: divide uint8 pixels by 255, apply GCN to
every feature of each image, then apply a ZCA matrix fit on training images
only. Fitting uses float64. The returned image arrays are float32.

For covariance ``C = U diag(λ) Uᵀ`` the whitening matrix is

``W = U diag(1 / sqrt(max(λ, 0) + regularization)) Uᵀ``.

The covariance divisor is the training count ``N``. The covariance is replaced
by its symmetric part before ``eigh`` so roundoff cannot break symmetry.
Negative eigenvalues are clipped to zero before the locked regularization is
added. The matrix is full rank in feature dimension; eigenvalues are not
dropped.
"""

from collections.abc import Iterator

import numpy as np

CHUNK_SIZE = 512


class PreprocessError(ValueError):
    """A preprocessing input or parameter is unusable."""


def global_contrast_normalize(images: np.ndarray, min_divisor: float) -> np.ndarray:
    """Center each image and divide by its root-mean-square.

    ``images`` has shape ``(N, H, W, C)``. All spatial locations and channels
    of one image are one feature vector. The result is float64 with the same
    shape. A constant image yields zeros because its centered values are zero.
    """
    if images.ndim != 4:
        raise PreprocessError(f"GCN expected 4 dimensions, got shape {images.shape}")
    if images.shape[0] < 1 or images.shape[1] < 1 or images.shape[2] < 1 or images.shape[3] < 1:
        raise PreprocessError(f"GCN received an empty image array with shape {images.shape}")
    if type(min_divisor) is not float or not min_divisor > 0.0:
        raise PreprocessError(f"GCN minimum divisor must be a positive float, got {min_divisor!r}")
    flat = np.reshape(images, (images.shape[0], -1))
    working = np.asarray(flat, dtype=np.float64)
    working -= working.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(working * working, axis=1, keepdims=True))
    np.maximum(rms, min_divisor, out=rms)
    working /= rms
    if not np.isfinite(working).all():
        raise PreprocessError("GCN produced a non-finite value")
    return working.reshape(images.shape)


def whitening_from_covariance(
    mean: np.ndarray,
    covariance: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the full ZCA mean and matrix from a training covariance."""
    if type(regularization) is not float or not regularization > 0.0:
        raise PreprocessError(
            f"ZCA regularization must be a positive float, got {regularization!r}"
        )
    mean_vector = np.asarray(mean, dtype=np.float64)
    matrix = np.asarray(covariance, dtype=np.float64)
    if mean_vector.ndim != 1 or matrix.shape != (mean_vector.shape[0], mean_vector.shape[0]):
        raise PreprocessError(
            "ZCA covariance must be square in the feature dimension of the training mean"
        )
    symmetric = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, 0.0)
    scales = 1.0 / np.sqrt(clipped + regularization)
    whitening = (eigenvectors * scales) @ eigenvectors.T
    if not np.isfinite(whitening).all() or not np.isfinite(mean_vector).all():
        raise PreprocessError("ZCA fitting produced a non-finite parameter")
    return mean_vector, whitening


def fit_zca(features: np.ndarray, regularization: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit full ZCA on ``(N, D)`` training features.

    The covariance is ``centered.T @ centered / N``.
    """
    working = np.asarray(features, dtype=np.float64)
    if working.ndim != 2 or working.shape[0] < 1 or working.shape[1] < 1:
        raise PreprocessError(f"ZCA features must have shape (N, D), got {working.shape}")
    mean = working.mean(axis=0)
    centered = working - mean
    covariance = (centered.T @ centered) / working.shape[0]
    return whitening_from_covariance(mean, covariance, regularization)


def apply_zca(features: np.ndarray, mean: np.ndarray, whitening: np.ndarray) -> np.ndarray:
    """Apply a training-derived ZCA transform to ``(N, D)`` features."""
    working = np.asarray(features, dtype=np.float64)
    if working.ndim != 2:
        raise PreprocessError(f"ZCA transform expected shape (N, D), got {working.shape}")
    if mean.shape != (working.shape[1],) or whitening.shape != (working.shape[1], working.shape[1]):
        raise PreprocessError(
            "ZCA parameters do not match the feature dimension "
            f"{working.shape[1]}: mean {mean.shape}, matrix {whitening.shape}"
        )
    transformed = (working - mean) @ whitening
    if not np.isfinite(transformed).all():
        raise PreprocessError("ZCA transform produced a non-finite value")
    return transformed


def _gcn_feature_batches(
    images: np.ndarray,
    min_divisor: float,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    if chunk_size < 1:
        raise PreprocessError(f"chunk size must be positive, got {chunk_size}")
    count = images.shape[0]
    for start in range(0, count, chunk_size):
        scaled = np.asarray(images[start : start + chunk_size], dtype=np.float64) / np.float64(255.0)
        normalized = global_contrast_normalize(scaled, min_divisor)
        yield np.reshape(normalized, (normalized.shape[0], -1))


def fit_zca_images(
    images: np.ndarray,
    min_divisor: float,
    regularization: float,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ZCA from uint8 images without keeping a full float64 feature matrix.

    Two passes compute the training mean and the ``/ N`` covariance from
    per-image GCN features. ``images`` must be ``(N, H, W, C)`` uint8.
    """
    _require_uint8_images(images)
    feature_count = int(np.prod(images.shape[1:]))
    total = np.zeros(feature_count, dtype=np.float64)
    count = 0
    for features in _gcn_feature_batches(images, min_divisor, chunk_size):
        total += features.sum(axis=0)
        count += features.shape[0]
    if count != images.shape[0]:
        raise PreprocessError("ZCA fitting did not see every training image")
    mean = total / count
    covariance = np.zeros((feature_count, feature_count), dtype=np.float64)
    seen = 0
    for features in _gcn_feature_batches(images, min_divisor, chunk_size):
        centered = features - mean
        covariance += centered.T @ centered
        seen += features.shape[0]
    if seen != count:
        raise PreprocessError("ZCA covariance pass did not see every training image")
    covariance /= count
    return whitening_from_covariance(mean, covariance, regularization)


def transform_images(
    images: np.ndarray,
    mean: np.ndarray,
    whitening: np.ndarray,
    min_divisor: float,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Scale, GCN, and ZCA-whiten uint8 images into float32 ``(N, H, W, C)``."""
    _require_uint8_images(images)
    height, width, channels = images.shape[1:]
    feature_count = height * width * channels
    if mean.shape != (feature_count,) or whitening.shape != (feature_count, feature_count):
        raise PreprocessError(
            "ZCA parameters do not match image features "
            f"{feature_count}: mean {mean.shape}, matrix {whitening.shape}"
        )
    output = np.empty(images.shape, dtype=np.float32)
    if chunk_size < 1:
        raise PreprocessError(f"chunk size must be positive, got {chunk_size}")
    start = 0
    for features in _gcn_feature_batches(images, min_divisor, chunk_size):
        transformed = apply_zca(features, mean, whitening)
        stop = start + features.shape[0]
        output[start:stop] = transformed.reshape(features.shape[0], height, width, channels)
        start = stop
    if start != images.shape[0]:
        raise PreprocessError("ZCA transform did not cover every image")
    return output


def _require_uint8_images(images: np.ndarray) -> None:
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8 or images.ndim != 4:
        raise PreprocessError(
            "preprocessing images must be a uint8 array of shape (N, H, W, C), "
            f"got {type(images).__name__} "
            f"{getattr(images, 'dtype', None)} {getattr(images, 'shape', None)}"
        )
    if images.shape[0] < 1 or min(images.shape[1:]) < 1:
        raise PreprocessError(f"preprocessing images are empty: shape {images.shape}")
