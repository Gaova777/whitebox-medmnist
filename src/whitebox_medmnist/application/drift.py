"""Cheap per-image signals to watch in production, and the corruptions used to test that watching works."""
import numpy as np
from scipy.ndimage import gaussian_filter, laplace


def features(images: np.ndarray) -> dict[str, np.ndarray]:
    """images: N x C x H x W in [0, 1]. Brightness, contrast, sharpness (variance of the Laplacian), entropy."""
    grey = images.mean(1)
    sharp = np.array([laplace(g).var() for g in grey])
    hist = [np.histogram(g, bins=32, range=(0, 1))[0] / g.size for g in grey]
    entropy = np.array([-(h[h > 0] * np.log2(h[h > 0])).sum() for h in hist])
    return {"brightness": grey.mean((1, 2)), "contrast": grey.std((1, 2)), "sharpness": sharp, "entropy": entropy}


def corrupt(images: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """What a different scanner, a bad acquisition or a compressed upload might do to the images."""
    if kind == "clean":
        return images
    if kind == "noise":
        return np.clip(images + rng.normal(0, 0.08, images.shape), 0, 1).astype(np.float32)
    if kind == "blur":
        return np.stack([gaussian_filter(img, sigma=(0, 1.5, 1.5)) for img in images]).astype(np.float32)
    if kind == "low_contrast":
        mean = images.mean((1, 2, 3), keepdims=True)
        return (mean + 0.5 * (images - mean)).astype(np.float32)
    raise ValueError(kind)
