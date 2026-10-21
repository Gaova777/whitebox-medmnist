"""How alike two saliency maps are: the four measures Adebayo et al. (2018) use."""
import numpy as np
from scipy.stats import spearmanr
from skimage.feature import hog
from skimage.metrics import structural_similarity


def normalize(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float64)
    span = m.max() - m.min()
    return (m - m.min()) / span if span > 0 else np.zeros_like(m)


def spearman_abs(a, b) -> float:
    """Rank correlation of absolute values: does the map point at the same places, whatever the sign?"""
    return float(spearmanr(np.abs(a).ravel(), np.abs(b).ravel()).statistic)


def spearman_signed(a, b) -> float:
    return float(spearmanr(a.ravel(), b.ravel()).statistic)


def ssim(a, b) -> float:
    return float(structural_similarity(normalize(a), normalize(b), data_range=1.0))


def hog_correlation(a, b) -> float:
    """Pearson correlation of the histograms of oriented gradients: same shapes, even if shifted a little?"""
    ha = hog(normalize(a), pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    hb = hog(normalize(b), pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    if ha.std() == 0 or hb.std() == 0:
        return 0.0
    return float(np.corrcoef(ha, hb)[0, 1])


def all_measures(a, b) -> dict:
    return {"spearman_abs": spearman_abs(a, b), "spearman_signed": spearman_signed(a, b), "ssim": ssim(a, b),
            "hog": hog_correlation(a, b)}
