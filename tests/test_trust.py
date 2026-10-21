import numpy as np

from whitebox_medmnist.application.similarity import all_measures
from whitebox_medmnist.domain.trust import badge, passes_sanity, top_k_overlap


def test_badge_levels():
    assert badge(False, 0.9)["level"] == "not an explanation"
    assert badge(True, 0.3)["level"] == "unstable"
    assert badge(True, 0.8)["level"] == "trusted"
    assert passes_sanity(0.2) and not passes_sanity(0.7)


def test_top_k_overlap():
    assert top_k_overlap([5, 4, 3, 0], [5, 4, 0, 3], k=2) == 1.0
    assert top_k_overlap([5, 4, 0, 0], [0, 0, 4, 5], k=2) == 0.0


def test_a_map_is_identical_to_itself_and_unlike_noise():
    rng = np.random.default_rng(0)
    m = np.zeros((32, 32))
    m[8:16, 8:16] = 1.0
    m += 0.01 * rng.standard_normal((32, 32))
    same = all_measures(m, m)
    assert all(abs(v - 1) < 1e-6 for v in same.values())
    noise = all_measures(m, rng.standard_normal((32, 32)))
    assert noise["spearman_abs"] < 0.2 and noise["ssim"] < 0.3
