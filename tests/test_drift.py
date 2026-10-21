import numpy as np

from whitebox_medmnist.application.drift import corrupt, features


def test_corruptions_move_the_signals_they_should():
    rng = np.random.default_rng(0)
    x = rng.random((4, 1, 32, 32)).astype(np.float32)
    base = features(x)
    assert features(corrupt(x, "blur", rng))["sharpness"].mean() < base["sharpness"].mean()
    assert features(corrupt(x, "low_contrast", rng))["contrast"].mean() < base["contrast"].mean()
    assert np.array_equal(corrupt(x, "clean", rng), x)
