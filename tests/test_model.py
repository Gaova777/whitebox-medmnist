import numpy as np
import torch

from whitebox_medmnist.application.model import BLOCKS, build, randomize
from whitebox_medmnist.application.training import scores


def test_small_and_large_models_take_their_inputs():
    assert build(2, 1, 28)(torch.zeros(2, 1, 28, 28)).shape == (2, 2)


def test_randomizing_the_top_block_changes_only_that_block():
    m = build(8, 3, 28)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    randomize(m, BLOCKS[:1])
    after = m.state_dict()
    assert not torch.equal(before["fc.weight"], after["fc.weight"])
    assert torch.equal(before["layer4.0.conv1.weight"], after["layer4.0.conv1.weight"])


def test_scores_binary_and_multiclass():
    assert scores(np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0, 1])) == {"acc": 1.0, "auc": 1.0}
    p = np.eye(3)[[0, 1, 2]] * 0.8 + 0.2 / 3
    assert scores(p, np.array([0, 1, 2]))["acc"] == 1.0
