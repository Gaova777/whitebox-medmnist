import pytest
import torch

from whitebox_medmnist.application.model import build
from whitebox_medmnist.application.saliency import METHODS, explain


@pytest.mark.parametrize("method", METHODS)
def test_every_method_gives_one_map_per_image(method):
    torch.manual_seed(0)
    model = build(3, 1, 28)
    x = torch.rand(2, 1, 28, 28)
    maps = explain(model, x, torch.tensor([0, 2]), method)
    assert maps.shape == (2, 28, 28) and (maps >= 0).all()
