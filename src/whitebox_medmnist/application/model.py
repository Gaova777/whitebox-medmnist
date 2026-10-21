"""ResNet-18 two ways: the CIFAR-style variant MedMNIST's benchmark uses at 28 px (3x3 first conv, no max-pool),
and torchvision's ImageNet-pretrained ResNet-18 at 128 px, where heat maps have room for detail."""
import torch
import torchvision
from torch import nn


def build(n_classes: int, in_channels: int, size: int) -> nn.Module:
    m = _build(n_classes, in_channels, size)
    for module in m.modules():  # Guided Backprop hooks the ReLUs; in-place ones would overwrite what it needs
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return m


def _build(n_classes: int, in_channels: int, size: int) -> nn.Module:
    if size <= 32:
        m = torchvision.models.resnet18(num_classes=n_classes)
        m.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        return m
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    if in_channels == 1:  # grey X-rays: average the pretrained RGB filters into one channel
        w = m.conv1.weight.detach().mean(1, keepdim=True)
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        m.conv1.weight = nn.Parameter(w)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


# top to bottom: the order in which the sanity check re-initializes the network
BLOCKS = ["fc", "layer4", "layer3", "layer2", "layer1", "conv1"]


def randomize(model: nn.Module, blocks: list[str], seed: int = 0) -> nn.Module:
    """Re-initialize the named blocks in place (cascading randomization, Adebayo et al. 2018)."""
    torch.manual_seed(seed)
    for name in blocks:
        for module in getattr(model, name).modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    return model
