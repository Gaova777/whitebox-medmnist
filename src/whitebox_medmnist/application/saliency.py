"""The six saliency methods Adebayo et al. (2018) test, through Captum, for a ResNet-18.

Every method returns one H x W map per image (channels summed by absolute value, GradCAM upsampled to the input).
"""
import torch
from captum.attr import (
    GuidedBackprop,
    GuidedGradCam,
    IntegratedGradients,
    LayerAttribution,
    LayerGradCam,
    NoiseTunnel,
    Saliency,
)

METHODS = ["gradient", "smoothgrad", "integrated_gradients", "guided_backprop", "gradcam", "guided_gradcam"]


def explain(model, x: torch.Tensor, target: torch.Tensor, method: str, seed: int = 0) -> torch.Tensor:
    model.eval()
    x = x.clone().requires_grad_(True)
    if method == "gradient":
        a = Saliency(model).attribute(x, target=target, abs=False)
    elif method == "smoothgrad":
        torch.manual_seed(seed)  # SmoothGrad averages over random noise: the seed is part of the explanation
        a = NoiseTunnel(Saliency(model)).attribute(x, target=target, nt_type="smoothgrad", nt_samples=16,
                                                   stdevs=0.15 * float(x.max() - x.min()), abs=False)
    elif method == "integrated_gradients":
        a = IntegratedGradients(model).attribute(x, target=target, baselines=torch.zeros_like(x), n_steps=32)
    elif method == "guided_backprop":
        a = GuidedBackprop(model).attribute(x, target=target)
    elif method == "gradcam":
        a = LayerGradCam(model, model.layer4).attribute(x, target=target, relu_attributions=True)
        a = LayerAttribution.interpolate(a, x.shape[-2:], interpolate_mode="bilinear")
    elif method == "guided_gradcam":
        a = GuidedGradCam(model, model.layer4).attribute(x, target=target)
    else:
        raise ValueError(method)
    return a.detach().abs().sum(1)
