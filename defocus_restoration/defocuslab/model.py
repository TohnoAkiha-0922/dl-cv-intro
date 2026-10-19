"""A deliberately small, known-defocus image-restoration network.

PyTorch is optional: the optics and Wiener modules never import this module.
The two input channels are raw observed intensity and defocus / condition_scale.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    base_channels: int = 8
    condition_scale: float = 2.0

    def __post_init__(self):
        if not isinstance(self.base_channels, int) or self.base_channels < 2:
            raise ValueError("base_channels must be an integer >= 2")
        if not math.isfinite(self.condition_scale) or self.condition_scale <= 0:
            raise ValueError("condition_scale must be finite and positive")


def conditioned_input(observation, defocus_waves, condition_scale=2.0):
    """Construct N x 2 x H x W input without clipping measured intensity.

    The known defocus value is a scalar or one value per image. It is not
    estimated from the image, and the parameter plane is never clipped.
    """
    if observation.ndim != 4 or observation.shape[1] != 1:
        raise ValueError("observation must have shape [N, 1, H, W]")
    if not math.isfinite(condition_scale) or condition_scale <= 0:
        raise ValueError("condition_scale must be finite and positive")
    values = torch.as_tensor(
        defocus_waves, dtype=observation.dtype, device=observation.device
    ).reshape(-1)
    if values.numel() == 1:
        values = values.expand(observation.shape[0])
    if values.numel() != observation.shape[0]:
        raise ValueError("Provide one defocus value per image, or one scalar")
    plane = (values / condition_scale).reshape(-1, 1, 1, 1)
    plane = plane.expand(-1, 1, observation.shape[-2], observation.shape[-1])
    return torch.cat((observation, plane), dim=1)


class _DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
        )


class ConditionalUNet(nn.Module):
    """Two-level residual U-Net, with circular convolutions matching the demo.

    The final convolution starts at zero, so an untrained model initially
    returns the observed image. There is no sigmoid: loss uses raw intensity;
    display and all comparison metrics apply the same [0, 1] clipping later.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config.base_channels
        self.enc1 = _DoubleConv(2, c)
        self.enc2 = _DoubleConv(c, 2 * c)
        self.bottleneck = _DoubleConv(2 * c, 4 * c)
        self.pool = nn.AvgPool2d(2)
        self.dec2 = _DoubleConv(6 * c, 2 * c)
        self.dec1 = _DoubleConv(3 * c, c)
        self.head = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 2 or min(x.shape[-2:]) < 8:
            raise ValueError("network input must be [N, 2, H, W], H/W >= 8")
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        middle = self.bottleneck(self.pool(e2))
        d2 = F.interpolate(middle, size=e2.shape[-2:], mode="nearest")
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")
        d1 = self.dec1(torch.cat((d1, e1), dim=1))
        return x[:, :1] + self.head(d1)
