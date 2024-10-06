from typing import Callable
from functools import cache

import torch
import torch.nn as nn

from torchvision.ops import Conv2dNormActivation

from torchprofile import profile_macs


class MobileNetV2Skeleton(nn.Module):
    """
    A skeleton of the MobileNetV2 architecture. Expects the seven bottleneck blocks
    in the middle to be passed to the instructor. The rest of the architecture
    implementation and weight initialization is copied from
    https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
    """
    resolution: int
    n_classes: int
    features: nn.Sequential
    pool: nn.Sequential  # avg pooling + flatten
    classifier: nn.Sequential  # dropout + linear

    def __init__(
        self,
        resolution: int,
        n_classes: int,
        block: list[nn.Module],
        block_in: int,
        block_out: int,
        last_out: int,
        dropout: float,
        norm_layer: Callable[..., torch.nn.Module] | None = torch.nn.BatchNorm2d,
        activation_layer: Callable[..., torch.nn.Module] | None = torch.nn.ReLU6
    ) -> None:
        super().__init__()

        self.resolution = resolution
        self.n_classes = n_classes

        layers: list[nn.Module] = [
            # Build the first feature layer
            Conv2dNormActivation(
                3,
                block_in,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            ),
            *block,
            # Build the last feature layer.
            Conv2dNormActivation(
                block_out,
                last_out,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        ]

        self.features = nn.Sequential(*layers)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_out, self.n_classes)
        )

        # Weight initialization.
        self._weight_initialization()

    def _weight_initialization(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.features(input)
        x = self.pool(x)
        return self.classifier(x)

    @property
    @cache
    def flops(self) -> int:
        """
        The FLOPs/MACs of this model. They are calculated once and the result is
        cached for future uses.

        Returns:
            int: The computed FLOPs/MACs.
        """
        input = torch.randn(1, 3, self.resolution, self.resolution)

        return profile_macs(self, input)
