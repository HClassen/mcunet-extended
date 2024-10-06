from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

from torchvision.ops import Conv2dNormActivation, SqueezeExcitation


__all__ = ["BaseOp", "Conv2dOp", "DWConv2d", "MBConv2d"]


def _add_se(
    layers: list[nn.Module],
    in_channels: int,
    se_ratio: float,
    activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
) -> None:
    if se_ratio == 0:
        return

    # Copied from
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    squeeze_channels = max(1, int(in_channels * se_ratio))

    layers.append(
        SqueezeExcitation(in_channels, squeeze_channels, activation_layer)
    )


class BaseOp(ABC, nn.Module):
    layers: nn.Sequential
    use_residual: bool

    # Configuration of this operation. The rest can be extracted from the
    # ``conv2d`` property.
    in_channels: int
    out_channels: int
    expansion_ratio: int
    se_ratio: float
    skip: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        se_ratio: float,
        skip: bool,
        stride: _size_2_t
    ) -> None:
        super().__init__()

        self.use_residual =  stride == 1 and \
                             in_channels == out_channels and \
                             skip

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_ratio = expansion_ratio
        self.se_ratio = se_ratio
        self.skip = skip

    @property
    @abstractmethod
    def conv2d(self) -> nn.Conv2d:
        """
        Get the main conv2d for the complete operation.
        """
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return input + self.layers(input)
        else:
            return self.layers(input)


class Conv2dOp(BaseOp):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip: bool,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__(
            in_channels, out_channels, 1, se_ratio, skip, stride
        )

        layers: list[nn.Module] = [
            Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                inplace=True
            )
        ]

        _add_se(layers, out_channels, se_ratio, activation_layer)

        self.layers = nn.Sequential(*layers)

    @property
    def conv2d(self) -> nn.Conv2d:
        return self.layers[0][0]


class DWConv2dOp(BaseOp):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip: bool,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__(
            in_channels, out_channels, 1, se_ratio, skip, stride
        )

        layers: list[nn.Module] = [
            Conv2dNormActivation(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                groups=in_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                inplace=True
            )
        ]

        # The order of operations is guessed from the MnasNet paper and
        # implementation. Both are not clear how to handle SE for the depthwise
        # convolution op, but for the mobile inverted bottleneck SE is performed
        # before the final pointwise convolution. So do the same here.
        _add_se(layers, in_channels, se_ratio, activation_layer)

        layers.extend([
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            norm_layer(out_channels)
        ])

        self.layers = nn.Sequential(*layers)

    @property
    def conv2d(self) -> nn.Conv2d:
        return self.layers[0][0]


class MBConv2dOp(BaseOp):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        se_ratio: float,
        skip: bool,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__(
            in_channels, out_channels, expansion_ratio, se_ratio, skip, stride
        )

        hidden = in_channels * expansion_ratio
        layers: list[nn.Module] = []

        if expansion_ratio > 1:
            layers.append(
                Conv2dNormActivation(
                    in_channels,
                    hidden,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    inplace=True
                )
            )

        layers.append(
            Conv2dNormActivation(
                hidden,
                hidden,
                kernel_size,
                stride,
                groups=hidden,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                inplace=True
            )
        )

        _add_se(layers, hidden, se_ratio, activation_layer)

        layers.extend([
            nn.Conv2d(
                hidden,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            norm_layer(out_channels)
        ])

        self.layers = nn.Sequential(*layers)

    @property
    def conv2d(self) -> nn.Conv2d:
        if self.expansion_ratio == 1:
            return self.layers[0][0]

        return self.layers[1][0]
