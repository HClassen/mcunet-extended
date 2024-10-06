from typing import Callable

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

from torchvision.ops import Conv2dNormActivation, SqueezeExcitation

from .searchspace import SkipOp


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
        SqueezeExcitation(
            in_channels,
            squeeze_channels,
            activation=activation_layer
        )
    )


class BaseOp(nn.Module):
    conv: nn.Sequential
    use_residual: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t,
        skip_op: SkipOp
    ) -> None:
        super().__init__()

        self.use_residual =  stride == 1 and \
                             in_channels == out_channels and \
                             skip_op == SkipOp.IDENTITY

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return input + self.conv(input)
        else:
            return self.conv(input)


class Conv2dOp(BaseOp):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip_op: SkipOp,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: _size_2_t | str | None = None,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__(in_channels, out_channels, stride, skip_op)

        layers: list[nn.Module] = [
            Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                inplace=True
            )
        ]

        _add_se(layers, out_channels, se_ratio, activation_layer)

        self.conv = nn.Sequential(*layers)


class DWConv2dOp(BaseOp):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip_op: SkipOp,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: _size_2_t | str | None = None,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__(in_channels, out_channels, stride, skip_op)

        layers: list[nn.Module] = [
            Conv2dNormActivation(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
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

        self.conv = nn.Sequential(*layers)


class Expand2d(nn.Module):
    expand: Conv2dNormActivation

    def __init__(
        self,
        in_channels: int,
        expand_ratio: float,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__()

        self.expand = Conv2dNormActivation(
            in_channels,
            int(round(in_channels * expand_ratio)),
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            inplace=True
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.expand(input)


class MBConv2dOp(BaseOp):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float,
        se_ratio: float,
        skip_op: SkipOp,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: _size_2_t | str | None = None,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__(in_channels, out_channels, stride, skip_op)

        hidden = int(round(in_channels * expand_ratio))
        layers: list[nn.Module] = []

        if expand_ratio > 1:
            layers.append(
                Expand2d(
                    in_channels,
                    expand_ratio,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer
                )
            )

        layers.append(
            Conv2dNormActivation(
                hidden,
                hidden,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
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

        self.conv = nn.Sequential(*layers)
