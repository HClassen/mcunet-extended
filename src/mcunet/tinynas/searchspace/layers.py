from abc import ABC
from enum import StrEnum
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

from torchvision.ops import Conv2dNormActivation, SqueezeExcitation


__all__ = ["Layer", "BaseOp", "Conv2dOp", "DWConv2d", "BDWRConv2d"]


class LayerName(StrEnum):
    """
    Names for the different layers in the supported operations. Used to retrieve
    them.
    """
    CONV2D = "conv2d"
    SE = "se"
    PWCONV2D = "pwconv2d"
    EXPANSION = "expansion"


def _add_se(
    layers: list[tuple[str, nn.Module]],
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
        (
            LayerName.SE,
            SqueezeExcitation(in_channels, squeeze_channels, activation_layer)
        )
    )


class BaseOp(ABC, nn.Module):
    layers: nn.ModuleDict
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

    def __getitem__(self, key: LayerName) -> nn.Module:
        return self.layers[key]

    def __contains__(self, key: LayerName) -> bool:
        return key in self.layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for layer in self.layers.values():
            outputs = layer(outputs)

        if self.use_residual:
            outputs = inputs + outputs

        return outputs


class Conv2dOp(BaseOp):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip: bool,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__(
            in_channels, out_channels, 1, se_ratio, skip, stride
        )

        layers: list[tuple[str, nn.Module]] = [
            (
                LayerName.CONV2D,
                Conv2dNormActivation(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    inplace=True
                )
            )
        ]

        _add_se(layers, out_channels, se_ratio, activation_layer)

        self.layers = nn.ModuleDict(layers)


class DWConv2dOp(BaseOp):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip: bool,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__(
            in_channels, out_channels, 1, se_ratio, skip, stride
        )

        layers: list[tuple[str, nn.Module]] = [
            (
                LayerName.CONV2D,
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
            )
        ]

        # The order of operations is guessed from the MnasNet paper and
        # implementation. Both are not clear how to handle SE for the depthwise
        # convolution op, but for the mobile inverted bottleneck SE is performed
        # before the final pointwise convolution. So do the same here.
        _add_se(layers, in_channels, se_ratio, activation_layer)

        layers.append(
            (
                LayerName.PWCONV2D,
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False
                    ),
                    norm_layer(out_channels)
                )
            )
        )

        self.layers = nn.ModuleDict(layers)


class BDWRConv2dOp(BaseOp):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        se_ratio: float,
        skip: bool,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__(
            in_channels, out_channels, expansion_ratio, se_ratio, skip, stride
        )

        hidden = in_channels * expansion_ratio
        layers: list[tuple[str, nn.Module]] = []

        if expansion_ratio > 1:
            layers.append(
                (
                    LayerName.EXPANSION,
                    Conv2dNormActivation(
                        in_channels,
                        hidden,
                        kernel_size=1,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer,
                        inplace=True
                    )
                )
            )

        layers.append(
            (
                LayerName.CONV2D,
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
        )

        _add_se(layers, hidden, se_ratio, activation_layer)

        layers.append(
            (
                LayerName.PWCONV2D,
                nn.Sequential(
                    nn.Conv2d(
                        hidden,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False
                    ),
                    norm_layer(out_channels)
                )
            )
        )

        self.layers = nn.ModuleDict(layers)
