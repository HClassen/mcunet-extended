from abc import ABC
from enum import StrEnum
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

from torchvision.ops import Conv2dNormActivation, SqueezeExcitation

from . import ConvOp, Layer


__all__ = [
    "LayerName",
    "squeeze_channels",
    "BaseModule", "Conv2dModule", "DWConv2dModule", "BDWRConv2dModule",
    "layer_to_module", "module_to_conv_op", "module_to_layer"
]


class LayerName(StrEnum):
    """
    Names for the different layers in the supported operations. Used to retrieve
    them.
    """
    CONV2D = "conv2d"
    SE = "se"
    PWCONV2D = "pwconv2d"
    EXPANSION = "expansion"


def squeeze_channels(in_channels: int, se_ratio: float) -> int:
    """
    Calculate the squeeze channels for a `SqueezeExcitation` layer.

    Copied from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py

    Args:
        in_channels (int):
            the input channels to the `SqueezeExcitation` layer.
        se_ratio (float):
            The squeeze-and-excitation ration.

    Returns:
        int:
            The intermediate channels for the `SqueezeExcitation` layer.
    """
    return max(1, int(in_channels * se_ratio))


def _add_se(
    layers: list[tuple[str, nn.Module]],
    in_channels: int,
    se_ratio: float,
    activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
) -> None:
    if se_ratio == 0:
        return

    se_channels = squeeze_channels(in_channels, se_ratio)

    layers.append(
        (
            LayerName.SE,
            SqueezeExcitation(in_channels, se_channels, activation_layer)
        )
    )


class BaseModule(ABC, nn.Module):
    layers: nn.ModuleDict
    use_shortcut: bool

    # Configuration of this operation. The rest can be extracted from the
    # ``conv2d`` property.
    in_channels: int
    out_channels: int
    expansion_ratio: int
    se_ratio: float
    shortcut: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        se_ratio: float,
        shortcut: bool,
        stride: _size_2_t
    ) -> None:
        super().__init__()

        self.use_shortcut = (stride == 1 or stride == (1, 1)) \
                            and in_channels == out_channels \
                            and shortcut

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_ratio = expansion_ratio
        self.se_ratio = se_ratio
        self.shortcut = shortcut

    def __getitem__(self, key: LayerName) -> nn.Module:
        return self.layers[key]

    def __contains__(self, key: LayerName) -> bool:
        return key in self.layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for layer in self.layers.values():
            outputs = layer(outputs)

        if self.use_shortcut:
            outputs = inputs + outputs

        return outputs


class Conv2dModule(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        shortcut: bool,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__(
            in_channels, out_channels, 1, se_ratio, shortcut, stride
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


class DWConv2dModule(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        shortcut: bool,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__(
            in_channels, out_channels, 1, se_ratio, shortcut, stride
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


class BDWRConv2dModule(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        se_ratio: float,
        shortcut: bool,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__(
            in_channels, out_channels, expansion_ratio, se_ratio, shortcut, stride
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


def layer_to_module(
    layer: Layer,
    norm_layer: Callable[..., nn.Module] | None,
    activation_layer: Callable[..., nn.Module] | None
) -> BaseModule:
    """
    Converts a `Layer` to a Pytorch module.

    Args:
        layer (Layer):
            The configuration of the layer.
        norm_layer (Callable[..., torch.nn.Module], None):
            The constructor for the norm layer.
        activation_layer (Callable[..., torch.nn.Module], None):
            The constructor for the activation layer.

    Returns:
        BaseModule:
            The Pytorch module.

    Raises:
        RuntimeError:
            If the convolution operation is unknown.
    """
    match layer.op:
        case ConvOp.CONV2D:
            return Conv2dModule(
                layer.in_channels,
                layer.out_channels,
                layer.se_ratio,
                layer.shortcut,
                layer.kernel_size,
                layer.stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        case ConvOp.DWCONV2D:
            return DWConv2dModule(
                layer.in_channels,
                layer.out_channels,
                layer.se_ratio,
                layer.shortcut,
                layer.kernel_size,
                layer.stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        case ConvOp.BDWRCONV2D:
            return BDWRConv2dModule(
                layer.in_channels,
                layer.out_channels,
                layer.expansion_ratio,
                layer.se_ratio,
                layer.shortcut,
                layer.kernel_size,
                layer.stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        case _:
            raise RuntimeError(f"unknown convolution operation: {layer.op}")


def module_to_conv_op(module: BaseModule) -> ConvOp:
    """
    Returns the `ConvOp` of a module.

    Args:
        module (BaseModule):
            The Pytorch module.

    Returns:
        ConvOp:
            The convoltion operation of the module.

    Raises:
        RuntimeError:
            If the convolution operation is unknown.
    """
    if isinstance(module, Conv2dModule):
        return ConvOp.CONV2D

    if isinstance(module, DWConv2dModule):
        return ConvOp.DWCONV2D

    if isinstance(module, BDWRConv2dModule):
        return ConvOp.BDWRCONV2D

    raise RuntimeError(f"unknown convolution operation: {type(module)}")


def module_to_layer(module: BaseModule) -> Layer:
    """
    Converts a `BaseModule` to a `Layer`.

    Args:
        module (BaseModule):
            The Pytorch module.

    Returns:
        Layer:
            The configuration of the module.

    Raises:
        RuntimeError:
            If the convolution operation is unknown.
    """
    kernel_size = module[LayerName.CONV2D][0].kernel_size
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    stride = module[LayerName.CONV2D][0].stride
    if isinstance(stride, int):
        stride = (stride, stride)

    return Layer(
        module_to_conv_op(module),
        module.in_channels,
        module.out_channels,
        kernel_size,
        stride,
        module.expansion_ratio,
        module.shortcut,
        module.se_ratio
    )
