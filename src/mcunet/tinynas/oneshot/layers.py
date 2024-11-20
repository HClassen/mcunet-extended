from typing import cast
from abc import ABC, abstractmethod
from collections.abc import Iterator, Iterable, Callable

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

from torchvision.ops import Conv2dNormActivation

from ..mnasnet import (
    CHANNEL_CHOICES,
    KERNEL_CHOICES,
    SE_CHOICES,
    SKIP_CHOICES,
    ConvOp,
    SkipOp,
    Block
)
from ..mnasnet.layers import BaseOp, Conv2dOp, DWConv2dOp, MBConv2dOp
from ..mobilenet import build_last


__all__ = [
    "BaseChoiceOp", "Conv2dChoiceOp", "DWConv2dChoiceOp", "MBConv2dChoiceOp",
    "BaseBlock", "Conv2dBlock", "DWConv2dBlock", "MBConv2dBlock", "SuperBlock",
    "LastConvChoiceOp"
]


def _permutations() -> Iterator[tuple[int, float, SkipOp]]:
    for k in KERNEL_CHOICES:
        for se in SE_CHOICES:
            for skip in SKIP_CHOICES:
                yield (k, se, skip)


class BaseChoiceOp(ABC, nn.Module):
    """
    Represents a single layer in the MobileNetV2 like one-shot NAS network. This
    is a base class and is meant to be inherited from. A subclass must implement
    two methods:
    1. ``_build_op``: returns an instance of a MnsaNet layer
    2. ``_make_shared``: creates the weight tensor shared by the conv2d
        operations
    """
    active: BaseOp | None
    in_channels: Iterable[int]
    out_channels: Iterable[int]

    weight: nn.Parameter
    choices: nn.ModuleList

    def __init__(
        self,
        range_in_channels: Iterable[int],
        range_out_channels: Iterable[int],
        range_expansion_ratios: Iterable[int],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__()

        self.active = None

        max_in_channels = max(range_in_channels)
        max_out_channels = max(range_out_channels)
        max_kernel_size = max(KERNEL_CHOICES)
        max_expansion_ratio = max(range_expansion_ratios)

        self.weight = nn.Parameter(
            self._make_shared(
                max_in_channels,
                max_out_channels,
                max_kernel_size,
                max_expansion_ratio
            )
        )

        choices = []

        for i in range_in_channels:
            for j in range_out_channels:
                for e in range_expansion_ratios:
                    for k, se, skip in _permutations():
                        choice = self._build_op(
                            i, j, se, skip, k,
                            stride,
                            e,
                            norm_layer,
                            activation_layer
                        )
                        choice.conv2d.weight = None

                        choices.append(choice)

        self.choices = nn.ModuleList(choices)

        self.in_channels = range_in_channels
        self.out_channels = range_out_channels

    @abstractmethod
    def _build_op(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip_op: SkipOp,
        kernel_size: int,
        stride: int,
        expansion_ratio: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> BaseOp:
        pass

    @abstractmethod
    def _make_shared(
        self,
        max_in_channels: int,
        max_out_channels: int,
        max_kernel_size: int,
        max_expansion_ratio: int
    ) -> torch.Tensor:
        pass

    def _share_weight(self, choice: BaseOp) -> None:
        _, _, shared_h, shared_w = self.weight.size()

        shared_hc = int((shared_h - 1) / 2)
        shared_wc = int((shared_w - 1) / 2)

        conv2d = choice.conv2d

        c_out = conv2d.out_channels
        c_in = int(conv2d.in_channels / conv2d.groups)

        if isinstance(conv2d.kernel_size, tuple):
            h, w = conv2d.kernel_size
        else:
            h, w = conv2d.kernel_size, conv2d.kernel_size

        hc = int((h - 1) / 2)
        wc = int((w - 1) / 2)

        h_start = shared_hc - hc
        h_end = shared_hc + hc + 1

        w_start = shared_wc - wc
        w_end = shared_wc + wc + 1

        conv2d.weight = nn.Parameter(
            self.weight[:c_out, :c_in, h_start:h_end, w_start:w_end]
        )

    def set(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip_op: SkipOp,
        kernel_size: int
    ) -> None:
        self.active = None

        for choice in self.choices:
            choice = cast(BaseOp, choice)

            if not (
                choice.in_channels == in_channels and
                choice.out_channels == out_channels and
                choice.se_ratio == se_ratio and
                choice.skip == (skip_op == SkipOp.IDENTITY) and
                choice.conv2d.kernel_size == (kernel_size, kernel_size)
            ):
                continue

            self.active = choice
            break

        if self.active is None:
            raise Exception(f"no suitable choice found")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.active is None:
            raise Exception("no choice was selected")

        self._share_weight(self.active)
        output = self.active(inputs)
        self.active.conv2d.weight = None

        return output


class Conv2dChoiceOp(BaseChoiceOp):
    """
    A concrete instance of a ``BaseChoiceOp`` which contains ``Conv2dOp`` of the
    MnasNet search space.
    """
    def _build_op(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip_op: SkipOp,
        kernel_size: int,
        stride: int,
        _: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> BaseOp:
        return Conv2dOp(
            in_channels,
            out_channels,
            se_ratio,
            skip_op,
            kernel_size,
            stride,
            norm_layer,
            activation_layer
        )

    def _make_shared(
        self,
        max_in_channels: int,
        max_out_channels: int,
        max_kernel_size: int,
        max_extension_ratio: int
    ) -> torch.Tensor:
        return torch.Tensor(
            max_out_channels,
            max_in_channels,
            max_kernel_size,
            max_kernel_size
        )


class DWConv2dChoiceOp(BaseChoiceOp):
    """
    A concrete instance of a ``BaseChoiceOp`` which contains ``DWConv2dOp`` of
    the MnasNet search space.
    """
    def _build_op(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip_op: SkipOp,
        kernel_size: int,
        stride: int,
        _: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> BaseOp:
        return DWConv2dOp(
            in_channels,
            out_channels,
            se_ratio,
            skip_op,
            kernel_size,
            stride,
            norm_layer,
            activation_layer
        )

    def _make_shared(
        self,
        max_in_channels: int,
        max_out_channels: int,
        max_kernel_size: int,
        max_extension_ratio: int
    ) -> torch.Tensor:
        return torch.Tensor(
            max_in_channels,
            1,
            max_kernel_size,
            max_kernel_size
        )


class MBConv2dChoiceOp(BaseChoiceOp):
    """
    A concrete instance of a ``BaseChoiceOp`` which contains ``MBConv2dOp`` of
    the MnasNet search space.
    """
    def _build_op(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip_op: SkipOp,
        kernel_size: int,
        stride: int,
        expansion_ratio: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> BaseOp:
        return MBConv2dOp(
            in_channels,
            out_channels,
            expansion_ratio,
            se_ratio,
            skip_op,
            kernel_size,
            stride,
            norm_layer,
            activation_layer
        )

    def _make_shared(
        self,
        max_in_channels: int,
        max_out_channels: int,
        max_kernel_size: int,
        max_extension_ratio: int
    ) -> torch.Tensor:
        hidden = int(round(max_in_channels * max_extension_ratio))

        return torch.Tensor(
            hidden,
            1,
            max_kernel_size,
            max_kernel_size
        )


class BaseBlock(ABC, nn.Module):
    """
    Represents a single block in the MobileNetV2 like one-shot NAS network. This
    is a base class and is meant to be inherited from. A subclass must implement
    one method:
    1. ``_build_choice_op``: returns an instance of a concrete ``BaseChoiceOp``
    """
    layers: nn.ModuleList
    n_layers: int | None

    in_channels: Iterable[int]
    out_channels: Iterable[int]

    def __init__(
        self,
        range_in_channels: Iterable[int],
        range_out_channels: Iterable[int],
        range_expansion_ratios: Iterable[int],
        depth: int,
        stride: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__()

        layers: list[BaseChoiceOp] = [
            self._build_choice_op(
                range_in_channels,
                range_out_channels,
                range_expansion_ratios,
                stride,
                norm_layer,
                activation_layer
            )
        ]

        depth -= 1

        layers.extend([
            self._build_choice_op(
                layers[0].out_channels,
                range_out_channels,
                range_expansion_ratios,
                1,
                norm_layer,
                activation_layer
            ) for _ in range(depth)
        ])

        self.n_layers = None

        self.in_channels = range_in_channels
        self.out_channels = layers[-1].out_channels

        self.layers = nn.ModuleList(layers)

    @abstractmethod
    def _build_choice_op(
        self,
        range_in_channels: Iterable[int],
        range_out_channels: Iterable[int],
        range_expansion_ratios: Iterable[int],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> BaseChoiceOp:
        pass

    def set(self, block: Block) -> None:
        self.n_layers = None

        layers = iter(self.layers)

        first = cast(BaseChoiceOp, next(layers))
        first.set(
            block.in_channels,
            block.out_channels,
            block.se_ratio,
            block.skip_op,
            block.kernel_size
        )

        for layer in layers:
            layer = cast(BaseChoiceOp, layer)

            layer.set(
                block.out_channels,
                block.out_channels,
                block.se_ratio,
                block.skip_op,
                block.kernel_size
            )

        self.n_layers = block.n_layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.n_layers is None:
            raise Exception("no choice was selected")

        for i in range(self.n_layers):
            inputs = self.layers[i](inputs)

        return inputs


class Conv2dBlock(BaseBlock):
    """
    A concrete instance of a ``BaseBlock`` which contains ``depth`` many layers
    of ``Conv2dChoiceOp``.
    """
    def _build_choice_op(
        self,
        range_in_channels: Iterable[int],
        range_out_channels: Iterable[int],
        range_expansion_ratio: Iterable[int],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> Conv2dChoiceOp:
        return Conv2dChoiceOp(
            range_in_channels,
            range_out_channels,
            range_expansion_ratio,
            stride,
            norm_layer,
            activation_layer
        )


class DWConv2dBlock(BaseBlock):
    """
    A concrete instance of a ``BaseBlock`` which contains ``depth`` many layers
    of ``DWConv2dChoiceOp``.
    """
    def _build_choice_op(
        self,
        range_in_channels: Iterable[int],
        range_out_channels: Iterable[int],
        range_expansion_ratio: Iterable[int],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> DWConv2dChoiceOp:
        return DWConv2dChoiceOp(
            range_in_channels,
            range_out_channels,
            range_expansion_ratio,
            stride,
            norm_layer,
            activation_layer
        )


class MBConv2dBlock(BaseBlock):
    """
    A concrete instance of a ``BaseBlock`` which contains ``depth`` many layers
    of ``MBConv2dChoiceOp``.
    """
    def _build_choice_op(
        self,
        range_in_channels: Iterable[int],
        range_out_channels: Iterable[int],
        range_expansion_ratio: Iterable[int],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> MBConv2dChoiceOp:
        return MBConv2dChoiceOp(
            range_in_channels,
            range_out_channels,
            range_expansion_ratio,
            stride,
            norm_layer,
            activation_layer
        )


class SuperBlock(nn.Module):
    """
    An aggregation of all available ``BaseBlock`` implementations.
    """
    active: BaseBlock | None
    in_channels: Iterable[int]
    out_channels: Iterable[int]

    choices: nn.ModuleList

    def __init__(
        self,
        conv2d: Conv2dBlock,
        dwconv2d: DWConv2dBlock,
        mbconv2d: MBConv2dBlock
    ) -> None:
        if not (
            conv2d.in_channels == dwconv2d.in_channels == mbconv2d.in_channels
            and
            conv2d.out_channels == dwconv2d.out_channels == mbconv2d.out_channels
        ):
            raise Exception("missmatch of in_channels our out_channels")

        super().__init__()

        self.active = None

        self.choices = nn.ModuleList([conv2d, dwconv2d, mbconv2d])

        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels

    def get_choice_block(self, op: ConvOp) -> BaseBlock:
        match op:
            case ConvOp.CONV2D:
                return self.choices[0]
            case ConvOp.DWCONV2D:
                return self.choices[1]
            case ConvOp.MBCONV2D:
                return self.choices[2]
            case _:
                raise Exception(f"unknown convolution operation {op}")

    def set(self, block: Block) -> None:
        self.active = None

        match block.conv_op:
            case ConvOp.CONV2D:
                self.active = self.choices[0]
            case ConvOp.DWCONV2D:
                self.active = self.choices[1]
            case ConvOp.MBCONV2D:
                self.active = self.choices[2]
            case _:
                raise Exception(
                    f"unknown convolution operation {block.conv_op}"
                )

        self.active.set(block)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.active is None:
            raise Exception("no choice was selected")

        return self.active(inputs)


class LastConvChoiceOp(nn.Module):
    active: Conv2dNormActivation | None
    in_channels: Iterable[int]
    out_channels: Iterable[int]

    weight: nn.Parameter
    choices: nn.ModuleList

    def __init__(
        self,
        range_in_channels: Iterable[int],
        out_channels: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__()

        self.active = None

        max_in_channels = max(range_in_channels)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, max_in_channels, 1, 1)
        )

        choices = [
            build_last(i, out_channels, norm_layer, activation_layer)
            for i in range_in_channels
        ]

        self.choices = nn.ModuleList(choices)

        self.in_channels = range_in_channels
        self.out_channels = [out_channels]

    def _share_weight(self, choice: Conv2dNormActivation) -> None:
        _, _, shared_h, shared_w = self.weight.size()

        shared_hc = int((shared_h - 1) / 2)
        shared_wc = int((shared_w - 1) / 2)

        conv2d: nn.Conv2d = choice[0]

        c_out, c_in = conv2d.out_channels, conv2d.in_channels
        h, w = conv2d.kernel_size

        hc = int((h - 1) / 2)
        wc = int((w - 1) / 2)

        h_start = shared_hc - hc
        h_end = shared_hc + hc + 1

        w_start = shared_wc - wc
        w_end = shared_wc + wc + 1

        conv2d.weight = nn.Parameter(
            self.weight[:c_out, :c_in, h_start:h_end, w_start:w_end]
        )

    def set(self, in_channels: int) -> None:
        self.active = None

        for choice in self.choices:
            choice = cast(Conv2dNormActivation, choice)

            if choice[0].in_channels != in_channels:
                continue

            self.active = choice
            break

        if self.active is None:
            raise Exception(f"no suitable choice found")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.active is None:
            raise Exception("no choice was selected")

        self._share_weight(self.active)
        output = self.active(inputs)
        self.active[0].weight = None

        return output
