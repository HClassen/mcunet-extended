from typing import cast
from abc import ABC, abstractmethod
from collections.abc import Iterator, Iterable, Callable

import torch
import torch.nn as nn

from torchvision.ops import Conv2dNormActivation

from ..mnasnet import (
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


def _share_weight_conv2d(conv2d: nn.Conv2d, weight: nn.Parameter) -> None:
    _, _, shared_h, shared_w = weight.size()

    shared_hc = int((shared_h - 1) / 2)
    shared_wc = int((shared_w - 1) / 2)

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
        weight[:c_out, :c_in, h_start:h_end, w_start:w_end]
    )


class BaseChoiceOp(ABC, nn.Module):
    """
    Represents a single layer in the MobileNetV2 like one-shot NAS network. This
    is a base class and is meant to be inherited from. A subclass must implement
    two methods:
    1. ``_make_shared``: creates the shared weight tensors shared
    2. ``_make_choices``: creates the choices for an operation type
    """
    active: BaseOp | None
    in_channels: Iterable[int]
    out_channels: Iterable[int]

    # Shared weights common to all operations.
    weight_conv2d: nn.Parameter
    weight_fc1: nn.Parameter
    weight_fc2: nn.Parameter

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

        self._make_shared(
            max_in_channels,
            max_out_channels,
            max_kernel_size,
            max_expansion_ratio
        )

        choices = self._make_choices(
            range_in_channels, range_out_channels, range_expansion_ratios,
            stride, norm_layer, activation_layer
        )

        self.choices = nn.ModuleList(choices)

        self.in_channels = range_in_channels
        self.out_channels = range_out_channels

    @abstractmethod
    def _make_shared(
        self,
        max_in_channels: int,
        max_out_channels: int,
        max_kernel_size: int,
        max_expansion_ratio: int
    ) -> None:
        pass

    @abstractmethod
    def _make_choices(
        self,
        range_in_channels: Iterable[int],
        range_out_channels: Iterable[int],
        range_expansion_ratios: Iterable[int],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> list[BaseOp]:
        pass

    def _set_weight(self, choice: BaseOp) -> None:
        _share_weight_conv2d(choice.layers["conv2d"][0], self.weight_conv2d)

        if "se" in choice.layers:
            _share_weight_conv2d(choice.layers["se"].fc1, self.weight_fc1)
            _share_weight_conv2d(choice.layers["se"].fc2, self.weight_fc2)

        self._post_set_weight(choice)

    def _post_set_weight(self, choice: BaseOp) -> None:
        pass

    def _unset_weight(self, choice: BaseOp) -> None:
        choice.layers["conv2d"][0].weight = None

        if "se" in choice.layers:
            choice.layers["se"].fc1.weight = None
            choice.layers["se"].fc2.weight = None

        self._post_unset_weight(choice)

    def _post_unset_weight(self, choice: BaseOp) -> None:
        pass

    def initialize_shared_weights(self) -> None:
        nn.init.kaiming_normal_(self.weight_conv2d, mode="fan_out")
        nn.init.kaiming_normal_(self.weight_fc1, mode="fan_out")
        nn.init.kaiming_normal_(self.weight_fc2, mode="fan_out")

        self._post_initialize_shared_weights()

    def _post_initialize_shared_weights(self) -> None:
        pass

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
                choice.layers["conv2d"][0].kernel_size == (kernel_size, kernel_size)
            ):
                continue

            self.active = choice
            break

        if self.active is None:
            raise Exception(f"no suitable choice found")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.active is None:
            raise Exception("no choice was selected")

        self._set_weight(self.active)
        output = self.active(inputs)
        self._unset_weight(self.active)

        return output


class Conv2dChoiceOp(BaseChoiceOp):
    """
    A concrete instance of a ``BaseChoiceOp`` which contains ``Conv2dOp`` of the
    MnasNet search space.
    """
    def _make_shared(
        self,
        max_in_channels: int,
        max_out_channels: int,
        max_kernel_size: int,
        max_extension_ratio: int
    ) -> None:
        self.weight_conv2d = nn.Parameter(
            torch.Tensor(
                max_out_channels,
                max_in_channels,
                max_kernel_size,
                max_kernel_size
            )
        )

        squeeze_channels = max(1, int(max_out_channels * max(SE_CHOICES)))

        self.weight_fc1 = nn.Parameter(
            torch.Tensor(
                squeeze_channels,
                max_out_channels,
                1, 1
            )
        )

        self.weight_fc2 = nn.Parameter(
            torch.Tensor(
                max_out_channels,
                squeeze_channels,
                1, 1
            )
        )

    def _make_choices(
        self,
        range_in_channels: Iterable[int],
        range_out_channels: Iterable[int],
        _: Iterable[int],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> list[Conv2dOp]:
        choices: list[Conv2dOp] = []

        for i in range_in_channels:
            for j in range_out_channels:
                for k, se, skip in _permutations():
                    choice = Conv2dOp(
                        i, j, se, skip, k, stride, norm_layer, activation_layer
                    )
                    self._unset_weight(choice)

                    choices.append(choice)

        return choices


class DWConv2dChoiceOp(BaseChoiceOp):
    """
    A concrete instance of a ``BaseChoiceOp`` which contains ``DWConv2dOp`` of
    the MnasNet search space.
    """
    weight_spconv2d: nn.Parameter

    def _make_shared(
        self,
        max_in_channels: int,
        max_out_channels: int,
        max_kernel_size: int,
        max_extension_ratio: int
    ) -> None:
        self.weight_conv2d = nn.Parameter(
            torch.Tensor(
                max_in_channels,
                1,
                max_kernel_size,
                max_kernel_size
            )
        )

        squeeze_channels = max(1, int(max_in_channels * max(SE_CHOICES)))

        self.weight_fc1 = nn.Parameter(
            torch.Tensor(
                squeeze_channels,
                max_in_channels,
                1, 1
            )
        )

        self.weight_fc2 = nn.Parameter(
            torch.Tensor(
                max_in_channels,
                squeeze_channels,
                1, 1
            )
        )

        self.weight_spconv2d = nn.Parameter(
            torch.Tensor(
                max_out_channels,
                max_in_channels,
                1, 1
            )
        )

    def _make_choices(
        self,
        range_in_channels: Iterable[int],
        range_out_channels: Iterable[int],
        _: Iterable[int],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> list[DWConv2dOp]:
        choices: list[DWConv2dOp] = []

        for i in range_in_channels:
            for j in range_out_channels:
                for k, se, skip in _permutations():
                    choice = DWConv2dOp(
                        i, j, se, skip, k, stride, norm_layer, activation_layer
                    )
                    self._unset_weight(choice)

                    choices.append(choice)

        return choices

    def _post_set_weight(self, choice: BaseOp) -> None:
        _share_weight_conv2d(choice.layers["spconv2d"], self.weight_spconv2d)

    def _post_unset_weight(self, choice: BaseOp) -> None:
        choice.layers["spconv2d"].weight = None

    def _post_initialize_shared_weights(self) -> None:
        nn.init.kaiming_normal_(self.weight_spconv2d, mode="fan_out")


class MBConv2dChoiceOp(BaseChoiceOp):
    """
    A concrete instance of a ``BaseChoiceOp`` which contains ``MBConv2dOp`` of
    the MnasNet search space.
    """
    weight_expand: nn.Parameter
    weight_spconv2d: nn.Parameter

    def _make_shared(
        self,
        max_in_channels: int,
        max_out_channels: int,
        max_kernel_size: int,
        max_extension_ratio: int
    ) -> None:
        hidden = int(round(max_in_channels * max_extension_ratio))

        self.weight_expand = nn.Parameter(
            torch.Tensor(
                hidden,
                max_in_channels,
                max_kernel_size,
                max_kernel_size
            )
        )

        self.weight_conv2d = nn.Parameter(
            torch.Tensor(
                hidden,
                1,
                max_kernel_size,
                max_kernel_size
            )
        )

        squeeze_channels = max(1, int(hidden * max(SE_CHOICES)))

        self.weight_fc1 = nn.Parameter(
            torch.Tensor(
                squeeze_channels,
                hidden,
                1, 1
            )
        )

        self.weight_fc2 = nn.Parameter(
            torch.Tensor(
                hidden,
                squeeze_channels,
                1, 1
            )
        )

        self.weight_spconv2d = nn.Parameter(
            torch.Tensor(
                max_out_channels,
                hidden,
                1, 1
            )
        )

    def _make_choices(
        self,
        range_in_channels: Iterable[int],
        range_out_channels: Iterable[int],
        range_expansion_ratios: Iterable[int],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> list[MBConv2dOp]:
        choices: list[MBConv2dOp] = []

        for i in range_in_channels:
            for j in range_out_channels:
                for e in range_expansion_ratios:
                    for k, se, skip in _permutations():
                        choice = MBConv2dOp(
                            i, j, e, se, skip, k, stride, norm_layer, activation_layer
                        )
                        self._unset_weight(choice)

                        choices.append(choice)

        return choices

    def _post_set_weight(self, choice: BaseOp) -> None:
        if "expand" in choice.layers:
            _share_weight_conv2d(choice.layers["expand"][0], self.weight_expand)

        _share_weight_conv2d(choice.layers["spconv2d"], self.weight_spconv2d)

    def _post_unset_weight(self, choice: BaseOp) -> None:
        if "expand" in choice.layers:
            choice.layers["expand"][0].weight = None

        choice.layers["spconv2d"].weight = None

    def _post_initialize_shared_weights(self) -> None:
        nn.init.kaiming_normal_(self.weight_expand, mode="fan_out")
        nn.init.kaiming_normal_(self.weight_spconv2d, mode="fan_out")


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

    def _set_weight(self, choice: Conv2dNormActivation) -> None:
        _share_weight_conv2d(choice[0], self.weight)

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

        self._set_weight(self.active)
        output = self.active(inputs)
        self.active[0].weight = None

        return output
