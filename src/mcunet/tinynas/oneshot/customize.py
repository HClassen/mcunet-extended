from typing import cast
from copy import deepcopy
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn as nn

from torchvision.ops import Conv2dNormActivation

from ..mobilenet import build_last, MobileSkeletonNet
from ..searchspace import KERNEL_CHOICES, SkipOp, Layer, Block, Model
from ..searchspace.layers import LayerName, BaseOp

from .share import ParameterSharer
from .helper import base_op_to_conv_op


__all__ = [
    "ChoicesMaker",
    "ChoiceLayer",
    "ChoiceLayerMaker",
    "ChoiceBlock",
    "ChoiceBlocksMaker",
    "LastChoiceLayer",
    "LastChoiceLayerMaker",
    "WarmUpSetter"
]


class ChoicesMaker(ABC):
    @abstractmethod
    def make(
        self,
        range_in_channels: tuple[int, ...],
        range_out_channels: tuple[int, ...],
        range_expansion_ratios: tuple[int, ...],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None,
        activation_layer: Callable[..., nn.Module] | None
    ) -> list[BaseOp]:
        pass


class ChoiceLayer(nn.Module):
    active: nn.Module | None
    choices: nn.ModuleList  # An array of `BaseOp` modules

    sharer: ParameterSharer

    def __init__(
        self,
        sharer: ParameterSharer,
        maker: ChoicesMaker,
        range_in_channels: tuple[int, ...],
        range_out_channels: tuple[int, ...],
        range_expansion_ratios: tuple[int, ...],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None,
        activation_layer: Callable[..., nn.Module] | None
    ) -> None:
        super().__init__()

        self.sharer = sharer

        max_in_channels = max(range_in_channels)
        max_out_channels = max(range_out_channels)
        max_kernel_size = max(KERNEL_CHOICES)
        max_expansion_ratio = max(range_expansion_ratios)
        self.sharer.make_shared(
            out_channels=max_out_channels,
            in_channels=max_in_channels,
            kernel_size=max_kernel_size,
            expansion_ratio=max_expansion_ratio
        )

        choices = maker.make(
            range_in_channels,
            range_out_channels,
            range_expansion_ratios,
            stride,
            norm_layer,
            activation_layer
        )

        for choice in choices:
            self.sharer.unset_shared(choice)

        self.choices = nn.ModuleList(choices)
        self.active = None

    def set(self, layer: Layer) -> None:
        self.active = None
        kernel_size = (layer.kernel_size, layer.kernel_size)

        for choice in self.choices:
            choice = cast(BaseOp, choice)

            if not (
                base_op_to_conv_op(choice) == layer.op
                and choice.in_channels == layer.in_channels
                and choice.out_channels == layer.out_channels
                and choice.se_ratio == layer.se_ratio
                and choice.skip == (layer.skip_op == SkipOp.IDENTITY)
                and choice[LayerName.CONV2D][0].kernel_size == kernel_size
                and choice.expansion_ratio == layer.expansion_ratio
            ):
                continue

            self.active = choice
            return

        if self.active is None:
            raise RuntimeError(f"no suitable choice found for {layer}")

    def get(self, copy: bool = False) -> nn.Module:
        if self.active is None:
            raise RuntimeError("no choice was selected")

        self.sharer.set_shared(self.active)
        return self.active if not copy else deepcopy(self.active)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.active is None:
            raise RuntimeError("no choice was selected")

        self.sharer.set_shared(self.active)
        outputs = self.active(inputs)
        self.sharer.unset_shared(self.active)

        return outputs


class ChoiceLayerMaker():
    _choices_maker: ChoicesMaker
    _sharer_factory: Callable[[], ParameterSharer]

    def __init__(
        self,
        choices_maker: ChoicesMaker,
        sharer_factory: Callable[[], ParameterSharer]
    ) -> None:
        self._choices_maker = choices_maker
        self._sharer_factory = sharer_factory

    def make(
        self,
        range_in_channels: tuple[int, ...],
        range_out_channels: tuple[int, ...],
        range_expansion_ratios: tuple[int, ...],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None,
        activation_layer: Callable[..., nn.Module] | None
    ) -> ChoiceLayer:
        return ChoiceLayer(
            self._sharer_factory(),
            self._choices_maker,
            range_in_channels,
            range_out_channels,
            range_expansion_ratios,
            stride,
            norm_layer,
            activation_layer
        )


class ChoiceBlock(nn.Module):
    active_layers: int | None
    layers: nn.ModuleList  # An array of `ChoiceLayer`

    def __init__(
        self,
        maker: ChoiceLayerMaker,
        range_in_channels: tuple[int, ...],
        range_out_channels: tuple[int, ...],
        range_expansion_ratios: tuple[int, ...],
        depth: int,
        stride: int,
        norm_layer: Callable[..., nn.Module] | None,
        activation_layer: Callable[..., nn.Module] | None
    ) -> None:
        super().__init__()

        layers: list[ChoiceLayer] = [
            maker.make(
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
            maker.make(
                range_out_channels,
                range_out_channels,
                range_expansion_ratios,
                1,
                norm_layer,
                activation_layer
            ) for _ in range(depth)
        ])

        self.active_layers = None
        self.layers = nn.ModuleList(layers)

    def _weight_initialization(self) -> None:
        for layer in self.layers:
            layer = cast(ChoiceLayer, layer)

            layer.sharer._weight_initialization()

    def set(self, block: Block) -> None:
        self.active_layers = None

        for choice_layer, block_layer in zip(self.layers, block.layers):
            choice_layer = cast(ChoiceLayer, choice_layer)

            choice_layer.set(block_layer)

        self.active_layers = len(block.layers)

    def get(self, copy: bool = False) -> list[nn.Module]:
        if self.active_layers is None:
            raise RuntimeError("no choice was selected")

        return [
            cast(ChoiceLayer, self.layers[i]).get(copy)
            for i in range(self.active_layers)
        ]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.active_layers is None:
            raise RuntimeError("no choice was selected")

        for i in range(self.active_layers):
            inputs = self.layers[i](inputs)

        return inputs


class ChoiceBlocksMaker(ABC):
    @abstractmethod
    def make(
        self,
        in_channels: tuple[int, ...],
        width_mult: float,
        round_nearest: int,
        norm_layer: Callable[..., nn.Module] | None,
        activation_layer: Callable[..., nn.Module] | None
    ) -> tuple[list[ChoiceBlock], tuple[int, ...]]:
        pass


class LastChoiceLayer(ChoiceLayer):
    def __init__(
        self,
        sharer: ParameterSharer,
        range_in_channels: tuple[int, ...],
        out_channels: int,
        norm_layer: Callable[..., nn.Module] | None,
        activation_layer: Callable[..., nn.Module] | None
    ) -> None:
        nn.Module.__init__(self)

        self.sharer = sharer

        max_in_channels = max(range_in_channels)
        self.sharer.make_shared(
            out_channels=out_channels, in_channels=max_in_channels
        )

        choices: list[Conv2dNormActivation] = []
        for i in range_in_channels:
            choice = build_last(i, out_channels, norm_layer, activation_layer)
            self.sharer.unset_shared(choice)

            choices.append(choice)

        self.choices = nn.ModuleList(choices)
        self.active = None

    def set(self, in_channels: int) -> None:
        self.active = None

        for choice in self.choices:
            choice = cast(Conv2dNormActivation, choice)

            if choice[0].in_channels != in_channels:
                continue

            self.active = choice
            break

        if self.active is None:
            raise RuntimeError(f"no suitable choice found")

    def get(self, copy: bool = False) -> Conv2dNormActivation:
        if self.active is None:
            raise RuntimeError("no choice was selected")

        self.sharer.set_shared(self.active)
        return self.active if not copy else deepcopy(self.active)


class LastChoiceLayerMaker():
    _sharer_factory: Callable[[], ParameterSharer]

    def __init__(self, sharer_factory: Callable[[], ParameterSharer]) -> None:
        self._sharer_factory = sharer_factory

    def make(
        self,
        range_in_channels: tuple[int, ...],
        out_channels: int,
        norm_layer: Callable[..., nn.Module] | None,
        activation_layer: Callable[..., nn.Module] | None
    ) -> LastChoiceLayer:
        return LastChoiceLayer(
            self._sharer_factory(),
            range_in_channels,
            out_channels,
            norm_layer,
            activation_layer
        )


class WarmUpSetter(ABC):
    def before(self, supernet: MobileSkeletonNet, device) -> None:
        pass

    @abstractmethod
    def set(
        self,
        supernet: MobileSkeletonNet,
        max_model: Model,
        max_net: MobileSkeletonNet
    ) -> None:
        pass

    def after(self, supernet: MobileSkeletonNet) -> None:
        pass
