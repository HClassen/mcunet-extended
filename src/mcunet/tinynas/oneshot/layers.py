from copy import deepcopy
from typing_extensions import Doc
from typing import cast, Annotated
from collections.abc import Callable

import torch
import torch.nn as nn

from torchvision.ops import Conv2dNormActivation

from ..mobilenet import build_last
from ..searchspace import Layer, Block
from ..searchspace.layers import module_to_layer, BaseModule

from .share import ParameterSharer, ParameterSharerConstructor


__all__ = [
    "ChoiceLayer",
    "ChoicesConstructor",
    "ChoiceBlock",
    "ChoiceBlockConstructor",
    "LastChoiceLayer"
]


type ChoicesConstructor = Annotated[
    Callable[
        [
            tuple[int, ...],
            tuple[int, ...],
            int,
            Callable[..., nn.Module] | None,
            Callable[..., nn.Module] | None
        ],
        list[BaseModule]
    ],
    Doc(
        """
        Creates all possible choices that a `ChoiceLayer` contains.

        Args:
            range_in_channels (tuple[int, ...]):
                All possible numbers of input channels to the layer.
            range_out_channels (tuple[int, ...]):
                All possible numbers of output channels to the layer.
            stride (int):
                The stride.
            norm_layer (Callable[..., nn.Module] | None):
                The constructor for the norm layer.
            activation_layer (Callable[..., nn.Module] | None):
                The constructor for the activation layer.

            Returns:
                list[BaseOp]:
                    All possible choices.
        """
    )
]


class ChoiceLayer(nn.Module):
    active: nn.Module | None
    choices: nn.ModuleList  # An array of `BaseOp` modules

    sharer: ParameterSharer

    def __init__(
        self,
        sharer_constructor: ParameterSharerConstructor,
        choices_constructor: ChoicesConstructor,
        range_in_channels: tuple[int, ...],
        range_out_channels: tuple[int, ...],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None,
        activation_layer: Callable[..., nn.Module] | None
    ) -> None:
        super().__init__()

        self.sharer = sharer_constructor(
            max(range_in_channels), max(range_out_channels),
        )

        choices = choices_constructor(
            range_in_channels,
            range_out_channels,
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

        for choice in self.choices:
            choice = cast(BaseModule, choice)

            if module_to_layer(choice) != layer:
                continue

            self.active = choice
            return

        if self.active is None:
            raise RuntimeError(f"no suitable choice found for {layer}")

    def unset(self) -> None:
        self.active = None

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


class ChoiceBlock(nn.Module):
    active_layers: int | None
    layers: nn.ModuleList  # An array of `ChoiceLayer`

    def __init__(
        self,
        sharer_constructor: ParameterSharerConstructor,
        choices_constructor: ChoicesConstructor,
        range_in_channels: tuple[int, ...],
        range_out_channels: tuple[int, ...],
        depth: int,
        stride: int,
        norm_layer: Callable[..., nn.Module] | None,
        activation_layer: Callable[..., nn.Module] | None
    ) -> None:
        super().__init__()

        layers: list[ChoiceLayer] = [
            ChoiceLayer(
                sharer_constructor,
                choices_constructor,
                range_in_channels,
                range_out_channels,
                stride,
                norm_layer,
                activation_layer
            )
        ]

        depth -= 1

        layers.extend([
            ChoiceLayer(
                sharer_constructor,
                choices_constructor,
                range_out_channels,
                range_out_channels,
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

    def unset(self) -> None:
        self.active_layers = None

        for layer in self.layers:
            layer = cast(ChoiceLayer, layer)

            layer.unset()

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


type ChoiceBlockConstructor = Annotated[
    Callable[
        [
            tuple[int, ...],
            int,
            int,
            float,
            int,
            Callable[..., nn.Module] | None,
            Callable[..., nn.Module] | None
        ],
        tuple[ChoiceBlock, tuple[int, ...]]
    ],
    Doc(
        """
        Creates a single `ChoiceBlock` can contain multiple `ChoiceLayer`.

        Args:
            range_in_channels (tuple[int, ...]):
                All possible numbers of input channels to the layer.
            out_channels (int):
                The number of output channels of the layer as per MobileNet.
            stride (int):
                The stride for this block as per MobileNet.
            width_mult (float):
                The width multiplicator as per `MobileNet`.
            round_nearest (int):
                Used in calculating the amount of channels. When applying
                `width_mult` to the channels, round to the nearest multiple of
                `round_nearest`. Usually `8`.
            norm_layer (Callable[..., nn.Module] | None):
                The constructor for the norm layer.
            activation_layer (Callable[..., nn.Module] | None):
                The constructor for the activation layer.

            Returns:
                tuple[ChoiceBlock, tuple[int, ...]]:
                    The created `ChoiceBlock` as well as its
                    `range_out_channels`.
        """
    )
]


class LastChoiceLayer(ChoiceLayer):
    def __init__(
        self,
        sharer_constructor: ParameterSharerConstructor,
        range_in_channels: tuple[int, ...],
        out_channels: int,
        norm_layer: Callable[..., nn.Module] | None,
        activation_layer: Callable[..., nn.Module] | None
    ) -> None:
        nn.Module.__init__(self)

        max_in_channels = max(range_in_channels)
        self.sharer = sharer_constructor(max_in_channels, out_channels)

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
