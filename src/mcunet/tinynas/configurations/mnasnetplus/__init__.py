from typing import cast
from collections.abc import Callable, Iterator

import torch
import torch.nn as nn

from torchvision.ops import Conv2dNormActivation

import numpy as np

from ...searchspace import (
    CONV_CHOICES,
    KERNEL_CHOICES,
    SE_CHOICES,
    SHORTCUT_CHOICES,
    LAYER_CHOICES,
    CHANNEL_CHOICES,
    EXPANSION_CHOICES,
    uniform_layers,
    uniform_model,
    ConvOp,
    Block,
    Model,
    SearchSpace
)
from ...searchspace.layers import (
    Conv2dModule,
    DWConv2dModule,
    BDWRConv2dModule
)
from ...oneshot import WarmUpCtx
from ...oneshot.layers import (
    ChoiceBlock,
    ChoiceBlockConstructor,
    LastChoiceLayer
)
from ...oneshot.share import (
    share_conv2d,
    share_batchnorm2d,
    ParameterSharer,
    ParameterSharerConstructor
)
from ...oneshot.helper import has_norm
from ...mobilenet import (
    FIRST_CONV_CHANNELS,
    LAYER_SETTINGS,
    LAST_CONV_CHANNELS,
    MobileSkeletonNet
)
from ...utils import make_divisible


__all__ = [
    "LastLayerSharer",
    "conv2d_choices",
    "dwconv2d_choices",
    "bdwrconv2d_choices",
    "SuperChoiceBlock",
    "super_choice_blocks_wrapper",
    "MnasNetPlus"
]


class LastLayerSharer(ParameterSharer):
    conv2d_weight: nn.Parameter
    norm_weight: nn.Parameter
    norm_bias: nn.Parameter

    def _make_shared(self, max_in_channels: int, max_out_channels: int) -> None:
        self.conv2d_weight = nn.Parameter(
            torch.Tensor(
                max_out_channels,
                max_in_channels,
                1, 1
            )
        )
        self.norm_weight = nn.Parameter(torch.Tensor(max_out_channels))
        self.norm_bias = nn.Parameter(torch.Tensor(max_out_channels))

    def set_shared(self, choice: Conv2dNormActivation) -> None:
        share_conv2d(choice[0], self.conv2d_weight, None)

        if has_norm(choice):
            share_batchnorm2d(
                choice[1], self.norm_weight, self.norm_bias, None, None, None
            )

    def unset_shared(self, choice: Conv2dNormActivation) -> None:
        choice[0].weight = None

        if has_norm(choice):
            choice[1].weight = None
            choice[1].bias = None

    def _weight_initialization(self) -> None:
        nn.init.kaiming_normal_(self.conv2d_weight, mode="fan_out")
        nn.init.ones_(self.norm_weight)
        nn.init.zeros_(self.norm_bias)


def _permutations() -> Iterator[tuple[int, float, bool]]:
    for k in KERNEL_CHOICES:
        for se in SE_CHOICES:
            for shortcut in SHORTCUT_CHOICES:
                yield (k, se, shortcut)


def conv2d_choices(
    range_in_channels: tuple[int, ...],
    range_out_channels: tuple[int, ...],
    stride: int,
    norm_layer: Callable[..., nn.Module] | None,
    activation_layer: Callable[..., nn.Module] | None
) -> list[Conv2dModule]:
    choices: list[Conv2dModule] = []

    for i in range_in_channels:
        for j in range_out_channels:
            for k, se, shortcut in _permutations():
                choices.append(Conv2dModule(
                    i, j, se, shortcut, k, stride, norm_layer, activation_layer
                ))

    return choices


def dwconv2d_choices(
    range_in_channels: tuple[int, ...],
    range_out_channels: tuple[int, ...],
    stride: int,
    norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
    activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
) -> list[DWConv2dModule]:
    choices: list[DWConv2dModule] = []

    for i in range_in_channels:
        for j in range_out_channels:
            for k, se, shortcut in _permutations():
                choices.append(DWConv2dModule(
                    i, j, se, shortcut, k, stride, norm_layer, activation_layer
                ))

    return choices


def bdwrconv2d_choices(
    range_in_channels: tuple[int, ...],
    range_out_channels: tuple[int, ...],
    stride: int,
    norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
    activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
) -> list[BDWRConv2dModule]:
    choices: list[BDWRConv2dModule] = []

    for i in range_in_channels:
        for j in range_out_channels:
            for e in EXPANSION_CHOICES:
                for k, se, shortcut in _permutations():
                    choices.append(BDWRConv2dModule(
                        i, j, e, se, shortcut, k, stride,
                        norm_layer, activation_layer
                    ))

    return choices


class SuperChoiceBlock(ChoiceBlock):
    """
    An aggregation of all available `ChoiceBlock` implementations.
    """
    active: ChoiceBlock | None
    choices: nn.ModuleList

    def __init__(
        self,
        conv2d: ChoiceBlock,
        dwconv2d: ChoiceBlock,
        bdwrconv2d: ChoiceBlock
    ) -> None:
        nn.Module.__init__(self)

        self.active = None
        self.choices = nn.ModuleList([conv2d, dwconv2d, bdwrconv2d])

    def _weight_initialization(self) -> None:
        for choice in self.choices:
            choice = cast(ChoiceBlock, choice)

            choice._weight_initialization()

    def set(self, block: Block) -> None:
        self.active = self.choices[block.layers[0].op]
        self.active.set(block)

    def get(self, copy: bool = False) -> list[nn.Module]:
        if self.active is None:
            raise RuntimeError("no choice was selected")

        return self.active.get(copy)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.active is None:
            raise RuntimeError("no choice was selected")

        return self.active(inputs)


def super_choice_blocks_wrapper(
    conv2d_sharer: ParameterSharerConstructor,
    dwconv2d_sharer: ParameterSharerConstructor,
    bdwrconv2d_sharer: ParameterSharerConstructor
) -> ChoiceBlockConstructor:
    def super_choice_blocks(
        in_channels: tuple[int, ...],
        out_channels: int,
        stride: int,
        width_mult: float,
        round_nearest: int,
        norm_layer: Callable[..., nn.Module] | None,
        activation_layer: Callable[..., nn.Module] | None
    ) -> tuple[ChoiceBlock, tuple[int, ...]]:
        depth = max(LAYER_CHOICES)

        out_channels = set([
            make_divisible(out_channels * choice * width_mult, round_nearest)
            for choice in CHANNEL_CHOICES
        ])

        conv2d = ChoiceBlock(
            conv2d_sharer,
            conv2d_choices,
            in_channels,
            out_channels,
            depth,
            stride,
            norm_layer,
            activation_layer
        )

        dwconv2d = ChoiceBlock(
            dwconv2d_sharer,
            dwconv2d_choices,
            in_channels,
            out_channels,
            depth,
            stride,
            norm_layer,
            activation_layer
        )

        bdwrconv2d = ChoiceBlock(
            bdwrconv2d_sharer,
            bdwrconv2d_choices,
            in_channels,
            out_channels,
            depth,
            stride,
            norm_layer,
            activation_layer
        )

        superblock = SuperChoiceBlock(conv2d, dwconv2d, bdwrconv2d)

        return superblock, out_channels

    return super_choice_blocks


class CommonWarmUpCtx(WarmUpCtx):
    _first_parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    _last_parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    _classifier_parameters: tuple[torch.Tensor, torch.Tensor]

    def pre(self, supernet: MobileSkeletonNet, device) -> None:
        first = supernet.first
        self._first_parameters = (
            torch.zeros_like(first[0].weight, device=device),
            torch.zeros_like(first[1].weight, device=device),
            torch.zeros_like(first[1].bias, device=device)
        )

        last = cast(LastChoiceLayer, supernet.last)
        sharer = cast(LastLayerSharer, last.sharer)
        self._last_parameters = (
            torch.zeros_like(sharer.conv2d_weight, device=device),
            torch.zeros_like(sharer.norm_weight, device=device),
            torch.zeros_like(sharer.norm_bias, device=device)
        )

        classifier = supernet.classifier
        self._classifier_parameters = (
            torch.zeros_like(classifier[1].weight, device=device),
            torch.zeros_like(classifier[1].bias, device=device)
        )

    def _after_set(self, max_net: MobileSkeletonNet) -> None:
        first = max_net.first
        self._first_parameters[0].add_(first[0].weight)
        self._first_parameters[1].add_(first[1].weight)
        self._first_parameters[2].add_(first[1].bias)

        last = max_net.last
        self._last_parameters[0].add_(last[0].weight)
        self._last_parameters[1].add_(last[1].weight)
        self._last_parameters[2].add_(last[1].bias)

        classifier = max_net.classifier
        self._classifier_parameters[0].add_(classifier[1].weight)
        self._classifier_parameters[1].add_(classifier[1].bias)

    def post(self, supernet: MobileSkeletonNet) -> None:
        for shared in self._first_parameters:
            shared.div_(3)

        for shared in self._last_parameters:
            shared.div_(3)

        for shared in self._classifier_parameters:
            shared.div_(3)

        with torch.no_grad():
            first = supernet.first
            first[0].weight.copy_(self._first_parameters[0])
            first[1].weight.copy_(self._first_parameters[1])
            first[1].bias.copy_(self._first_parameters[2])

            last = cast(LastChoiceLayer, supernet.last)
            sharer = cast(LastLayerSharer, last.sharer)
            sharer.conv2d_weight.copy_(self._last_parameters[0])
            sharer.norm_weight.copy_(self._last_parameters[1])
            sharer.norm_bias.copy_(self._last_parameters[2])

            classifier = supernet.classifier
            classifier[1].weight.copy_(self._classifier_parameters[0])
            classifier[1].bias.copy_(self._classifier_parameters[1])


def _sample_block(
    rng: np.random.Generator,
    depth: int,
    in_channels: int,
    out_channels: int,
    first_stride: int
) -> Block:
    """
    Randomly amples a single instance of a ``Block`` according to the
    MnasNet search space.

    Args:
        rng (numpy.random.Generator):
            A random number generator to uniformly distribute the blocks
            parameters.
        depth (int):
            The number of layers.
        in_channels (int):
            The number of in channels.
        out_channels (int):
            The number of out channels.
        first_stride (int):
            The stride for the first layer.

    Returns:
        Block:
            The newly created block.
    """
    choice = rng.integers(0, len(CONV_CHOICES))
    op = CONV_CHOICES[choice]

    choice = rng.integers(0, len(KERNEL_CHOICES))
    kernel_size = KERNEL_CHOICES[choice]

    choice = rng.integers(0, len(SE_CHOICES))
    se_ratio = SE_CHOICES[choice]

    choice = rng.integers(0, len(SHORTCUT_CHOICES))
    shortcut = SHORTCUT_CHOICES[choice]

    expansion_ratio = EXPANSION_CHOICES[
        rng.integers(0, len(EXPANSION_CHOICES))
    ] if op == ConvOp.BDWRCONV2D else 1

    layers = uniform_layers(
        op,
        in_channels,
        out_channels,
        kernel_size,
        first_stride,
        expansion_ratio,
        shortcut,
        se_ratio,
        depth
    )

    return Block(layers)


def _sample_model(rng: np.random.Generator, width_mult: float) -> Model:
    """
    Randomly samples a single instance of a `Model` according to the
    MnasNetPlus search space configuration. The MnasNet search space makes
    use of the MobileNetV2 layer settings and input and output channels.
    These settings are used here.

    Args:
        rng (numpy.random.Generator):
            A random number generator to uniformly distribute the models
            parameters.
        width_mult (float):
            Modifies the input and output channels of the hidden layers.

    Returns:
        Model:
            The newly created model.
    """
    blocks: list[Block] = []

    round_nearest = 8

    # As per MobileNetV2 architecture. Copied from
    # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
    in_channels = make_divisible(
        FIRST_CONV_CHANNELS * width_mult, round_nearest
    )

    for settings in LAYER_SETTINGS:
        depth = LAYER_CHOICES[rng.integers(0, len(LAYER_CHOICES))]

        choice = CHANNEL_CHOICES[rng.integers(0, len(CHANNEL_CHOICES))]
        out_channels = make_divisible(
            settings[1] * choice * width_mult, round_nearest
        )

        blocks.append(
            _sample_block(rng, depth, in_channels, out_channels, settings[3])
        )

        in_channels = out_channels

    last_channels = make_divisible(
        LAST_CONV_CHANNELS * max(1.0, width_mult), round_nearest
    )

    return Model(blocks, last_channels)


class MnasNetPlus(SearchSpace):
    _max_model_with_se: bool

    def __init__(
        self, width_mult: float, resolution: int, max_model_with_se: bool = True
    ) -> None:
        super().__init__(width_mult, resolution)

        self._max_model_with_se = max_model_with_se

    def sample_model(self, rng: np.random.Generator) -> Model:
        return _sample_model(rng, self.width_mult)

    def max_models(self) -> tuple[Model, ...]:
        se = max(SE_CHOICES) if self._max_model_with_se else min(SE_CHOICES)
        return (
            uniform_model(
                op,
                max(KERNEL_CHOICES),
                se,
                False,
                max(LAYER_CHOICES),
                max(CHANNEL_CHOICES),
                max(EXPANSION_CHOICES),
                self.width_mult
            )
            for op in (ConvOp.CONV2D, ConvOp.DWCONV2D, ConvOp.BDWRCONV2D)
        )
