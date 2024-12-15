from typing import cast
from collections.abc import Callable

import torch
import torch.nn as nn

import numpy as np

from ..searchspace import (
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
from ..searchspace.layers import BaseModule
from ..oneshot.layers import (
    ChoiceLayer,
    ChoiceBlock,
)
from ..oneshot.helper import l1_reorder_conv2d
from ..mobilenet import (
    FIRST_CONV_CHANNELS,
    LAYER_SETTINGS,
    LAST_CONV_CHANNELS,
    MobileSkeletonNet
)
from ..utils import make_divisible
from . import (
    dwconv2d_choices,
    local_reorder_dwconv2d,
    global_reorder_dwconv2d,
    DWConv2dSharer,
    CommonWarmUpCtx
)


__all__ = [
    "choice_block_constructor",
    "WarmUpCtxLocalReorder",
    "WarmUpCtxGlobalReorder"
    "DWConv2dSearchSpace",
]


def choice_block_constructor(
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

    choiceblock = ChoiceBlock(
        DWConv2dSharer,
        dwconv2d_choices,
        in_channels,
        out_channels,
        depth,
        stride,
        norm_layer,
        activation_layer
    )

    return choiceblock, out_channels


class WarmUpCtxLocalReorder(CommonWarmUpCtx):
    def set(
        self,
        supernet: MobileSkeletonNet,
        max_model: Model,
        max_net: MobileSkeletonNet
    ) -> None:
        op = max_model.blocks[0].layers[0].op

        indices = None
        for i, block in enumerate(supernet.blocks):
            block = cast(ChoiceBlock, block)

            choice = cast(ChoiceBlock, block.choices[op])
            for j, layer in enumerate(choice.layers):
                layer = cast(ChoiceLayer, layer)
                src = cast(
                    BaseModule, max_net.blocks[i * max(LAYER_CHOICES) + j]
                )

                indices = local_reorder_dwconv2d(src, indices)
                layer.sharer._weight_copy(src)

        l1_reorder_conv2d(max_net.last[0], 1, indices)
        self._after_set(max_net)


class WarmUpCtxGlobalReorder(CommonWarmUpCtx):
    _norms: list[torch.Tensor]

    def __init__(self, norms: list[torch.Tensor]) -> None:
        self._norms = norms

    def set(
        self,
        supernet: MobileSkeletonNet,
        max_model: Model,
        max_net: MobileSkeletonNet
    ) -> None:
        op = max_model.blocks[0].layers[0].op

        indices = None
        for i, block in enumerate(supernet.blocks):
            block = cast(ChoiceBlock, block)

            choice = cast(ChoiceBlock, block.choices[op])
            for j, layer in enumerate(choice.layers):
                layer = cast(ChoiceLayer, layer)

                idx = i * max(LAYER_CHOICES) + j
                src = cast(BaseModule, max_net.blocks[idx])

                indices = global_reorder_dwconv2d(
                    src, indices, self._norms[idx]
                )
                layer.sharer._weight_copy(src)

        l1_reorder_conv2d(max_net.last[0], 1, indices)
        self._after_set(max_net)


def _sample_block(
    rng: np.random.Generator,
    depth: int,
    in_channels: int,
    out_channels: int,
    first_stride: int
) -> Block:
    """
    Randomly amples a single instance of a `Block` from all choices
    possible, except the used convolution operation. Keep that fixed
    at `ConvOp.DWCONV2D`.

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
    op = ConvOp.DWCONV2D

    choice = rng.integers(0, len(KERNEL_CHOICES))
    kernel_size = KERNEL_CHOICES[choice]

    choice = rng.integers(0, len(SE_CHOICES))
    se_ratio = SE_CHOICES[choice]

    choice = rng.integers(0, len(SHORTCUT_CHOICES))
    shortcut = SHORTCUT_CHOICES[choice]

    expansion_ratio = 1

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
    Randomly samples a single instance of a `Model` from the `ConvOp.DWConv2d`
    search space. The backbone of the search space is MobileNetV2, so use its
    layer settings and input and output channels.

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

    return Model(width_mult, blocks, last_channels)


def mutator(i: int, width_mult: float, block: Block) -> Block:
    rng = np.random.default_rng()

    depth = LAYER_CHOICES[rng.integers(0, len(LAYER_CHOICES))]

    choice = CHANNEL_CHOICES[rng.integers(0, len(CHANNEL_CHOICES))]
    out_channels = make_divisible(
        LAYER_SETTINGS[i][1] * choice * width_mult, 8
    )

    new = _sample_block(
        rng,
        depth,
        block.layers[0].in_channels,
        out_channels,
        LAYER_SETTINGS[i][3]
    )

    return new


class DWConv2dSearchSpace(SearchSpace):
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
                ConvOp.DWCONV2D,
                max(KERNEL_CHOICES),
                se,
                False,
                max(LAYER_CHOICES),
                max(CHANNEL_CHOICES),
                max(EXPANSION_CHOICES),
                self.width_mult
            ),
        )
