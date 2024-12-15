from typing import cast
from collections.abc import Callable

import torch
import torch.nn as nn

import numpy as np

from ..searchspace import (
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
from ..searchspace.layers import BaseModule
from ..oneshot.layers import (
    ChoiceLayer,
    ChoiceBlock,
    ChoiceBlockConstructor,
)
from ..oneshot.share import ParameterSharerConstructor
from ..oneshot.helper import l1_reorder_conv2d
from ..mobilenet import (
    FIRST_CONV_CHANNELS,
    LAYER_SETTINGS,
    LAST_CONV_CHANNELS,
    MobileSkeletonNet
)
from ..utils import make_divisible
from . import (
    conv2d_choices,
    dwconv2d_choices,
    bdwrconv2d_choices,
    local_reorder_conv2d,
    local_reorder_dwconv2d,
    local_reorder_bdwrconv2d,
    global_reorder_conv2d,
    global_reorder_dwconv2d,
    global_reorder_bdwrconv2d,
    CommonWarmUpCtx
)


__all__ = [
    "SuperChoiceBlock",
    "super_choice_blocks_wrapper",
    "WarmUpCtxLocalReorder",
    "WarmUpCtxGlobalReorder"
    "FullSearchSpace",
]


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

    def unset(self) -> None:
        if self.active is not None:
            self.active.unset()

        self.active = None

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


class WarmUpCtxLocalReorder(CommonWarmUpCtx):
    _reorder: dict[
        ConvOp,
        Callable[[BaseModule, torch.Tensor | None], torch.Tensor | None]
    ]

    def __init__(self) -> None:
        self._reorder = {
            ConvOp.CONV2D: local_reorder_conv2d,
            ConvOp.DWCONV2D: local_reorder_dwconv2d,
            ConvOp.BDWRCONV2D: local_reorder_bdwrconv2d
        }

    def set(
        self,
        supernet: MobileSkeletonNet,
        max_model: Model,
        max_net: MobileSkeletonNet
    ) -> None:
        op = max_model.blocks[0].layers[0].op

        indices = None
        for i, block in enumerate(supernet.blocks):
            block = cast(SuperChoiceBlock, block)

            choice = cast(ChoiceBlock, block.choices[op])
            for j, layer in enumerate(choice.layers):
                layer = cast(ChoiceLayer, layer)
                src = cast(
                    BaseModule, max_net.blocks[i * max(LAYER_CHOICES) + j]
                )

                indices = self._reorder[op](src, indices)
                layer.sharer._weight_copy(src)

        l1_reorder_conv2d(max_net.last[0], 1, indices)
        self._after_set(max_net)


class WarmUpCtxGlobalReorder(CommonWarmUpCtx):
    _reorder: dict[
        ConvOp,
        Callable[
            [BaseModule, torch.Tensor | None, torch.Tensor | None],
            torch.Tensor | None
        ]
    ]
    _norms: list[torch.Tensor]

    def __init__(self, norms: list[torch.Tensor]) -> None:
        self._reorder = {
            ConvOp.CONV2D: global_reorder_conv2d,
            ConvOp.DWCONV2D: global_reorder_dwconv2d,
            ConvOp.BDWRCONV2D: global_reorder_bdwrconv2d
        }
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
            block = cast(SuperChoiceBlock, block)

            choice = cast(ChoiceBlock, block.choices[op])
            for j, layer in enumerate(choice.layers):
                layer = cast(ChoiceLayer, layer)

                idx = i * max(LAYER_CHOICES) + j
                src = cast(BaseModule, max_net.blocks[idx])

                indices = self._reorder[op](src, indices, self._norms[idx])
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
    possible.

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
    Randomly samples a single instance of a `Model` afrom the full search space.
    The backbone of the search space is MobileNetV2, so use its layer settings
    and input and output channels.

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


class FullSearchSpace(SearchSpace):
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
