from typing import cast
from copy import deepcopy
from collections.abc import Callable, Iterable, Iterator

import torch
import torch.nn as nn

from torchvision.ops import Conv2dNormActivation

import numpy as np

from ..mobilenet import (
    LAYER_SETTINGS,
    FIRST_CONV_CHANNELS,
    LAST_CONV_CHANNELS,
    build_first,
    build_pool,
    build_classifier,
    MobileSkeletonNet
)
from ..mnasnet import (
    LAYER_CHOICES,
    CHANNEL_CHOICES,
    EXPANSION_CHOICES,
    ConvOp,
    Block,
    Model
)
from ..mnasnet.layers import BaseOp, Conv2dOp, DWConv2dOp, MBConv2dOp
from ..utils import make_divisible

from .layers import *


__all__ = [
    "OneShotNet", "SuperNet", "initial_population", "crossover", "mutate"
]


def _get_conv_op(op: BaseOp) -> ConvOp:
    if isinstance(op, Conv2dOp):
        return ConvOp.CONV2D

    if isinstance(op, DWConv2dOp):
        return ConvOp.DWCONV2D

    if isinstance(op, MBConv2dOp):
        return ConvOp.MBCONV2D

    raise ValueError(f"unknown convolution operation: {type(op)}")


class OneShotNet(MobileSkeletonNet):
    _block_lengths: list[int]

    def __init__(
        self,
        first: nn.Module,
        blocks: Iterable[nn.Module],
        last: nn.Module,
        pool: nn.Sequential,
        classifier: nn.Sequential,
        block_lengths: list[int]
    ) -> None:
        super().__init__(first, blocks, last, pool, classifier, False)

        self._block_lengths = block_lengths

    def to_model(self) -> Model:
        """
        Build a ``Model`` from this ``OneShotNet``.

        Returns:
            Model:
                The configuration of this network.
        """
        accum = 0
        firsts: list[BaseOp] = []
        for length in self._block_lengths:
            firsts.append(self.blocks[accum])

            accum += length

        blocks: list[Block] = []
        for block, length in zip(firsts, self._block_lengths):
            blocks.append(
                Block(
                    length,
                    block.in_channels,
                    block.out_channels,
                    block.layers["conv2d"][0].stride,
                    block.expansion_ratio,
                    _get_conv_op(block),
                    block.layers["conv2d"][0].kernel_size,
                    block.se_ratio, block.skip
                )
            )

        return Model(blocks, self.blocks[-1].out_channels)


class SuperNet(MobileSkeletonNet):
    first: Conv2dNormActivation
    last: LastConvChoiceOp

    def __init__(
        self,
        classes: int,
        width_mult: float,
        dropout: float = 0.2,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        round_nearest = 8
        in_channels = [make_divisible(
            FIRST_CONV_CHANNELS * width_mult, round_nearest
        )]

        # Build the first conv2d layer.
        first = build_first(in_channels[0], norm_layer, activation_layer)

        # Build the 7 blocks.
        blocks: list[SuperBlock] = []

        depth = max(LAYER_CHOICES)
        for _, c, _, s in LAYER_SETTINGS:
            out_channels = set([
                make_divisible(c * choice * width_mult, round_nearest)
                for choice in CHANNEL_CHOICES
            ])

            conv2d = Conv2dBlock(
                in_channels,
                out_channels, [1], depth, s,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )

            dwconv2d = DWConv2dBlock(
                in_channels,
                out_channels, [1], depth, s,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )

            mbconv2d = MBConv2dBlock(
                in_channels,
                out_channels, EXPANSION_CHOICES, depth, s,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )

            superblock = SuperBlock(conv2d, dwconv2d, mbconv2d)
            blocks.append(superblock)

            in_channels = superblock.out_channels

        # Build the last conv2d layer.
        last_out = make_divisible(
            LAST_CONV_CHANNELS * max(1.0, width_mult), round_nearest
        )

        last = LastConvChoiceOp(
            in_channels, last_out, norm_layer, activation_layer
        )

        pool = build_pool()
        classifier = build_classifier(classes, last_out, dropout)

        super().__init__(first, blocks, last, pool, classifier)

    def _weight_initialization(self) -> None:
        super()._weight_initialization()

        for m in self.blocks.modules():
            if isinstance(m, BaseChoiceOp):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

        nn.init.kaiming_normal_(self.last.weight, mode="fan_out")

    def set(self, model: Model) -> None:
        if len(self.blocks) != len(model.blocks):
            raise Exception(f"invalid blocks length {len(model.blocks)}")

        for superblock, block in zip(self.blocks, model.blocks):
            superblock = cast(SuperBlock, superblock)

            superblock.set(block)

        self.last.set(model.blocks[-1].out_channels)

    def get(self, copy: bool = False) -> OneShotNet:
        """
        Collects the current active operations in a single net. A call to
        ``set`` is requiered beforehand.

        Args:
            copy (bool):
                If ``True`` perform a deepcopy on all operations specific to
                the current active operations.

        Returns:
            OneShotNet:
                The concrete net.
        """
        blocks: list[nn.Module] = []
        block_lengths: list[int] = []

        for superblock in self.blocks:
            superblock = cast(SuperBlock, superblock)

            if superblock.active is None:
                raise Exception("no choice was selected")

            active = superblock.active
            if active.n_layers is None:
                raise Exception("no choice was selected")

            block_lengths.append(active.n_layers)

            for i in range(active.n_layers):
                layer = cast(BaseChoiceOp, active.layers[i])

                if layer.active is None:
                    raise Exception("no choice was selected")

                layer._share_weight(layer.active)

                blocks.append(layer.active)

        if self.last.active is None:
            raise Exception("no choice was selected")

        self.last._share_weight(self.last.active)
        last = self.last.active

        if copy:
            blocks = deepcopy(blocks)
            last = deepcopy(last)

        return OneShotNet(
            self.first,
            blocks,
            last,
            self.pool,
            self.classifier,
            block_lengths
        )


def initial_population(
    space: Iterator[Model],
    size: int,
    supernet: SuperNet,
    fitness: Callable[[Model], bool]
) -> list[OneShotNet]:
    """
    Select the initial population of models for evolution search.

    Args:
        space (Iterator[Model]):
            An iterator over the search space, which (randomly) samples models.
        size (int):
            The initial population size.
        supernet (SuperNet):
            A (pre trained) instance of ``SuperNet`` to get the actual NNs from.
        fitness (Callable[[Model], bool]):
            A function to evaluate the fitness of a model for some criterion.

    Returns:
        list[Model]:
            The selected population.
    """
    population: list[OneShotNet] = []

    for _ in range(size):
        model = next(space)

        while not fitness(model):
            model = next(space)

        supernet.set(model)
        population.append(supernet.get(True))

    return population


def _copy_batchnorm(
    new: nn.BatchNorm2d, old: nn.BatchNorm2d, c_out: int
) -> None:
    new.weight = nn.Parameter(old.weight[:c_out])
    new.bias = nn.Parameter(old.bias[:c_out])
    new.running_mean = old.running_mean[:c_out]
    new.running_var = old.running_var[:c_out]


def _copy_conv2d(
    new: Conv2dNormActivation, old: Conv2dNormActivation, c_in: int, c_out: int
) -> None:
    new[0].weight = nn.Parameter(old[0].weight[:c_out, :c_in])
    _copy_batchnorm(new[1], old[1], c_out)


def _copy_squeeze(
    new: nn.Conv2d, old: nn.Conv2d, c_in: int, c_out: int
) -> None:
    new.weight = nn.Parameter(old.weight[:c_out, :c_in])
    new.bias = nn.Parameter(old.bias[:c_out])


def _reduce_channels(op: BaseOp, in_channels: int, out_channels: int) -> BaseOp:
    expansion_ratio = op.expansion_ratio
    se_ratio = op.se_ratio
    skip = op.skip
    kernel_size = op.layers["conv2d"][0].kernel_size
    stride = op.layers["conv2d"][0].stride

    reduced = None

    if isinstance(op, Conv2dOp):
        reduced = Conv2dOp(
            in_channels,
            out_channels,
            se_ratio,
            skip,
            kernel_size,
            stride,
        )

        _copy_conv2d(
            reduced.layers["conv2d"], op.layers["conv2d"], in_channels, out_channels
        )

        in_channels = out_channels
    elif isinstance(op, DWConv2dOp):
        reduced = DWConv2dOp(
            in_channels,
            out_channels,
            se_ratio,
            skip,
            kernel_size,
            stride,
        )

        _copy_conv2d(
            reduced.layers["expand"], op.layers["expand"], in_channels, in_channels
        )

        reduced.layers["spconv2d"].weight = nn.Parameter(
            op.layers["spconv2d"].weight[:out_channels, :in_channels]
        )
        _copy_batchnorm(
            reduced.layers["spbatchnorm"], op.layers["spbatchnorm"], out_channels
        )
    elif isinstance(op, MBConv2dOp):
        reduced = MBConv2dOp(
            in_channels,
            out_channels,
            expansion_ratio,
            se_ratio,
            skip,
            kernel_size,
            stride,
        )

        hidden = in_channels * expansion_ratio
        if expansion_ratio > 1:
            _copy_conv2d(
                reduced.layers["expand"], op.layers["expand"], in_channels, hidden
            )

        _copy_conv2d(
            reduced.layers["conv2d"], op.layers["conv2d"], hidden, hidden
        )

        reduced.layers["spconv2d"].weight = nn.Parameter(
            op.layers["spconv2d"].weight[:out_channels, :hidden]
        )
        _copy_batchnorm(
            reduced.layers["spbatchnorm"], op.layers["spbatchnorm"], out_channels
        )

        in_channels = hidden
    else:
        raise Exception(f"unknown operation {type(op)}")

    if se_ratio > 0.0:
        out, _, _, _ = reduced.layers["se"].fc1.weight.size()

        _copy_squeeze(
            reduced.layers["se"].fc1,
            op.layers["se"].fc1,
            in_channels,
            out
        )

        _copy_squeeze(
            reduced.layers["se"].fc2,
            op.layers["se"].fc2,
            out,
            in_channels
        )

    return reduced


def crossover(p1: OneShotNet, p2: OneShotNet) -> OneShotNet:
    """
    Performs single point crossover on ``p1`` and ``p1``. Randomly select a
    split point ``s`` of the seven inner blocks. Then select the blocks ``0`` to
    ``s`` from ``p1`` and merge them with the blocks ``s + 1`` to ``6`` from
    ``p2`` to create the offspring model. ``s`` is chosen from [0, 5], so that
    at least one block of a parent is part of the child.

    To adjust for different ``out_channels`` of the last selected layer from
    ``p1`` and ``in_channels`` of the first selected layer from ``p2`` take the
    result of ``min(out_channels, in_channels)``. If ``out_channels`` is the
    minimum then reduce ``in_channels`` to ``out_channels``. If ``in_channels``
    is the minimum reduce the ``in_channels`` and ``out_channels`` of all layers
    in the ``p1`` block to the minimum.

    Also take the last conv2d layer of ``p2`` as it matches the out_channels of
    the newly created seven blocks.

    Since the first conv2d, pooling and classification are shared between
    choices in the ``SuperNet`` just take them from ``p1``.

    Args:
        p1 (OneShotNet):
            The first parent.
        p2 (OneShotNet):
            The second parent.

    Returns:
        OneShotNet:
                The offspring model.
    """
    split = torch.randint(0, 6, [1]).item()

    p1_split_lengths = p1._block_lengths[:split + 1]
    p2_split_lengths = p2._block_lengths[split + 1:]

    p1_split = sum(p1_split_lengths)
    p2_split = len(p2.blocks) - sum(p2_split_lengths)

    p1_blocks = deepcopy(p1.blocks[:p1_split])
    p2_blocks = deepcopy(p2.blocks[p2_split:])

    p1_out_channels = p1_blocks[-1].out_channels
    p2_in_channels = p2_blocks[0].in_channels

    if p1_out_channels < p2_in_channels:
        # Only need to adjust the in_channels of the first p2 layer.
        p2_blocks[0] = _reduce_channels(
            p2_blocks[0], p1_out_channels, p2_blocks[0].out_channels
        )
    elif p2_in_channels < p1_out_channels:
        # All layers of the last p1 block need their out_channels adjusted.
        # For all but the first of this block also adjust the in_channels.
        start = sum(p1_split_lengths[:-1])
        p1_blocks[start] = _reduce_channels(
            p1_blocks[start],
            p1_blocks[start].in_channels,
            p2_in_channels
        )

        for i in range(start + 1, p1_split):
            p1_blocks[i] = _reduce_channels(
                p1_blocks[i],
                p2_in_channels,
                p2_in_channels
            )

    blocks = p1_blocks + p2_blocks
    last = deepcopy(p2.last)

    return OneShotNet(
        p1.first,
        blocks,
        last,
        p1.pool,
        p1.classifier,
        p1_split_lengths + p2_split_lengths
    )


def mutate(oneshot: OneShotNet, p: float) -> OneShotNet:
    """
    Mutate the weights of the conv2d and linear layers. For each weights if the
    threshold ``p`` is passed add gaussian noise with std = 0.1.

    Args:
        oneshot (OneShotNet):
            The net to mutate.
        p (float):
            The mutation probability.

    Returns:
        OneShotNet:
            The mutated net.
    """
    # Helper to mutate only Linear and Conv2d.
    def maybe_mutate(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)) and torch.rand(1).item() < p:
            with torch.no_grad():
                m.weight += torch.rand_like(m.weight) * 0.1

    first = deepcopy(oneshot.first)
    for m in first.modules():
        maybe_mutate(m)

    blocks: list[BaseOp] = []
    for block in oneshot.blocks:
        block = deepcopy(block)

        for m in block.modules():
            maybe_mutate(m)

        blocks.append(block)

    last = deepcopy(oneshot.last)
    for m in last.modules():
        maybe_mutate(m)

    classifier = deepcopy(oneshot.classifier)
    for m in classifier.modules():
        maybe_mutate(m)

    return OneShotNet(
        first,
        blocks,
        last,
        oneshot.pool,
        classifier,
        oneshot._block_lengths
    )
