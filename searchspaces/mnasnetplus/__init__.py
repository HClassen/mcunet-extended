from typing import cast
from collections.abc import Iterator, Iterable, Callable

import torch
import torch.nn as nn

from torchvision.ops import Conv2dNormActivation

import numpy as np

from mcunet.tinynas.searchspace import (
    CONV_CHOICES,
    KERNEL_CHOICES,
    SE_CHOICES,
    SKIP_CHOICES,
    LAYER_CHOICES,
    CHANNEL_CHOICES,
    EXPANSION_CHOICES,
    uniform_layers,
    uniform_model,
    ConvOp,
    SkipOp,
    Block,
    Model,
    SearchSpace
)
from mcunet.tinynas.searchspace.layers import (
    LayerName,
    Conv2dOp,
    DWConv2dOp,
    BDWRConv2dOp
)
from mcunet.tinynas.oneshot import ChoiceBlock, ChoiceBlocksMaker
from mcunet.tinynas.oneshot.customize import (
    ChoicesMaker,
    ChoiceLayerMaker,
    LastChoiceLayer,
    WarmUpSetter
)
from mcunet.tinynas.oneshot.share import (
    share_conv2d,
    share_batchnorm2d,
    ParameterSharer
)
from mcunet.tinynas.oneshot.helper import has_norm
from mcunet.tinynas.mobilenet import (
    FIRST_CONV_CHANNELS,
    LAYER_SETTINGS,
    LAST_CONV_CHANNELS,
    MobileSkeletonNet
)
from mcunet.tinynas.utils import make_divisible


__all__ = [
    "BaseChoiceLayer",
    "BaseConv2dChoiceOp",
    "BaseDWConv2dChoiceOp",
    "BaseBDWRConv2dChoiceOp",
    "ChoiceOpBuilder",
    "ChoiceBlock",
    "SuperChoiceBlock",
    "BaseLastChoiceOp",
    "MnasNetPlus"
]


class LastLayerSharer(ParameterSharer):
    conv2d_weight: nn.Parameter
    norm_weight: nn.Parameter
    norm_bias: nn.Parameter

    def set_shared(self, module: Conv2dOp) -> None:
        share_conv2d(module[LayerName.CONV2D][0], self.conv2d_weight, None)

        if has_norm(module[LayerName.CONV2D]):
            share_batchnorm2d(
                module[LayerName.CONV2D][1], self.norm_weight, self.norm_bias,
                None, None, None
            )

    def unset_shared(self, module: Conv2dOp) -> None:
        module[LayerName.CONV2D][0].weight = None

        if has_norm(module[LayerName.CONV2D]):
            module[LayerName.CONV2D][1].weight = None
            module[LayerName.CONV2D][1].bias = None

    def _weight_initialization(self) -> None:
        nn.init.kaiming_normal_(self.conv2d_weight, mode="fan_out")
        nn.init.ones_(self.norm_weight)
        nn.init.zeros_(self.norm_bias)

    def make_shared(self, **kwargs) -> None:
        out_channels = kwargs["out_channels"]
        in_channels = kwargs["in_channels"]

        self.conv2d_weight = nn.Parameter(
            torch.Tensor(
                out_channels,
                in_channels,
                1, 1
            )
        )
        self.norm_weight = nn.Parameter(torch.Tensor(out_channels))
        self.norm_bias = nn.Parameter(torch.Tensor(out_channels))

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


def _permutations() -> Iterator[tuple[int, float, SkipOp]]:
    for k in KERNEL_CHOICES:
        for se in SE_CHOICES:
            for skip in SKIP_CHOICES:
                yield (k, se, skip)


class Conv2dChoicesMaker(ChoicesMaker):
    def make(
        self,
        range_in_channels: Iterable[int],
        range_out_channels: Iterable[int],
        range_expansion_ratio: Iterable[int],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None,
        activation_layer: Callable[..., nn.Module] | None
    ) -> list[Conv2dOp]:
        choices: list[Conv2dOp] = []

        for i in range_in_channels:
            for j in range_out_channels:
                for k, se, skip in _permutations():
                    choices.append(Conv2dOp(
                        i, j, se, skip, k, stride, norm_layer, activation_layer
                    ))

        return choices


class DWConv2dChoicesMaker(ChoicesMaker):
    def make(
        self,
        range_in_channels: Iterable[int],
        range_out_channels: Iterable[int],
        range_expansion_ratio: Iterable[int],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> list[DWConv2dOp]:
        choices: list[DWConv2dOp] = []

        for i in range_in_channels:
            for j in range_out_channels:
                for k, se, skip in _permutations():
                    choices.append(DWConv2dOp(
                        i, j, se, skip, k, stride, norm_layer, activation_layer
                    ))

        return choices


class BDWRConv2dChoicesMaker(ChoicesMaker):
    def make(
        self,
        range_in_channels: Iterable[int],
        range_out_channels: Iterable[int],
        range_expansion_ratios: Iterable[int],
        stride: int,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> list[BDWRConv2dOp]:
        choices: list[BDWRConv2dOp] = []

        for i in range_in_channels:
            for j in range_out_channels:
                for e in range_expansion_ratios:
                    for k, se, skip in _permutations():
                        choices.append(BDWRConv2dOp(
                            i, j, e, se, skip, k, stride, norm_layer, activation_layer
                        ))

        return choices


class SuperChoiceBlock(nn.Module):
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
        super().__init__()

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


class ChoiceBlocksMaker(ChoiceBlocksMaker):
    _conv2d_maker: ChoiceLayerMaker
    _dwconv2d_maker: ChoiceLayerMaker
    _bdwrconv2d_maker: ChoiceLayerMaker

    def __init__(
        self,
        conv2d_maker: ChoiceLayerMaker,
        dwconv2d_maker: ChoiceLayerMaker,
        bdwrconv2d_maker: ChoiceLayerMaker
    ) -> None:
        self._conv2d_maker = conv2d_maker
        self._dwconv2d_maker = dwconv2d_maker
        self._bdwrconv2d_maker = bdwrconv2d_maker

    def make(
        self,
        in_channels: tuple[int, ...],
        width_mult: float,
        round_nearest: int,
        norm_layer: Callable[..., nn.Module] | None,
        activation_layer: Callable[..., nn.Module] | None
    ) -> tuple[list[ChoiceBlock], tuple[int, ...]]:
        blocks: list[SuperChoiceBlock] = []

        depth = max(LAYER_CHOICES)
        for _, c, _, s in LAYER_SETTINGS:
            out_channels = set([
                make_divisible(c * choice * width_mult, round_nearest)
                for choice in CHANNEL_CHOICES
            ])

            conv2d = ChoiceBlock(
                self._conv2d_maker,
                in_channels,
                out_channels,
                [1],
                depth,
                s,
                norm_layer,
                activation_layer
            )

            dwconv2d = ChoiceBlock(
                self._dwconv2d_maker,
                in_channels,
                out_channels,
                [1],
                depth,
                s,
                norm_layer,
                activation_layer
            )

            bdwrconv2d = ChoiceBlock(
                self._bdwrconv2d_maker,
                in_channels,
                out_channels,
                EXPANSION_CHOICES,
                depth,
                s,
                norm_layer,
                activation_layer
            )

            superblock = SuperChoiceBlock(conv2d, dwconv2d, bdwrconv2d)
            blocks.append(superblock)

            in_channels = out_channels

        return blocks, in_channels


class CommonWarmUpSetter(WarmUpSetter):
    _first_parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    _last_parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    _classifier_parameters: tuple[torch.Tensor, torch.Tensor]

    def before(self, supernet: MobileSkeletonNet, device) -> None:
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

    def after(self, supernet: MobileSkeletonNet) -> None:
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

    choice = rng.integers(0, len(SKIP_CHOICES))
    skip_op = SKIP_CHOICES[choice]

    expansion_ratio = EXPANSION_CHOICES[
        rng.integers(0, len(EXPANSION_CHOICES))
    ] if op == ConvOp.BDWRCONV2D else 1

    layers = uniform_layers(
        op,
        in_channels,
        out_channels,
        first_stride,
        expansion_ratio,
        kernel_size,
        se_ratio,
        skip_op,
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

    for _, c, _, s in LAYER_SETTINGS:
        depth = LAYER_CHOICES[rng.integers(0, len(LAYER_CHOICES))]

        choice = CHANNEL_CHOICES[rng.integers(0, len(CHANNEL_CHOICES))]
        out_channels = make_divisible(
            c * choice * width_mult, round_nearest
        )

        blocks.append(
            _sample_block(rng, depth, in_channels, out_channels, s)
        )

        in_channels = out_channels

    last_channels = make_divisible(
        LAST_CONV_CHANNELS * max(1.0, width_mult), round_nearest
    )

    return Model(blocks, last_channels)


class MnasNetPlus(SearchSpace):
    _max_model_with_se: bool

    def __init__(
        self, width_mult: float, resolution: int, max_model_with_se: bool
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
                SkipOp.NOSKIP,
                max(LAYER_CHOICES),
                max(CHANNEL_CHOICES),
                max(EXPANSION_CHOICES),
                self.width_mult
            )
            for op in (ConvOp.CONV2D, ConvOp.DWCONV2D, ConvOp.BDWRCONV2D)
        )
