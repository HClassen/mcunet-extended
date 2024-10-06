from enum import IntEnum
from typing_extensions import Doc
from abc import abstractmethod, ABC
from collections.abc import Iterator
from typing import Any, Annotated, Final
from dataclasses import dataclass, asdict

import numpy as np

from ..utils import make_divisible
from ..mobilenet import LAYER_SETTINGS, FIRST_CONV_CHANNELS, LAST_CONV_CHANNELS


"""
This subproject contains the general implemenation of a search space and relevant
parameters.

Specifig implementation of a search space need to sub-class `SearchSpace` and
implement the two methods `sample_model` and `max_models`.

The general parameterization of the seach spaces are a combiantion of MnasNet [1]
and MCUNet [2].

Also included are Pytorch moduls implementing the different
operations with the different parameters and function to convert a sampled
`Model` to a complete Pytorch module.

[1]: https://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper
[2]: https://arxiv.org/pdf/2007.10319
"""


__all__ = [
    "ConvOp",
    "CONV_CHOICES", "KERNEL_CHOICES", "SE_CHOICES", "SKIP_CHOICES",
    "LAYER_CHOICES", "CHANNEL_CHOICES",
    "Block", "sample_block",
    "Model", "sample_model", "uniform_model",
    "SearchSpace"
]


class ConvOp(IntEnum):
    """
    The three different types of convolution operation available in the search
    space.
    """
    CONV2D = 0
    DWCONV2D = 1
    BDWRCONV2D = 2

    def __str__(self):
        return ConvOp._STR_VALUES[self.value]

ConvOp._STR_VALUES = {
    ConvOp.CONV2D: "conv2d",
    ConvOp.DWCONV2D: "dwconv2d",
    ConvOp.BDWRCONV2D: "bdwrconv2d"
}


CONV_CHOICES: Annotated[
    Final[list[ConvOp]],
    Doc(
        """
        The possible choices for the convolution. Taken from the MnasNet paper.
        """
    )
] = [ConvOp.CONV2D, ConvOp.DWCONV2D, ConvOp.BDWRCONV2D]

KERNEL_CHOICES: Annotated[
    Final[list[int]],
    Doc(
        """
        The possible choices for the size of the convolution kernel. Taken from
        the MnasNet paper. 7 was added as it appears in the MCUNet models.
        """
    )
] = [3, 5, 7]

SE_CHOICES: Annotated[
    Final[list[float]],
    Doc(
        """
        The possible choices for the squeeze-and-excitation. Taken from the
        MnasNet paper.
        """
    )
] = [0, 0.25]

SHORTCUT_CHOICES: Annotated[
    Final[list[bool]],
    Doc(
        """
        The possible choices to use a shortcut in a layer. In the MnasNet
        paper this is called skip. Besides sjortcut hey also include "pooling".
        The actual implementation does not use skip in this form at all. They
        only use shortcuts. So provide only two options here: don't use a
        shortcut (`False`) or use one (`True`).
        """
    )
] = [False, True]

LAYER_CHOICES: Annotated[
    Final[list[int]],
    Doc(
        """
        The possible choices of layers per block. Taken from the MCUNet paper
        ("1" additionally added).
        """
    )
] = [1, 2, 3, 4]

CHANNEL_CHOICES: Annotated[
    Final[list[float]],
    Doc(
        """
        The possible choices to apply to the channels per layers of the
        MobileNetV2 settings. Taken from the MnasNet paper.
        """
    )
] = [0.75, 1.0, 1.25]

EXPANSION_CHOICES: Annotated[
    Final[list[int]],
    Doc(
        """
        The possible choices for the expansion ration of a bottleneck
        depthwise-separable convolution with residual operation. Taken from the
        MCUNet paper.
        """
    )
] = [3, 4, 6]

_width_step: Final[float] = 0.1
WIDTH_CHOICES: Annotated[
    Final[list[float]],
    Doc(
        """
        The selected choices for the width multiplier from MCUNet.
        """
    )
] = [0.2 + i * _width_step for i in range(9)]

_resolution_step: Final[int] = 16
RESOLUTION_CHOICES: Annotated[
    Final[list[int]],
    Doc(
        """
        The selected choices for the resolution from MCUNet.
        """
    )
] = [48 + i * _resolution_step for i in range(12)]


def configurations() -> Iterator[tuple[float, int]]:
    """
    Creates all 108 combinations of alpha (width multiplier) and rho (resolution)
    as per the MCUNet paper.

    Returns:
        Iterator[tuple[float, int]]:
            An iterator of all combinations.
    """
    for with_mult in WIDTH_CHOICES:
        for resolution in RESOLUTION_CHOICES:
            yield (with_mult, resolution)


@dataclass(frozen=True)
class Layer():
    op: ConvOp

    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]

    expansion_ratio: int
    shortcut: bool

    se_ratio: float


@dataclass(frozen=True)
class Block():
    layers: list[Layer]


@dataclass
class Model():
    blocks: list[Block]
    last_channels: int

    def to_dict(self) -> dict[str, Any]:
        """
        Converts a `Model` to a `dict`.

        Returns:
            dict[str, Any]:
                A `dict` containing the content of this model.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'Model':
        """
        Converts a `dict` to a `Model`.

        Args:
            config (dict[str, Any]):
                The `dict` containing the content of a model to be loaded.

        Returns:
            Model:
                The model constructed from the `dict`.
        """

        blocks: list[Block] = [
            Block([
                Layer(**layer_config) for layer_config in block_config["layers"]
            ]) for block_config in config["blocks"]
        ]

        return cls(blocks, config["last_channels"])


def uniform_layers(
    op: ConvOp,
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    expansion_ratio: int,
    shortcut: bool,
    se_ratio: float,
    depth: int
) -> list[Layer]:
    """
    Create `depth` many layers with the same configuration.

    An exception is the first layer. Its configuration is different from the
    following layers. It translates from `in_channels` to `out_channels` and
    has a stride of `stride` applied.

    The reamining layers have the same number of input and output channels
    defined in `out_channels` and a stride of 1.

    Args:
        op (ConvOp):
            The convolution operation of all layers.
        in_channels (int):
            The input channels for tis block of layers. Only applied to the
            first layer.
        out_channels (int):
            The output channels of all layers as well as the input channels for
            all layers after the first.
        kernel_size (int, tuple[int, int]):
            The kernel size for all layers.
        stride (int, tuple[int, int]):
            The stride for the first layer.
        expansion_ratio (int):
            The expansion ration for all layers if `op` is `ConvOp.BDWRCONV2D`.
        shortcut (bool):
            The use of a shortcut for all layers.
        se_ratio (float):
            The squeeze-and-excitation ratio for all layers.
        depth (int):
            The total amount of layers to be created.

    Returns:
        list[BaseOp]:
            A list containting `depth` many layers.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride)

    return [
        Layer(
            op,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            expansion_ratio,
            shortcut,
            se_ratio
        )
    ] + [
        Layer(
            op,
            out_channels,
            out_channels,
            kernel_size,
            (1, 1),
            expansion_ratio,
            shortcut,
            se_ratio
        ) for _ in range(depth - 1)
    ]


def uniform_model(
    op: ConvOp,
    kernel_size: int | tuple[int, int],
    se_ratio: float,
    shortcut: bool,
    depth: int,
    channel_mult: float,
    expansion_ratio: int,
    width_mult: float = 1.0
) -> Model:
    """
    Sample a `Model` from the search space, where every `Block` has the same
    configuration.

    Args:
        op (ConvOp):
            The convolution operation for all blocks to use for all layers.
        kernel_size (int, tuple[int, int]):
            The kernel size for the convolution operation for all layers.
        se_ratio (float):
            The squeeze-and-excitation ratio for all layers.
        shortcut (bool):
            The use of a shortcut for all layers.
        depth (int):
            The amount of layers for all blocks.
        channel_mult (float):
            The additionel multiplier for th e amount of channels.
        expansion_ratio (int):
            The expansion ratio to use with mobile inverted bottleneck blocks.
        width_mult (float):
            Modifies the input and output channels of the hidden layers.

    Returns:
        Model:
            The newly created model.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    blocks: list[Block] = []

    if op != ConvOp.BDWRCONV2D:
        expansion_ratio = 1

    round_nearest = 8
    in_channels = make_divisible(
        FIRST_CONV_CHANNELS * width_mult, round_nearest
    )

    for settings in LAYER_SETTINGS:
        out_channels = make_divisible(
            settings[1] * channel_mult * width_mult, round_nearest
        )

        layers = uniform_layers(
            op,
            in_channels,
            out_channels,
            kernel_size,
            (settings[3], settings[3]),
            expansion_ratio,
            shortcut,
            se_ratio,
            depth
        )

        blocks.append(Block(layers))

        in_channels = out_channels

    last_channels = make_divisible(
        LAST_CONV_CHANNELS * max(1.0, width_mult), round_nearest
    )

    return Model(blocks, last_channels)


class SearchSpace(ABC):
    """
    Represents a search space with a given width multiplier and resolution. It
    implements the `Iterator` pattern to randomly sample models.
    """
    width_mult: float
    resolution: int

    _rng: np.random.Generator | None

    def __init__(self, width_mult: float, resolution: int) -> None:
        self.width_mult = width_mult
        self.resolution = resolution

        self._rng = None

    def oneshot(self) -> Model:
        """
        Samples one model from the search space. Meant to be used in a oneshot
        way, where only one model is needed.

        Returns:
            Model:
                A randomly sampled model from the MnasNet searchspace.
        """
        rng = np.random.default_rng()
        return self.sample_model(rng)

    def __iter__(self) -> 'SearchSpace':
        if self._rng is None:
            self._rng = np.random.default_rng()

        return self

    def __next__(self) -> Model:
        if self._rng is None:
            raise StopIteration()

        return self.sample_model(self._rng)

    @abstractmethod
    def sample_model(self, rng: np.random.Generator) -> Model:
        """
        Sample a single model from the search space.

        Args:
            rng (numpy.random.Generator):
                A random number generator.

        Returns:
            Model:
                The sampled model.
        """
        pass

    @abstractmethod
    def max_models(self) -> tuple[Model, ...]:
        """
        Build the maximum model(s) in this search space.

        Returns:
            tuple[Model, ...]:
                All the maximum models.
        """
        pass
