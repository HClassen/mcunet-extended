from enum import IntEnum
from typing import Any, Final
from operator import itemgetter
from dataclasses import dataclass, field, asdict

import numpy as np

from ..utils import make_divisible
from ..mobilenet import LAYER_SETTINGS,  FIRST_CONV_CHANNELS, LAST_CONV_CHANNELS


__all__ = [
    "ConvOp", "CONV_CHOICES", "KERNEL_CHOICES", "SE_CHOICES",
    "SkipOp", "SKIP_CHOICES", "LAYER_CHOICES", "CHANNEL_CHOICES",
    "Block", "sample_block", "Model", "sample_model", "uniform_model"
]


class ConvOp(IntEnum):
    """
    The possible choices for the convolution. Taken from the MnasNet paper.
    """
    CONV2D = 0
    DWCONV2D = 1
    MBCONV2D = 2

    def __str__(self):
        return ConvOp._STR_VALUES[self.value]

ConvOp._STR_VALUES = {
    ConvOp.CONV2D: "conv2d",
    ConvOp.DWCONV2D: "dwconv2d",
    ConvOp.MBCONV2D: "mbconv2d"
}

CONV_CHOICES: Final[list[ConvOp]] = [
    ConvOp.CONV2D, ConvOp.DWCONV2D, ConvOp.MBCONV2D
]

# The possible choices for the size of the convolution kernel. Taken from from
# the MnasNet paper. 7 was added as it appears in the MCUNet models.
KERNEL_CHOICES: Final[list[int]] = [3, 5, 7]

# The possible choices for the squeeze-and-excitation. Taken from the MnasNet
# paper.
SE_CHOICES: Final[list[float]] = [0, 0.25]

class SkipOp(IntEnum):
    """
    The possible choices for the skip op on the first layer. The MnasNet paper
    only says "pooling". The actual implementation does not use skip in this
    form at all. They only use residual skip. So provide only two options here:
    don't perform a skip operation or use identity residual skip.
    """
    NOSKIP = 0
    IDENTITY = 1

    def __str__(self):
        return SkipOp._STR_VALUES[self.value]

SkipOp._STR_VALUES = {
    SkipOp.NOSKIP: "noskip",
    SkipOp.IDENTITY: "identity"
}

SKIP_CHOICES: Final[list[SkipOp]] = [SkipOp.NOSKIP, SkipOp.IDENTITY]

# The possible choices of layers per block. Taken from the MCUNet paper
# ("1" additionally added).
LAYER_CHOICES: Final[list[int]] = [1, 2, 3, 4]

# The possible choices to apply to the channels per layers of the MobileNetV2
# settings. Taken from the MnasNet paper.
CHANNEL_CHOICES: Final[list[float]] = [0.75, 1.0, 1.25]

# The possible choices for the expansion ration of a mobile inverted bottleneck
# block. Taken from the MCUNet paper.
EXPANSION_CHOICES: Final[list[int]] = [3, 4, 6]


@dataclass(frozen=True)
class Block():
    n_layers: int
    in_channels: int
    out_channels: int
    first_stride: int  # Is only applied to the first layer of the block (see MobilenetV2).
    expansion_ratio: int  # Only relevant when conv_op is ConvOp.MBCONV2D

    # Configuration of the layers.
    conv_op: ConvOp
    kernel_size: int
    se_ratio: float
    skip_op: SkipOp


def sample_block(
    rng: np.random.Generator,
    n_layers: int,
    in_channels: int,
    out_channels: int,
    first_stride: int,
    expansion_ratio: int
) -> Block:
    """
    Randomly amples a single instance of a ``Block`` according to the
    MnasNet search space.

    Args:
        rng (Generator):
            A random number generator to uniformly distribute the blocks
            parameters.
        n_layers (int):
            The number of layers.
        in_channels (int):
            The number of in channels.
        out_channels (int):
            The number of out channels.
        first_stride (int):
            The width of the stride for the first layer.
        expansion_ratio (int):
            Expand ratio for a mobile inverted bottleneck if choosen.

    Returns:
        Block:
            The newly created block.
    """
    choice = rng.integers(0, len(CONV_CHOICES))
    conv_op = CONV_CHOICES[choice]

    choice = rng.integers(0, len(KERNEL_CHOICES))
    kernel_size = KERNEL_CHOICES[choice]

    choice = rng.integers(0, len(SE_CHOICES))
    se_ratio = SE_CHOICES[choice]

    choice = rng.integers(0, len(SKIP_CHOICES))
    skip_op = SKIP_CHOICES[choice]

    return Block(
        n_layers, in_channels, out_channels, first_stride, expansion_ratio,
        conv_op, kernel_size, se_ratio, skip_op
    )


@dataclass
class Model():
    # Always use 7 blocks. In the final model these 7 blocks replace the 7
    # bottleneck blocks in the MobileNetV2 architecture. I. e. a conv2d is
    # added before them and conv2d, avg pool + flatten and classifier is added
    # after them. They are not part of this model config, as they are same for
    # every generated model.
    blocks: list[Block]
    last_channels: int

    def to_dict(self) -> dict[str, Any]:
        """
        Converts a ``Model`` to a ``dict``.

        Returns:
            dict[str, Any]:
                A ``dict`` containing the content of this model.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'Model':
        """
        Converts a ``dict`` to a ``Model``.

        Args:
            config (dict[str, Any]):
                The ``dict`` containing the content of a model to be loaded.

        Returns:
            Model:
                The model constructed from the ``dict``.
        """

        blocks: list[Block] = []
        for block_config in config["blocks"]:
            n_layers, in_channels, out_channels, first_stride, expansion_ratio = \
                itemgetter(
                    "n_layers",
                    "in_channels",
                    "out_channels",
                    "first_stride",
                    "expansion_ratio"
                )(block_config)

            conv_op, kernel_size, se_ratio, skip_op = \
                itemgetter(
                    "conv_op", "kernel_size", "se_ratio", "skip_op"
                )(block_config)

            block = Block(
                n_layers,
                in_channels,
                out_channels,
                first_stride,
                expansion_ratio,
                conv_op,
                kernel_size,
                se_ratio,
                skip_op
            )

            blocks.append(block)

        return cls(blocks, config["last_channels"])


def sample_model(rng: np.random.Generator, width_mult: float = 1.0) -> Model:
    """
    Randomly samples a single instance of a ``Model`` according
    to the MnasNet search space. The MnasNet search space makes use of the
    MobileNetV2 layer settings and input and output channels. These settings
    are used here.

    Args:
        rng (Generator):
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
        n_layers = LAYER_CHOICES[rng.integers(0, len(LAYER_CHOICES))]

        choice = CHANNEL_CHOICES[rng.integers(0, len(CHANNEL_CHOICES))]
        out_channels = make_divisible(
            c * choice * width_mult, round_nearest
        )

        expansion_ratio = EXPANSION_CHOICES[
            rng.integers(0, len(EXPANSION_CHOICES))
        ]

        blocks.append(
            sample_block(
                rng,
                n_layers,
                in_channels,
                out_channels,
                s,
                expansion_ratio
            )
        )

        in_channels = out_channels

    last_channels = make_divisible(
        LAST_CONV_CHANNELS * max(1.0, width_mult), round_nearest
    )

    return Model(blocks, last_channels)


def uniform_model(
    conv_op: ConvOp,
    kernel_size: int,
    se_ratio: float,
    skip_op: SkipOp,
    n_layers: int,
    channel_mult: float,
    expansion_ratio: int,
    width_mult: float = 1.0
) -> Model:
    """
    Sample a ``Model`` from the search space, where every ``Block`` has the same
    configuration.

    Args:
        conv_op (ConvOp):
            The convolution operation for all blocks to use.
        kernel_size (int):
            The kernel size for the convolution operation.
        se_ratio (float):
            The squeeze-and-excitation ratio.
        skip_op (SkipOp):
            The skip operation per block.
        n_layers (int):
            The amount of layers foar all blocks.
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
    blocks: list[Block] = []

    round_nearest = 8
    in_channels = make_divisible(
        FIRST_CONV_CHANNELS * width_mult, round_nearest
    )

    for _, c, _, s in LAYER_SETTINGS:
        out_channels = make_divisible(
            c * channel_mult * width_mult, round_nearest
        )

        blocks.append(
            Block(
                n_layers,
                in_channels,
                out_channels,
                s,
                expansion_ratio,
                conv_op,
                kernel_size,
                se_ratio,
                skip_op
            )
        )

        in_channels = out_channels

    last_channels = make_divisible(
        LAST_CONV_CHANNELS * max(1.0, width_mult), round_nearest
    )

    return Model(blocks, last_channels)
