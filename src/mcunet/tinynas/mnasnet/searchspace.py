import json
from typing import Any
from enum import IntEnum
from operator import itemgetter
from dataclasses import dataclass, field, asdict

import numpy as np

from ..utils import make_divisible

class ConvOp(IntEnum):
    """
    The possible choices for the convolution. Taken from the MnasNet paper.
    """
    CONV2D = 0
    DWCONV2D = 1
    MBCONV2d = 2

    def __str__(self):
        return ConvOp._STR_VALUES[self.value]

ConvOp._STR_VALUES = {
    ConvOp.CONV2D: "conv2d",
    ConvOp.DWCONV2D: "dwconv2d",
    ConvOp.MBCONV2d: "inverted_bottleneck_conv2d"
}

conv_choices: list[ConvOp] = [ConvOp.CONV2D, ConvOp.DWCONV2D, ConvOp.MBCONV2d]

# The possible choices for the size of the convolution kernel. Taken from from
# the MnasNet paper. 7 was added as it appears in the MCUNet models.
kernel_choices: list[int] = [3, 5, 7]

# The possible choices for the squeeze-and-excitation. Taken from the MnasNet
# paper.
se_choices: list[float] = [0, 0.25]

class SkipOp(IntEnum):
    """
    The possible choices for the skip op on the first layer. The MnasNet paper
    only says "pooling". The actual implementation does not use skip in this
    form at all. They only use residual skip.

    While this enum provides four choices for the skip operation, only NOSKIP
    and IDENTITY are used. The POOLING_AVG and POOLING_MAX values are only kept
    for future clarification.
    """
    NOSKIP = 0
    IDENTITY = 1
    POOLING_AVG = 2
    POOLING_MAX = 3

    def __str__(self):
        return SkipOp._STR_VALUES[self.value]

SkipOp._STR_VALUES = {
    SkipOp.NOSKIP: "noskip",
    SkipOp.IDENTITY: "identity",
    SkipOp.POOLING_AVG: "pooling_avg",
    SkipOp.POOLING_MAX: "pooling_max"
}

skip_choices: list[SkipOp] = [SkipOp.NOSKIP, SkipOp.IDENTITY]

# The possible choices to apply to the number of layers of the MobileNetV2
# settings. Taken from the MnasNet paper.
layer_choices: list[int] = [0, 1, -1]

# The possible choices to apply to the channels per layers of the MobileNetV2
# settings. Taken from the MnasNet paper.
channel_choices: list[float] = [0.75, 1.0, 1.25]

# MobileNetV2 settings for bottleneck layers. MnasNet makes use of them.
# Copied from
# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
layer_settings: list[list[int]] = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
]


@dataclass(frozen=True)
class Block():
    n_layers: int
    in_channels: int
    out_channels: int
    first_stride: int  # Is only applied to the first layer of the block (see MobilenetV2).
    expand_ratio: int  # Only relevant when conv_op is ConvOp.BOTTLENECK

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
    expand_ratio: int
) -> Block:
        """
        Randomly amples a single instance of a ``mcunet.tinynas.mnasnet.searchspace.Block``
        according to the MnasNet search space.

        Args:
            rng (np.random.Generator):
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
            expand_ratio (int):
                Expand ratio for ``.layers.MBConv2dOp`` if choosen.

        Returns:
            mcunet.tinynas.mnasnet.searchspace.Block:
                The newly created block.
        """
        choice = rng.integers(0, len(conv_choices))
        conv_op = conv_choices[choice]

        choice = rng.integers(0, len(kernel_choices))
        kernel_size = kernel_choices[choice]

        choice = rng.integers(0, len(se_choices))
        se_ratio = se_choices[choice]

        choice = rng.integers(0, len(skip_choices))
        skip_op = skip_choices[choice]

        return Block(
            n_layers, in_channels, out_channels, first_stride, expand_ratio,
            conv_op, kernel_size, se_ratio, skip_op
        )


@dataclass
class Model():
    n_classes: int
    width_mult: float = field(default=1.0)  # alpha parameter of MobileNet
    resolution: int = field(default=224)  # rho parameter of Mobilenet
    dropout: float = 0.2

    # Always use 7 blocks. In the final model these 7 blocks replace the 7
    # bottleneck blocks in the MobileNetV2 architecture. I. e. a conv2d is
    # added before them and conv2d, avg pool + flatten and classifier is added
    # after them. They are not part of this model config, as they are same for
    # every generated model.
    blocks: list[Block] = field(init=False)
    last_channels: int = field(init=False)


def sample_model(
        rng: np.random.Generator,
        n_classes: int,
        width_mult: float = 1.0,
        resolution: int = 224,
        dropout: float = 0.2,
    ) -> Model:
        """
        Randomly samples a single instance of a ``x.searchspace.Model`` according
        to the MnasNet search space. The MnasNet search space makes use of the
        MobileNetV2 layer settings and input and output channels. These settings
        are used here.

        Args:
            rng (np.random.Generator):
                A random number generator to uniformly distribute the models
                parameters.
            n_classes (int):
                The number of classes the networks use.
            width_mult (float):
                Modifies the input and output channels of the hidden layers.
            resolution (int):
                The resolution of the input image.
            dropout (float):
                Dropout rate for the classifier of MobileNetV2.

        Returns:
            mcunet.tinynas.mnasnet.searchspace.Model:
                The newly created model.
        """
        model = Model(n_classes, width_mult, resolution, dropout)
        model.blocks = []

        round_nearest = 8

        # As per MobileNetV2 architecture. Copied from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        in_channels = make_divisible(32 * width_mult, round_nearest)

        for t, c, n, s in layer_settings:
            choice = layer_choices[rng.integers(0, len(layer_choices))]
            n_layers = n + choice

            if n_layers == 0:
                continue

            choice = channel_choices[rng.integers(0, len(channel_choices))]
            out_channels = make_divisible(
                c * choice * width_mult, round_nearest
            )

            model.blocks.append(
                sample_block(rng, n_layers, in_channels, out_channels, s, t)
            )

            in_channels = out_channels

        model.last_channels = make_divisible(
            1280 * max(1.0, width_mult), round_nearest
        )

        return model


def to_json(model: Model) -> str:
    """
    Converts the ``model`` to JSON.

    Args:
        model (mcunet.tinynas.mnasnet.searchspace.Model):
            The model to be serialized.

    Returns:
        str:
            A string containing the model as a JSON.
    """
    return json.dumps(asdict(model))


def from_json(config: str | dict[str, Any]) -> Model:
    """
    Converts JSON to a ``mcunet.tinynas.mnasnet.searchspace.Model``.

    Args:
        config (str, dict[str, Any]):
            Either a string containing the configuration in JSON format or a
            ``dict``.

    Returns:
        mcunet.tinynas.mnasnet.searchspace.Model:
            The model constructed from the JSON configuration.
    """
    if isinstance(config, str):
        config = json.loads(config)

    n_classes, width_mult, resolution, dropout = \
        itemgetter("n_classes", "width_mult", "resolution", "dropout")(config)

    model = Model(n_classes, width_mult, resolution, dropout)

    blocks: list[Block] = []
    for block_config in config["blocks"]:
        n_layers, in_channels, out_channels, first_stride, expand_ratio = \
            itemgetter(
                "n_layers",
                "in_channels",
                "out_channels",
                "first_stride",
                "expand_ratio"
            )(block_config)

        block = Block(
            n_layers, in_channels, out_channels, first_stride, expand_ratio
        )

        block.conv_op, \
        block.kernel_size, \
        block.se_ratio, \
        block.skip_op, \
            itemgetter(
                "conv_op", "kernel_size", "se_ratio", "skip_op"
            )(block_config)

        blocks.append(block)

    model.blocks = blocks
    model.last_channels = config["last_channels"]

    return model
