import json
from typing import Any
from enum import StrEnum
from operator import itemgetter
from collections.abc import Iterator
from dataclasses import dataclass, field, asdict

import numpy as np

import matplotlib.pyplot as plt

class ConvOp(StrEnum):
    """
    The possible choices for the convolution. Taken from the MnasNet paper.
    """
    CONV2D = "conv2d"
    DWCONV2D = "dwconv2d"
    BOTTLENECK = "inverted_bottleneck_conv2d"

_conv_choices: list[ConvOp] = [ConvOp.CONV2D, ConvOp.DWCONV2D, ConvOp.BOTTLENECK]

#The possible choices for the size of the convolution kernel. Taken from from
# the MnasNet paper. SEVEN was added as it appears in the MCUNet models.
_kernel_choices: list[int] = [3, 5, 7]

# The possible choices for the squeeze-and-excitation. Taken from the MnasNet
# paper.
_se_choices: list[float] = [0, 0.25]

class SkipOp(StrEnum):
    """
    The possible choices for the skip op on the first layer. The MnasNet paper
    only says "pooling", here it is represented by avg and max pooling. For all
    layers which are not the first layer of a block, identity residual is used.
    """
    NOSKIP = "noskip"
    IDENTITY = "identity"
    POOLING_AVG = "pooling_avg"
    POOLING_MAX = "pooling_max"

_skip_choices: list[SkipOp] = [SkipOp.POOLING_AVG, SkipOp.POOLING_MAX]

# The possible choices to apply to the number of layers of the MobileNetV2
# settings. Taken from the MnasNet paper.
_layer_choices: list[int] = [0, 1, -1]

# The possible choices to apply to the channels per layers of the MobileNetV2
# settings. Taken from the MnasNet paper.
_channel_choices: list[float] = [0.75, 1.0, 1.25]

# Ensures that all layers have a channel number that is divisible by 8.
# Copied from
# https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py
def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@dataclass
class Block():
    n_layers: int
    in_channels: int
    out_channels: int
    first_stride: int  # Is only applied to the first layer of the block (see MobilenetV2).
    expand_ratio: int  # Only relevant when conv_op is ConvOp.BOTTLENECK

    # Configuration of the layers.
    conv_op: ConvOp = field(init=False)
    kernel_size: int = field(init=False)
    se_ratio: float = field(init=False)

    # First layer is special regarding the skip op.
    first_skip: SkipOp = field(init=False)
    rest_skip: SkipOp = field(init=False)

    def sample(self, rng: np.random.Generator) -> 'Block':
        """
        Samples a single instance of a ``x.searchspace.Block`` according to the
        MnasNet search space.

        Args:
            rng (np.random.Generator): A random number generator to uniformly distribute the blocks parameters.

        Returns:
            x.searchspace.Block: Itself to allow for method chaining.
        """
        self.conv_op = _conv_choices[rng.integers(0, len(_conv_choices))]
        self.kernel_size = _kernel_choices[rng.integers(0, len(_kernel_choices))]
        self.se_ratio = _se_choices[rng.integers(0, len(_se_choices))]

        # Make booleaness of skip op explicit.
        skip_op = bool(_skip_choices[rng.integers(0, 2)])

        # For skip op differantiate between first layer and all others. Only in
        # the first layer of block is a stride != 1 possible. So if a skip op
        # should be used, the residual must also change it dimensionality. Use
        # pooling for that and choose between max and avg.
        if skip_op:
            self.first_skip = _skip_choices[rng.integers(0, len(_skip_choices))]
            self.rest_skip = SkipOp.IDENTITY
        else:
            self.first_skip = SkipOp.NOSKIP
            self.rest_skip = SkipOp.NOSKIP

        return self


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

    def sample(self, rng: np.random.Generator) -> 'Model':
        """
        Samples a single instance of a ``x.searchspace.Model`` according to the
        MnasNet search space. The MnasNet search space makes use of the MobileNetV2
        layer settings and input and output channels. These settings are used
        here.

        Args:
            rng (np.random.Generator): A random number generator to uniformly distribute the models parameters.

        Returns:
            x.searchspace.Model: Itself to allow for method chaining.
        """
        self.blocks = []

        round_nearest = 8

        # As per MobileNetV2 architecture. Copied from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        in_channels = _make_divisible(32 * self.width_mult, round_nearest)

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

        for t, c, n, s in layer_settings:
            choice = _layer_choices[rng.integers(0, len(_layer_choices))]
            n_layers = n + choice

            if n_layers == 0:
                continue

            choice = _channel_choices[rng.integers(0, len(_channel_choices))]
            out_channels = _make_divisible(
                c * choice * self.width_mult, round_nearest
            )

            self.blocks.append(
                Block(n_layers, in_channels, out_channels, s, t).sample(rng)
            )

            in_channels = out_channels

        self.last_channels = _make_divisible(
            1280 * max(1.0, self.width_mult), round_nearest
        )

        return self


def to_json(model: Model) -> str:
    """
    Converts the ``model`` to JSON.

    Args:
        model (x.searchspace.Model): The model to be serialized.

    Returns:
        str: A string containing the model as a JSON.
    """
    return json.dumps(asdict(model))


def from_json(config: str | dict[str, Any]) -> Model:
    """
    Converts JSON to a ``x.searchspace.Model``.

    Args:
        config (str, dict[str, Any]): Either a string containing the configuration in JSON format or a ``dict``.

    Returns:
        x.searchspace.Model: The model constructed from the JSON configuration.
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
        block.first_skip, \
        block.rest_skip = \
            itemgetter(
                "conv_op", "kernel_size", "se_ratio", "first_skip", "rest_skip"
            )(block_config)

        blocks.append(block)

    model.blocks = blocks
    model.last_channels = config["last_channels"]

    return model

def sample(
    n_classes: int,
    width_mult: float = 1.0,
    resolution: int = 224,
    dropout: float = 0.2,
    m: int = 1000
) -> Iterator[Model]:
    """
    Provides ``m`` samples for the search space given by ``width_mult`` and
    ``resolution``.

    Args:
        n_classes (int): The number of classes the networks use.
        width_mult (float): Modifies the input and output channels of the hidden layers.
        resolution (int): The resolution of the input image.
        dropout (float): Dropout rate for the classifier of MobileNetV2.
        m (int): The amount of samples generated.

    Returns:
        Iterator[x.searchspace.Model]: A generator to fetch the sampled models.
    """
    size = range(m)
    rng = np.random.default_rng()

    for _ in size:
        yield Model(n_classes, width_mult, resolution, dropout).sample(rng)


@dataclass(frozen=True)
class Configuration():
    width_mult: float
    resolution: int


def configurations() -> list[Configuration]:
    width_step = 16
    width_choices = [48 + i * width_step for i in range(12)]

    resolution_step = 0.1
    resolution_choices = [0.2 + i * resolution_step for i in range(9)]

    # cross product to get all combinations
    cross = np.array(
        np.meshgrid(width_choices, resolution_choices)
    ).T.reshape(-1, 2)

    return [Configuration(float(w), int(r)) for r, w in cross]



class EDF():
    _sorted: np.ndarray
    _buckets: np.ndarray
    _events: np.ndarray

    def __init__(self, samples: list[int] | np.ndarray) -> None:
        if isinstance(sample, list):
            samples = np.asarray(samples)

        self._sorted = np.sort(samples)
        self._buckets, counts = np.unique(self._sorted, return_counts=True)
        self._events = np.cumsum(counts)

    def __call__(
        self, x: int | list[int] | np.ndarray
    ) -> float | list[float] | np.ndarray:
        """
        Evaluates the EDF (Estimated Distribution Function) for ``x``. The function
        is defined as:
        .. math::
            F(x) = 1/n * \sum_{i = 0}^{n} 1[x_i < x]

        :math:`F(x)` returns the fraction of values in samples less than x.
        The return value mirrors the input type.

        Args:
            x (float, list[float], np.ndarray): The x values(s) to evaluate the function for.

        Returns:
            float, list[float], np.ndarray: The y values(s) of the function.
        """
        if isinstance(x, int):
            xs = np.asarray([x])
        elif isinstance(x, list):
            xs = np.asarray(x)
        else:
            xs = x

        ys = self._evaluate(xs)

        if isinstance(x, int):
            return float(ys[0])
        elif isinstance(x, list):
            return float(list(ys))
        else:
            return ys

    def _evaluate(self, xs: np.ndarray) -> np.ndarray:
        # Expand the spaced input to make use of broadcasting. Switch the
        # dimensions of the matrix since we want the transpose.
        expanded = np.broadcast_to(xs, (self._buckets.size, xs.size)).T

        # An array containing the indices to self._events for each entry of
        # spaced.
        idxs = np.argmax(self._buckets >= expanded, axis=1)

        # Special case: if x is larger than the max sample, self._buckets >= x
        # returns [False, ...] and thus np.argmax() returns 0. But it needs to
        # be self._buckets.size - 1, as 100% of samples are below this x.
        for i, idx in enumerate(idxs):
            if idx == 0 and xs[i] > self._buckets[-1]:
                idxs[i] = self._buckets.size - 1

        return self._events[idxs] / self._sorted.size

    def show(self, sample: int | None = None, ax: plt.Axes | None = None) -> None:
        """
        Plots this EDF.

        Args:
            sample (int, None): The number of x values to plot for. The default
                                is the amount of samples provided in the constructor.
            ax (matplotlib.pyplot.Axes, None): A pyplot Axes to draw on.
        """
        if sample is None:
            sample = self._sorted.size

        spaced = np.linspace(
            start=self._buckets[0],
            stop=self._buckets[-1],
            num=sample,
            dtype=np.int32
        )

        if ax is not None:
            ax.plot(spaced, self._evaluate(spaced))
        else:
            plt.plot(spaced, self._evaluate(spaced))

    def percentile(self, p: float) -> int:
        """
        Gets the FLOPs for which ``p``% models are equal or below.

        Args:
                p (float): The percentile.

        Returns:
                int: The FLOPs.
        """
        steps = self._events / self._sorted.size

        for i in range(steps.size):
            idx = steps.size - i - 1

            if p <= steps[idx]:
                return self._buckets[idx]

        return self._buckets[0]
