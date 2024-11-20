from typing import Annotated
from typing_extensions import Doc
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn as nn

from torchvision.ops import SqueezeExcitation

from .helper import l1_sort


__all__ = [
    "make_shared_conv2d",
    "make_shared_batchnorm2d",
    "share_conv2d",
    "share_batchnorm2d",
    "init_conv2d",
    "init_batchnorm2d",
    "set_conv2d",
    "set_batchnorm2d",
    "SharedWeightsConv2d",
    "SharedWeightsBatchNorm2d",
    "SharedWeightsSqueezeExcitation",
    "ParameterSharer",
    "ParameterSharerConstructor"
]


def make_shared_conv2d(
    shape: torch.Size | tuple[int, int, int, int], bias: bool
) -> tuple[nn.Parameter, nn.Parameter | None]:
    """
    Creates shared weight and bias for of a conv2d. The bias is optional.

    Args:
        shape (torch.Size, tuple[int, int, int, int]):
            The dimensions of the shared parameters.
        bias (bool):
            Switch if a bias is needed or not.

    Returns:
        tuple[torch.nn.Parameter, torch.nn.Parameter | None]:
            The shared weight and bias of a conv2d.
    """
    if len(shape) != 4:
        raise ValueError(f"expected 4 dimensions got {len(shape)}")

    weight = nn.Parameter(torch.empty(shape))
    bias = nn.Parameter(torch.empty((shape[0]))) if bias else None

    return weight, bias


def make_shared_batchnorm2d(
    shape: torch.Size | int, track_running_stats: bool
) -> tuple[
    nn.Parameter,
    nn.Parameter,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None
]:
    """
    Creates shared parameters for a batchnorm2d. The running mean, variance and
    number of tracked batches are optional.

    Args:
        shape (torch.Size, tuple[int, int, int, int]):
            The dimensions of the shared parameters.
        track_running_stats (bool):
            Switch if running stats is needed or not.

    Returns:
        tuple[
            torch.nn.Parameter,
            torch.nn.Parameter,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None
        ]:
            The shared weight, biasm running mean, running variance and number
            of tracked batches of a batchnorm2d.
    """
    if isinstance(shape, torch.Size) and len(shape) != 1:
        raise ValueError(f"expected 1 dimensions got {len(shape)}")

    weight = nn.Parameter(torch.empty(shape))
    bias = nn.Parameter(torch.empty(shape))

    mean, var, tracked = None, None, None
    if track_running_stats:
        mean = torch.empty(shape)
        var = torch.empty(shape)
        tracked = torch.tensor(0, dtype=torch.long)

    return weight, bias, mean, var, tracked


def share_conv2d(
    conv2d: nn.Conv2d,
    weight: nn.Parameter,
    bias: nn.Parameter | None
) -> None:
    """
    Set the weight and bias of `conv2d` to shared parameters. The bias is only
    shared if `bias is not None and conv2d.bias is not None`.

    The size of `weight` and `bias` must be equal or greater than required by
    `conv2d`.

    With c_in, c_out as the in and out channels of `conv2d` and h, w the height
    and width of the kernel, the sharing is applied as
    `conv2d.weight = weight[:c_out, :c_in, h - center:h + center, w - center:c + center]`
    and
    `conv2d.bias = bias[c:out:]`.

    Args:
        conv2d (torch.nn.Conv2d):
            The convolution module.
        weight (torch.nn.Parameter):
            The shared weight.
        bias (torch.nn.Parameter, None):
            The shared bias.
    """
    _, _, weight_height, weight_width = weight.size()

    weight_height_center = int((weight_height - 1) / 2)
    weight_width_center = int((weight_width - 1) / 2)

    if isinstance(conv2d.kernel_size, tuple):
        kernel_height, kernel_width = conv2d.kernel_size
    else:
        kernel_height, kernel_width = conv2d.kernel_size, conv2d.kernel_size

    height_center = int((kernel_height - 1) / 2)
    width_center = int((kernel_width - 1) / 2)

    h_start = weight_height_center - height_center
    h_end = weight_height_center + height_center + 1

    w_start = weight_width_center - width_center
    w_end = weight_width_center + width_center + 1

    out_channels = conv2d.out_channels
    in_channels = int(conv2d.in_channels / conv2d.groups)

    conv2d.weight = nn.Parameter(
        weight[:out_channels, :in_channels, h_start:h_end, w_start:w_end]
    )

    if conv2d.bias is None or bias is None:
        return

    conv2d.bias = nn.Parameter(bias[:out_channels])


def share_batchnorm2d(
    batchnorm2d: nn.BatchNorm2d,
    weight: nn.Parameter,
    bias: nn.Parameter,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    num_batches_tracked: torch.Tensor | None
) -> None:
    """
    Set the weight, bias, running mean, running variance and number of tracked
    batches of `batchnorm2d` to shared parameters. The running mean, running
    variance  and number of tracked batches are optional.

    The sizes of the parameters must be equal or greater than required by
    `batchnorm2d`.

    The parameters are index by `[:num_features]`, except for
    `num_batches_tracked`.

    Args:
        batchnorm2d (torch.nn.BatchNorm2d):
            The batchnorm2de.
        weight (nn.Parameter):
            The shared weight.
        bias (torch.nn.Parameter):
            The shared bias.
        running_mean (torch.Tensor, None):
            The shared running mean.
        running_var (torch.Tensor, None):
            The shared running variance.
        num_batches_tracked (torch.Tensor, None):
            The shared number of tracked batches.
    """
    num_features = batchnorm2d.num_features
    batchnorm2d.weight = nn.Parameter(weight[:num_features])
    batchnorm2d.bias = nn.Parameter(bias[:num_features])

    if running_mean is not None:
        batchnorm2d.running_mean = running_mean[:num_features]

    if running_var is not None:
        batchnorm2d.running_var = running_var[:num_features]

    if num_batches_tracked is not None:
        batchnorm2d.num_batches_tracked = num_batches_tracked


def init_conv2d(weight: nn.Parameter, bias: nn.Parameter | None) -> None:
    """
    Initializes the two shared parameters of a conv2d.

    Args:
        weight (torch.nn.Parameter):
            The shared weight of the conv2d.
        bias (torch.nn.Parameter, None):
            The shared optinal bias of the conv2d.
    """
    nn.init.kaiming_normal_(weight, mode="fan_out")
    if bias is not None:
        nn.init.zeros_(bias)


def init_batchnorm2d(
    weight: nn.Parameter,
    bias: nn.Parameter,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    num_batches_tracked: torch.Tensor | None,
) -> None:
    """
    Initializes the five shared parameters of a batchnorm2d.

    Args:
        weight (torch.nn.Parameter):
            The shared weight of the batchnorm2d.
        bias (torch.nn.Parameter):
            The shared bias of the batchnorm2d.
        running_mean (torch.Tensor, None):
            The optional shared running mean of the batchnorm2d.
        running_var (torch.Tensor, None):
            The optional shared running variance of the batchnorm2d.
        num_batches_tracked (torch.Tensor, None):
            The optional shared number of tracked batches of the batchnorm2d.
    """
    nn.init.ones_(weight)
    nn.init.zeros_(bias)

    if running_mean is not None:
        nn.init.zeros_(running_mean)

    if running_var is not None:
        nn.init.ones_(running_var)

    if num_batches_tracked is not None:
        nn.init.zeros_(num_batches_tracked)


def set_conv2d(
    weight: nn.Parameter,
    bias: nn.Parameter | None,
    conv2d: nn.Conv2d
) -> list[int]:
    """
    Set the two shared parameters of a conv2d to the weights in
    `conv2d`.

    Args:
        weight (torch.nn.Parameter):
            The shared weight of the conv2d.
        bias (torch.nn.Parameter, None):
            The shared bias of the conv2d.
        conv2d (torch.nn.Conv2d):
            The layers to copy the weights from.

    Returns:
        list[int]:
            The reordered channels.
    """
    src_weight = conv2d.weight
    reordered, importance = l1_sort(src_weight)

    with torch.no_grad():
        weight.copy_(reordered)

        if conv2d.bias is None or bias is None:
            return

        src_bias = conv2d.bias
        for k, channel in enumerate(importance):
            bias[k] = src_bias[channel]

    return importance


def set_batchnorm2d(
    weight: nn.Parameter,
    bias: nn.Parameter,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    num_batches_tracked: torch.Tensor | None,
    batchnorm2d: nn.BatchNorm2d,
    reorder: list[int] | None
) -> None:
    """
    Set the three shared parameters of a conv2d followed by a
    batch-normalisation to the weights in `sequential`.

    Args:
        weight (torch.nn.Parameter):
            The shared weight of the conv2d.
        norm_weight (torch.nn.Parameter):
            The shared weight of the batch-normalisation.
        norm_bias (torch.nn.Parameter):
            The shared bias of the batch-normalisation.
        sequential (torch.nn.Sequential):
            The layers to copy the weights from.
    """
    reorder = reorder if reorder is not None \
              else [i for i in range(weight.size()[0])]

    with torch.no_grad():
        src_weight = batchnorm2d.weight
        src_bias = batchnorm2d.bias

        src_running_mean = batchnorm2d.running_mean
        src_running_var = batchnorm2d.running_var
        for k, channel in enumerate(reorder):
            weight[k] = src_weight[channel]
            bias[k] = src_bias[channel]

            if running_mean is not None and src_running_mean is not None:
                running_mean[k] = src_running_mean[channel]

            if running_var is not None and src_running_var is not None:
                running_var[k] = src_running_var[channel]

        src_num_batches_tracked = batchnorm2d.num_batches_tracked
        if num_batches_tracked is not None \
           and src_num_batches_tracked is not None:
            num_batches_tracked.copy_(src_num_batches_tracked)


class SharedWeightsConv2d(nn.Module):
    _weight: nn.Parameter
    _bias: nn.Parameter | None

    _reorder: list[int] | None

    def __init__(
        self, shape: torch.Size | tuple[int, int, int, int], bias: bool
    ) -> None:
        super().__init__()

        self._weight, self._bias = make_shared_conv2d(shape, bias)
        self._reorder = None

    def share(self, conv2d: nn.Conv2d) -> None:
        share_conv2d(conv2d, self._weight, self._bias)

    def _weight_initialization(self) -> None:
        init_conv2d(self._weight, self._bias)

    def _weight_copy(self, src: nn.Conv2d) -> None:
        self._reorder = set_conv2d(self._weight, self._bias, src)


class SharedWeightsBatchNorm2d(nn.Module):
    _weight: nn.Parameter
    _bias: nn.Parameter

    _running_mean: torch.Tensor | None
    _running_var: torch.Tensor | None
    _num_batches_tracked: torch.Tensor | None

    def __init__(
        self, shape: torch.Size | int, track_running_stats: bool
    ) -> None:
        super().__init__()

        shared = make_shared_batchnorm2d(shape, track_running_stats)

        self._weight = shared[0]
        self._bias = shared[1]

        self.register_buffer("_running_mean", shared[2])
        self.register_buffer("_running_var", shared[3])
        self.register_buffer("_num_batches_tracked", shared[4])

    def share(self, batchnorm2d: nn.BatchNorm2d) -> None:
        share_batchnorm2d(
            batchnorm2d,
            self._weight,
            self._bias,
            self._running_mean,
            self._running_var,
            self._num_batches_tracked
        )

    def _weight_initialization(self) -> None:
        init_batchnorm2d(
            self._weight,
            self._bias,
            self._running_mean,
            self._running_var,
            self._num_batches_tracked
        )

    def _weight_copy(
        self, src: nn.BatchNorm2d, reorder: list[int] | None
    ) -> None:
        set_batchnorm2d(
            self._weight,
            self._bias,
            self._running_mean,
            self._running_var,
            self._num_batches_tracked,
            src,
            reorder
        )


class SharedWeightsSqueezeExcitation(nn.Module):
    _fc1: SharedWeightsConv2d
    _fc2: SharedWeightsConv2d

    def __init__(self, in_channels: int, se_channels: int) -> None:
        super().__init__()

        self._fc1 = SharedWeightsConv2d((se_channels, in_channels, 1, 1), True)
        self._fc2 = SharedWeightsConv2d((in_channels, se_channels, 1, 1), True)

    def share(self, se: SqueezeExcitation) -> None:
        self._fc1.share(se.fc1)
        self._fc2.share(se.fc2)

    def _weight_initialization(self) -> None:
        self._fc1._weight_initialization()
        self._fc2._weight_initialization()

    def _weight_copy(self, se: SqueezeExcitation) -> None:
        self._fc1._weight_copy(se.fc1)
        self._fc2._weight_copy(se.fc2)


class ParameterSharer(ABC, nn.Module):
    def __init__(self, max_in_channels: int, max_out_channels: int) -> None:
        super().__init__()

        self._make_shared(max_in_channels, max_out_channels)

    @abstractmethod
    def _make_shared(self, max_in_channels: int, max_out_channels: int) -> None:
        pass

    @abstractmethod
    def set_shared(self, module: nn.Module) -> None:
        pass

    @abstractmethod
    def unset_shared(self, module: nn.Module) -> None:
        pass

    @abstractmethod
    def _weight_initialization(self) -> None:
        pass

    @abstractmethod
    def _weight_copy(self, module: nn.Module) -> None:
        pass


type ParameterSharerConstructor = Annotated[
    Callable[[int, int], ParameterSharer],
    Doc(
        """
        Creates an instance of `ParameterSharer`. This just a concrete type for
        the `ParameterSharer` constructor used for type annotations.

        Args:
            max_in_channels (int):
                The maximum number of input channels for this layer.
            max_out_channels (int):
                The maximum number of output channels for this layer.

        Returns:
            ParameterSharer:
                An instance of a `ParameterSharer` sub-class.
        """
    )
]
