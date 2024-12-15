from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .helper import l1_sort, has_norm


__all__ = [
    "share_conv2d",
    "share_batchnorm2d",
    "shared_conv2d_batchnorm2d",
    "init_conv2d_batchnorm2d",
    "set_conv2d_batchnorm2d",
    "ParameterSharer"
]


def share_conv2d(
    conv2d: nn.Conv2d,
    weight: nn.Parameter,
    bias: nn.Parameter | None
) -> None:
    """
    Set the weight and bias of `conv2d` to shared parameters. The bias is only
    shared of `bias is not None and conv2d.bias is not None`.

    The size of `weight` and `bias` must be equal or greater than required by
    `conv2d`.

    With c_in, c_out as the in and out channels of `conv2d` and h, w the height
    and width of the kernel, the sharing is applied as
    `conv2d.weight = weight[:c_out, :c_in, h - center:h + center, w - center:c + center]`
    and
    `conv2d.bias = bias[c:out:]`.

    Args:
        conv2d (nn.Conv2d):
            The convolution module.
        weight (nn.Parameter):
            The shared weight.
        bias (nn.Parameter, None):
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
    norm: nn.BatchNorm2d,
    weight: nn.Parameter,
    bias: nn.Parameter,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    num_batches_tracked: torch.Tensor | None
) -> None:
    """
    Set the weight, bias, running mean, running variance and number of tracked
    batches of `norm` to shared parameters. The running mean, running variance
    and number of tracked batches are optional.

    The sizes of the parameters must be equal or greater than required by `norm`.

    The parameters are index by `[:num_features]`, except for
    `num_batches_tracked`.

    Args:
        norm (nn.BatchNorm2d):
            The batch normalization module.
        weight (nn.Parameter):
            The shared weight.
        bias (nn.Parameter):
            The shared bias.
        running_mean (torch.Tensor, None):
            The shared running mean.
        running_var (torch.Tensor, None):
            The shared running variance.
        num_batches_tracked (torch.Tensor, None):
            The shared number of tracked batches.
    """
    num_features = norm.num_features
    norm.weight = nn.Parameter(weight[:num_features])
    norm.bias = nn.Parameter(bias[:num_features])

    if running_mean is not None:
        norm.running_mean = running_mean[:num_features]

    if running_var is not None:
        norm.running_var = running_var[:num_features]

    if num_batches_tracked is not None:
        norm.num_batches_tracked = num_batches_tracked


def shared_conv2d_batchnorm2d(
    out_channels: int,
    in_channels: int,
    kernel_size: tuple[int, int]
) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter]:
    """
    Creates the three shared parameters for a conv2d followed by a
    batch-normalisation.

    Args:
        out_channels (int):
            The out channels of the two layers.
        in_channels (int):
            The in channels of the two layers.
        kernel_size (tuple[int, int]):
            The kernel size of the conv2d.

    Returns:
        tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
            The shared weight of the conv2d, the shared weight and bias of the
            batch-normalisation.
    """
    conv2d_weight = nn.Parameter(
        torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1])
    )
    norm_weight = nn.Parameter(torch.Tensor(out_channels))
    norm_bias = nn.Parameter(torch.Tensor(out_channels))

    return (conv2d_weight, norm_weight, norm_bias)


def init_conv2d_batchnorm2d(
    weight: nn.Parameter, norm_weight: nn.Parameter, norm_bias: nn.Parameter
) -> None:
    """
    Initializes the three shared parameters of a conv2d followed by a
    batch-normalisation.

    Args:
        weight (torch.nn.Parameter):
            The shared weight of the conv2d.
        norm_weight (torch.nn.Parameter):
            The shared weight of the batch-normalisation.
        norm_bias (torch.nn.Parameter):
            The shared bias of the batch-normalisation.
    """
    nn.init.kaiming_normal_(weight, mode="fan_out")
    nn.init.ones_(norm_weight)
    nn.init.zeros_(norm_bias)


def set_conv2d_batchnorm2d(
    weight: nn.Parameter,
    norm_weight: nn.Parameter,
    norm_bias: nn.Parameter,
    sequential: nn.Sequential
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
    src_weight = sequential[0].weight
    reordered, importance = l1_sort(src_weight)

    with torch.no_grad():
        weight.copy_(reordered)

        if not has_norm(sequential):
            return

        src_norm_weight = sequential[1].weight
        src_norm_bias = sequential[1].bias
        for k, channel in enumerate(importance):
            norm_weight[k] = src_norm_weight[channel]
            norm_bias[k] = src_norm_bias[channel]


class ParameterSharer(ABC, nn.Module):
    @abstractmethod
    def make_shared(self, **kwargs) -> None:
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
