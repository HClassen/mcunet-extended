from bisect import insort_left

import torch
import torch.nn as nn
import torch.linalg as la

from torchvision.ops import SqueezeExcitation

from ..searchspace import ConvOp
from ..searchspace.layers import (
    LayerName,
    BaseOp,
    Conv2dOp,
    DWConv2dOp,
    BDWRConv2dOp
)


__all__ = ["l1_sort", "has_norm", "reduce_channels", "base_op_to_conv_op"]


def l1_sort(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """
    Sorts the values of the tensor `weight` by taking the L1 norm of the entries
    of its first dimension. The higher the result, the more important is the
    entry.

    Args:
        tensor (torch.Tensor):
            The input tensor

    Returns:
        tuple[torch.Tensor, tuple[int, ...]]:
            A new tensor with the sorted entries of `tensor` and a tuple with
            the indices of sorted entries.
    """
    importance: list[tuple[int, float]] = []

    for channel in range(tensor.shape[0]):
        l1 = la.norm(tensor[channel].view(-1), ord=1)
        insort_left(
            importance, (channel, l1), key=lambda entry: -entry[1]
        )

    reordered = torch.empty_like(tensor, requires_grad=tensor.requires_grad)
    with torch.no_grad():
        for k, (channel, _) in enumerate(importance):
            reordered[k].copy_(tensor[channel])

    return reordered, tuple(map(lambda x: x[0], importance))


def _copy_weights_conv2d(
    dst: nn.Conv2d, src: nn.Conv2d, in_channels: int, out_channels: int
) -> None:
    dst.weight.copy_(src.weight[:out_channels, :in_channels])
    if src.bias is not None and dst.bias is not None:
        dst.bias.copy_(src.bias[:out_channels])


def _copy_weights_norm(
    dst: nn.BatchNorm2d, src: nn.BatchNorm2d, out_channels: int
) -> None:
    dst.weight.copy_(src.weight[:out_channels])
    dst.bias.copy_(src.bias[:out_channels])
    dst.running_mean.copy_(src.running_mean[:out_channels])
    dst.running_var.copy_(src.running_var[:out_channels])
    dst.num_batches_tracked = src.num_batches_tracked


def _copy_weights_squeeze(
    dst: SqueezeExcitation, src: SqueezeExcitation, in_channels: int
) -> None:
    out_channels = dst.fc1.weight.shape[0]

    _copy_weights_conv2d(dst.fc1, src.fc1, in_channels, out_channels)
    _copy_weights_conv2d(dst.fc2, src.fc2, out_channels, in_channels)


def has_norm(sequential: nn.Sequential) -> bool:
    """
    Checks if the second module in `sequential` is a a `BatchNorm2d` layer.

    Args:
        sequential (torch.nn.Sequential):
            The module to check.

    Returns:
        bool:
            `True`, if the scond module is `BatchNorm2d` layer.
    """
    return len(sequential) > 1 and isinstance(sequential[1], nn.BatchNorm2d)


def _reduce_conv2d(
    op: Conv2dOp, in_channels: int, out_channels: int
) -> Conv2dOp:
    reduced = Conv2dOp(
        in_channels,
        out_channels,
        op.se_ratio,
        op.skip,
        op[LayerName.CONV2D][0].kernel_size,
        op[LayerName.CONV2D][0].stride,
    )

    with torch.no_grad():
        _copy_weights_conv2d(
            reduced[LayerName.CONV2D][0],
            op[LayerName.CONV2D][0],
            in_channels, out_channels
        )
        if has_norm(reduced[LayerName.CONV2D]):
            _copy_weights_norm(
                reduced[LayerName.CONV2D][1],
                op[LayerName.CONV2D][1],
                out_channels
            )

        if LayerName.SE in reduced:
            _copy_weights_squeeze(
                reduced[LayerName.SE], op[LayerName.SE], out_channels
            )

    return reduced


def _reduce_dwconv2d(
    op: DWConv2dOp, in_channels: int, out_channels: int
) -> Conv2dOp:
    reduced = DWConv2dOp(
        in_channels,
        out_channels,
        op.se_ratio,
        op.skip,
        op[LayerName.CONV2D][0].kernel_size,
        op[LayerName.CONV2D][0].stride,
    )

    with torch.no_grad():
        _copy_weights_conv2d(
            reduced[LayerName.CONV2D][0], op[LayerName.CONV2D][0],
            in_channels, in_channels
        )
        if has_norm(reduced[LayerName.CONV2D]):
            _copy_weights_norm(
                reduced[LayerName.CONV2D][1],
                op[LayerName.CONV2D][1],
                in_channels
            )

        if LayerName.SE in reduced:
            _copy_weights_squeeze(
                reduced[LayerName.SE], op[LayerName.SE], in_channels
            )

        _copy_weights_conv2d(
            reduced[LayerName.PWCONV2D][0], op[LayerName.PWCONV2D][0],
            in_channels, out_channels
        )

        if has_norm(reduced[LayerName.PWCONV2D]):
            _copy_weights_norm(
                reduced[LayerName.PWCONV2D][1],
                op[LayerName.PWCONV2D][1],
                out_channels
            )

    return reduced


def _reduce_bdwrconv2d(
    op: BDWRConv2dOp, in_channels: int, out_channels: int
) -> Conv2dOp:
    reduced = BDWRConv2dOp(
        in_channels,
        out_channels,
        op.expansion_ratio,
        op.se_ratio,
        op.skip,
        op[LayerName.CONV2D][0].kernel_size,
        op[LayerName.CONV2D][0].stride,
    )

    with torch.no_grad():
        hidden = in_channels * op.expansion_ratio
        _copy_weights_conv2d(
            reduced[LayerName.EXPANSION][0],
            op[LayerName.EXPANSION][0],
            in_channels, hidden
        )
        if has_norm(reduced[LayerName.EXPANSION]):
            _copy_weights_norm(
                reduced[LayerName.EXPANSION][1],
                op[LayerName.EXPANSION][1],
                hidden
            )

        _copy_weights_conv2d(
            reduced[LayerName.CONV2D][0],
            op[LayerName.CONV2D][0],
            hidden, hidden
        )
        if has_norm(reduced[LayerName.CONV2D]):
            _copy_weights_norm(
                reduced[LayerName.CONV2D][1], op[LayerName.CONV2D][1], hidden
            )

        if LayerName.SE in reduced:
            _copy_weights_squeeze(
                reduced[LayerName.SE], op[LayerName.SE], hidden
            )

        _copy_weights_conv2d(
            reduced[LayerName.PWCONV2D][0],
            op[LayerName.PWCONV2D][0],
            hidden, out_channels
        )
        if has_norm(reduced[LayerName.PWCONV2D]):
            _copy_weights_norm(
                reduced[LayerName.PWCONV2D][1],
                op[LayerName.PWCONV2D][1],
                out_channels
            )

    return reduced


def reduce_channels(op: BaseOp, in_channels: int, out_channels: int) -> BaseOp:
    reducer = {
        Conv2dOp: _reduce_conv2d,
        DWConv2dOp: _reduce_dwconv2d,
        BDWRConv2dOp: _reduce_bdwrconv2d
    }

    fn = reducer.get(op.__class__)
    if fn is None:
        raise RuntimeError(f"unknown operation {type(op)}")

    return fn(op, in_channels, out_channels)


def base_op_to_conv_op(op: BaseOp) -> ConvOp:
    if isinstance(op, Conv2dOp):
        return ConvOp.CONV2D

    if isinstance(op, DWConv2dOp):
        return ConvOp.DWCONV2D

    if isinstance(op, BDWRConv2dOp):
        return ConvOp.BDWRCONV2D

    raise ValueError(f"unknown convolution operation: {type(op)}")
