import torch
import torch.nn as nn

from torchvision.ops import SqueezeExcitation

from ..searchspace.layers import (
    LayerName,
    BaseModule,
    Conv2dModule,
    DWConv2dModule,
    BDWRConv2dModule
)


__all__ = [
    "l1_sort",
    "l1_reorder_conv2d",
    "l1_reorder_batchnorm2d",
    "l1_reorder_se",
    "l1_reorder_main",
    "has_norm",
    "reduce_channels"
]


def l1_sort(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sorts the values of the tensor `tensor` by taking the L1 norm per output
    channel. The higher the result, the more important is the entry.

    Args:
        tensor (torch.Tensor):
            The input tensor

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            The indices of the L1 norm sorting and the a new `torch.Tensor`
            containing the values of `tensor` reordered according to the
            indices.
    """
    if len(tensor.size()) != 4:
        raise ValueError(f"expected 4 dimensions got {len(tensor)}")

    with torch.no_grad():
        l1 = torch.sum(torch.abs(tensor), dim=(1, 2, 3))
        _, indices = torch.sort(l1, dim=0, descending=True)
        reordered = torch.index_select(tensor, dim=0, index=indices)

    return indices, reordered


def l1_reorder_conv2d(
    conv2d: nn.Conv2d, dim: int, indices: torch.Tensor | None
) -> None:
    """
    Performs the reordering of all parameters in `conv2d` according to
    `indices`.

    Args:
        conv2d (torch.nn.Conv2d):
            The conv2d.
        indices (torch.Tensor, None):
            The indices to which the parameters should be reordered. If not
            applicable, set to `None`.
    """
    if indices is None:
        return

    conv2d.weight = nn.Parameter(
        torch.index_select(conv2d.weight, dim=dim, index=indices)
    )


def l1_reorder_batchnorm2d(
    batchnorm2d: nn.BatchNorm2d, indices: torch.Tensor | None
) -> None:
    """
    Performs the reordering of all parameters in `batchnorm2d` according to
    `indices`.

    Args:
        batchnorm2d (torch.nn.BatchNorm2d):
            The batchnorm2d.
        indices (torch.Tensor, None):
            The indices to which the parameters should be reordered. If not
            applicable, set to `None`.
    """
    if indices is None:
        return

    batchnorm2d.weight = nn.Parameter(
        torch.index_select(batchnorm2d.weight, dim=0, index=indices)
    )
    batchnorm2d.bias = nn.Parameter(
        torch.index_select(batchnorm2d.bias, dim=0, index=indices)
    )

    if batchnorm2d.running_mean is not None:
        batchnorm2d.register_buffer(
            "running_mean",
            torch.index_select(batchnorm2d.running_mean, dim=0, index=indices)
        )

    if batchnorm2d.running_var is not None:
        batchnorm2d.register_buffer(
            "running_var",
            torch.index_select(batchnorm2d.running_var, dim=0, index=indices)
        )


def l1_reorder_se(
    se: SqueezeExcitation, indices: torch.Tensor | None
) -> None:
    """
    Performs the reordering of all parameters in `se` according to
    `indices`.

    Args:
        se (torchvision.ops.SqueezeExcitation):
            The squeeze and excitation.
        indices (torch.Tensor, None):
            The indices to which the parameters should be reordered. If not
            applicable, set to `None`.
    """
    if indices is None:
        return

    fc1: nn.Conv2d = se.fc1
    fc1.weight = nn.Parameter(
        torch.index_select(fc1.weight, dim=1, index=indices)
    )

    fc2: nn.Conv2d = se.fc2
    fc2.weight = nn.Parameter(
        torch.index_select(fc2.weight, dim=0, index=indices)
    )
    fc2.bias = nn.Parameter(
        torch.index_select(fc2.bias, dim=0, index=indices)
    )


def l1_reorder_main(
    conv2d: nn.Conv2d, prev_indices: torch.Tensor | None
) -> torch.Tensor:
    """
    Performs two reorderings of `conv2d`. First, reorder the input channel
    weights according to the indices of a reordering of a previous layer.
    Second, reorder the output channel weights of `conv2d`. If `conv2d` has a
    bias, apply the necessary reordering of its weights too.

    Args:
        conv2d (torch.nn.Conv2d):
            The conv2d.
        prev_indices (torch.Tensor, None):
            The indices of the reordering of a previous layer. If not applicable,
            set to `None`.

    Returns:
        torch.Tensor:
            The indices of the reordering of the weights of `conv2d`.
    """
    l1_reorder_conv2d(conv2d, 1, prev_indices)

    indices, reordered = l1_sort(conv2d.weight)
    conv2d.weight = nn.Parameter(reordered)

    if conv2d.bias is not None:
        conv2d.bias = nn.Parameter(
            torch.index_select(conv2d.bias, dim=0, index=indices)
        )

    return indices


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
    module: Conv2dModule, in_channels: int, out_channels: int
) -> Conv2dModule:
    reduced = Conv2dModule(
        in_channels,
        out_channels,
        module.se_ratio,
        module.shortcut,
        module[LayerName.CONV2D][0].kernel_size,
        module[LayerName.CONV2D][0].stride,
    )

    with torch.no_grad():
        _copy_weights_conv2d(
            reduced[LayerName.CONV2D][0],
            module[LayerName.CONV2D][0],
            in_channels, out_channels
        )
        if has_norm(reduced[LayerName.CONV2D]):
            _copy_weights_norm(
                reduced[LayerName.CONV2D][1],
                module[LayerName.CONV2D][1],
                out_channels
            )

        if LayerName.SE in reduced:
            _copy_weights_squeeze(
                reduced[LayerName.SE], module[LayerName.SE], out_channels
            )

    return reduced


def _reduce_dwconv2d(
    module: DWConv2dModule, in_channels: int, out_channels: int
) -> Conv2dModule:
    reduced = DWConv2dModule(
        in_channels,
        out_channels,
        module.se_ratio,
        module.shortcut,
        module[LayerName.CONV2D][0].kernel_size,
        module[LayerName.CONV2D][0].stride,
    )

    with torch.no_grad():
        _copy_weights_conv2d(
            reduced[LayerName.CONV2D][0], module[LayerName.CONV2D][0],
            in_channels, in_channels
        )
        if has_norm(reduced[LayerName.CONV2D]):
            _copy_weights_norm(
                reduced[LayerName.CONV2D][1],
                module[LayerName.CONV2D][1],
                in_channels
            )

        if LayerName.SE in reduced:
            _copy_weights_squeeze(
                reduced[LayerName.SE], module[LayerName.SE], in_channels
            )

        _copy_weights_conv2d(
            reduced[LayerName.PWCONV2D][0], module[LayerName.PWCONV2D][0],
            in_channels, out_channels
        )

        if has_norm(reduced[LayerName.PWCONV2D]):
            _copy_weights_norm(
                reduced[LayerName.PWCONV2D][1],
                module[LayerName.PWCONV2D][1],
                out_channels
            )

    return reduced


def _reduce_bdwrconv2d(
    module: BDWRConv2dModule, in_channels: int, out_channels: int
) -> Conv2dModule:
    reduced = BDWRConv2dModule(
        in_channels,
        out_channels,
        module.expansion_ratio,
        module.se_ratio,
        module.shortcut,
        module[LayerName.CONV2D][0].kernel_size,
        module[LayerName.CONV2D][0].stride,
    )

    with torch.no_grad():
        hidden = in_channels * module.expansion_ratio
        _copy_weights_conv2d(
            reduced[LayerName.EXPANSION][0],
            module[LayerName.EXPANSION][0],
            in_channels, hidden
        )
        if has_norm(reduced[LayerName.EXPANSION]):
            _copy_weights_norm(
                reduced[LayerName.EXPANSION][1],
                module[LayerName.EXPANSION][1],
                hidden
            )

        _copy_weights_conv2d(
            reduced[LayerName.CONV2D][0],
            module[LayerName.CONV2D][0],
            hidden, hidden
        )
        if has_norm(reduced[LayerName.CONV2D]):
            _copy_weights_norm(
                reduced[LayerName.CONV2D][1], module[LayerName.CONV2D][1], hidden
            )

        if LayerName.SE in reduced:
            _copy_weights_squeeze(
                reduced[LayerName.SE], module[LayerName.SE], hidden
            )

        _copy_weights_conv2d(
            reduced[LayerName.PWCONV2D][0],
            module[LayerName.PWCONV2D][0],
            hidden, out_channels
        )
        if has_norm(reduced[LayerName.PWCONV2D]):
            _copy_weights_norm(
                reduced[LayerName.PWCONV2D][1],
                module[LayerName.PWCONV2D][1],
                out_channels
            )

    return reduced


def reduce_channels(
    module: BaseModule, in_channels: int, out_channels: int
) -> BaseModule:
    """
    Produces a copy of `module` with the input and output channels reduces to
    `in_channels` and `out_channels`. Copies all the weights cut off at the new
    channel size.

    Args:
        module (BaseModule):
            The module to reduce.
        in_channels (int):
            The new and reduces number of input channels.
        out_channels (int):
            The new and reduces number of output channels.

    Returns:
        BaseModule:
            A copy of `module` with the input ad output channels reduced.
    """
    reducer = {
        Conv2dModule: _reduce_conv2d,
        DWConv2dModule: _reduce_dwconv2d,
        BDWRConv2dModule: _reduce_bdwrconv2d
    }

    fn = reducer.get(module.__class__)
    if fn is None:
        raise RuntimeError(f"unknown operation {type(module)}")

    return fn(module, in_channels, out_channels)
