from typing import cast

import torch
import torch.nn as nn

from torchvision.ops import SqueezeExcitation

from mcunet.tinynas.searchspace import LAYER_CHOICES, Model
from mcunet.tinynas.searchspace.layers import LayerName, BaseOp
from mcunet.tinynas.oneshot.customize import ChoiceLayer, ChoiceBlock
from mcunet.tinynas.oneshot.helper import l1_sort, has_norm
from mcunet.tinynas.mobilenet import MobileSkeletonNet

from . import SuperChoiceBlock, CommonWarmUpSetter, simple


__all__ = [
    "Conv2dSharer", "DWConv2dSharer", "BDWRConv2dSharer", "MoreWarmUpSetter"
]


class Conv2dSharer(simple.Conv2dSharer):
    pass


class DWConv2dSharer(simple.DWConv2dSharer):
    pass


class BDWRConv2dSharer(simple.BDWRConv2dSharer):
    pass


def _set_conv2d(
    dst: nn.Conv2d,
    reordered: torch.Tensor,
    bias: torch.Tensor | None,
    importance: tuple[int, ...]
) -> None:
    out_channels, in_channels = dst.weight.shape[0:2]

    with torch.no_grad():
        dst.weight.copy_(reordered[:out_channels, :in_channels])

        for k, channel in enumerate(importance):
            if k >= dst.bias.shape[0]:
                break

            dst.bias[k] = bias[channel]


def _set_se(
    dst: BaseOp,
    fc1_reordered: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc1_importance: tuple[int, ...],
    fc2_reordered: torch.Tensor,
    fc2_bias: torch.Tensor,
    fc2_importance: tuple[int, ...]
) -> None:
    if LayerName.SE not in dst:
        return

    _set_conv2d(
        dst[LayerName.SE].fc1,
        fc1_reordered,
        fc1_bias,
        fc1_importance
    )
    _set_conv2d(
        dst[LayerName.SE].fc2,
        fc2_reordered,
        fc2_bias,
        fc2_importance
    )


def _set_batchnorm2d(
    dst: nn.BatchNorm2d, src: nn.BatchNorm2d, importance: tuple[int, ...]
) -> None:
    with torch.no_grad():
        for k, channel in enumerate(importance):
            if k >= dst.weight.shape[0]:
                break

            dst.weight[k] = src.weight[channel]
            dst.bias[k] = src.bias[channel]


def _set_conv2d_batchnorm2d(
    dst: BaseOp,
    src: BaseOp,
    key: LayerName,
    reordered: torch.Tensor | None,
    importance: tuple[int, ...] | None
) -> None:
    if (
        key not in dst
        or key not in src
        or reordered is None
        or importance is None
    ):
        return

    sequential_dst: nn.Sequential = dst[key]
    sequential_src: nn.Sequential = src[key]

    _set_conv2d(sequential_dst[0], reordered, None, ())
    if has_norm(sequential_dst) and has_norm(sequential_src):
        _set_batchnorm2d(sequential_dst[1], sequential_src[1], importance)


def _set_not_shared(dst: ChoiceLayer, src: BaseOp) -> None:
    se: SqueezeExcitation = src[LayerName.SE]

    fc1_reordered, fc1_importance = l1_sort(se.fc1.weight)
    fc2_reordered, fc2_importance = l1_sort(se.fc2.weight)

    if LayerName.EXPANSION in src:
        expansion: nn.Sequential = src[LayerName.EXPANSION]
        expansion_reordered, expansion_importance = l1_sort(expansion[0].weight)
    else:
        expansion_reordered, expansion_importance = None, None

    if LayerName.PWCONV2D in src:
        pwconv2d: nn.Sequential = src[LayerName.PWCONV2D]
        pwconv2d_reordered, pwconv2d_importance = l1_sort(pwconv2d[0].weight)
    else:
        pwconv2d_reordered, pwconv2d_importance = None, None

    for choice in dst.choices:
        _set_se(
            choice,
            fc1_reordered,
            se.fc1.bias,
            fc1_importance,
            fc2_reordered,
            se.fc2.bias,
            fc2_importance
        )

        _set_conv2d_batchnorm2d(
            choice,
            src,
            LayerName.EXPANSION,
            expansion_reordered,
            expansion_importance
        )

        _set_conv2d_batchnorm2d(
            choice,
            src,
            LayerName.PWCONV2D,
            pwconv2d_reordered,
            pwconv2d_importance
        )


class MoreWarmUpSetter(CommonWarmUpSetter):
    def set(
        self,
        supernet: MobileSkeletonNet,
        max_model: Model,
        max_net: MobileSkeletonNet
    ) -> None:
        op = max_model.blocks[0].layers[0].op

        for i, block in enumerate(supernet.blocks):
            block = cast(SuperChoiceBlock, block)

            choice = cast(ChoiceBlock, block.choices[op])
            for j, layer in enumerate(choice.layers):
                layer = cast(ChoiceLayer, layer)
                src = cast(BaseOp, max_net.blocks[i * max(LAYER_CHOICES) + j])

                simple._set_shared(layer, src)
                _set_not_shared(layer, src)

        self._after_set(max_net)
