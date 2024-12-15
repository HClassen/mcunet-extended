from typing import cast

import torch.nn as nn

from ....searchspace import (
    KERNEL_CHOICES,
    EXPANSION_CHOICES,
    LAYER_CHOICES,
    Model
)
from ....searchspace.layers import LayerName, BaseModule
from ....oneshot.layers import ChoiceLayer, ChoiceBlock
from ....oneshot.share import (
    share_conv2d,
    share_batchnorm2d,
    shared_conv2d_batchnorm2d,
    init_conv2d_batchnorm2d,
    set_conv2d_batchnorm2d,
    ParameterSharer
)
from ....oneshot.helper import has_norm
from ....mobilenet import MobileSkeletonNet

from .. import SuperChoiceBlock, CommonWarmUpCtx


__all__ = [
    "Conv2dSharer", "DWConv2dSharer", "BDWRConv2dSharer", "SimpleWarmUpCtx"
]


class SimpleSharer(ParameterSharer):
    conv2d_weight: nn.Parameter
    norm_weight: nn.Parameter
    norm_bias: nn.Parameter

    def set_shared(self, module: BaseModule) -> None:
        share_conv2d(module[LayerName.CONV2D][0], self.conv2d_weight, None)
        if has_norm(module[LayerName.CONV2D]):
            share_batchnorm2d(
                module[LayerName.CONV2D][1], self.norm_weight, self.norm_bias,
                None, None, None
            )

    def unset_shared(self, module: BaseModule) -> None:
        module[LayerName.CONV2D][0].weight = None
        if has_norm(module[LayerName.CONV2D]):
            module[LayerName.CONV2D][1].weight = None
            module[LayerName.CONV2D][1].bias = None

    def _weight_initialization(self) -> None:
        init_conv2d_batchnorm2d(
            self.conv2d_weight, self.norm_weight, self.norm_bias
        )


class Conv2dSharer(SimpleSharer):
    def _make_shared(self, max_in_channels: int, max_out_channels: int) -> None:
        max_kernel_size = (max(KERNEL_CHOICES), max(KERNEL_CHOICES))

        shared = shared_conv2d_batchnorm2d(
            max_out_channels, max_in_channels, max_kernel_size
        )
        self.conv2d_weight, self.norm_weight, self.norm_bias = shared


class DWConv2dSharer(SimpleSharer):
    def _make_shared(self, max_in_channels: int, _: int) -> None:
        max_kernel_size = (max(KERNEL_CHOICES), max(KERNEL_CHOICES))

        shared = shared_conv2d_batchnorm2d(max_in_channels, 1, max_kernel_size)
        self.conv2d_weight, self.norm_weight, self.norm_bias = shared


class BDWRConv2dSharer(SimpleSharer):
    def _make_shared(self, max_in_channels: int, _: int) -> None:
        max_expansion_ratio = max(EXPANSION_CHOICES)
        max_kernel_size = max(KERNEL_CHOICES)
        hidden = int(round(max_in_channels * max_expansion_ratio))
        shared = shared_conv2d_batchnorm2d(
            hidden, 1, (max_kernel_size, max_kernel_size)
        )

        self.conv2d_weight, self.norm_weight, self.norm_bias = shared


def _set_shared(dst: ChoiceLayer, src: BaseModule) -> None:
    sharer = cast(SimpleSharer, dst.sharer)
    set_conv2d_batchnorm2d(
        sharer.conv2d_weight,
        sharer.norm_weight,
        sharer.norm_bias,
        src[LayerName.CONV2D]
    )


class SimpleWarmUpCtx(CommonWarmUpCtx):
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
                src = cast(BaseModule, max_net.blocks[i * max(LAYER_CHOICES) + j])

                _set_shared(layer, src)

        self._after_set(max_net)
