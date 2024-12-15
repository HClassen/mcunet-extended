from typing import cast

import torch.nn as nn

from mcunet.tinynas.searchspace import LAYER_CHOICES, Model
from mcunet.tinynas.searchspace.layers import LayerName, BaseOp, Conv2dOp
from mcunet.tinynas.oneshot.customize import ChoiceLayer, ChoiceBlock
from mcunet.tinynas.oneshot.share import (
    share_conv2d,
    share_batchnorm2d,
    shared_conv2d_batchnorm2d,
    init_conv2d_batchnorm2d,
    set_conv2d_batchnorm2d,
    ParameterSharer
)
from mcunet.tinynas.oneshot.helper import has_norm
from mcunet.tinynas.mobilenet import MobileSkeletonNet

from . import SuperChoiceBlock, CommonWarmUpSetter


__all__ = [
    "Conv2dSharer", "DWConv2dSharer", "BDWRConv2dSharer", "FullWarmUpSetter"
]


class SimpleSharer(ParameterSharer):
    conv2d_weight: nn.Parameter
    norm_weight: nn.Parameter
    norm_bias: nn.Parameter

    def set_shared(self, module: Conv2dOp) -> None:
        share_conv2d(module[LayerName.CONV2D][0], self.conv2d_weight, None)

        if has_norm(module[LayerName.CONV2D]):
            share_batchnorm2d(
                module[LayerName.CONV2D][1], self.norm_weight, self.norm_bias,
                None, None, None
            )

    def unset_shared(self, module: Conv2dOp) -> None:
        module[LayerName.CONV2D][0].weight = None
        if has_norm(module[LayerName.CONV2D]):
            module[LayerName.CONV2D][1].weight = None
            module[LayerName.CONV2D][1].bias = None

    def _weight_initialization(self) -> None:
        init_conv2d_batchnorm2d(
            self.conv2d_weight, self.norm_weight, self.norm_bias
        )


class Conv2dSharer(SimpleSharer):
    def make_shared(self, **kwargs) -> None:
        out_channels = kwargs["out_channels"]
        in_channels = kwargs["in_channels"]
        kernel_size = kwargs["kernel_size"]

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.conv2d_weight, self.norm_weight, self.norm_bias = \
            shared_conv2d_batchnorm2d(out_channels, in_channels, kernel_size)


class DWConv2dSharer(SimpleSharer):
    pwconv2d_weight: nn.Parameter
    pwnorm_weight: nn.Parameter
    pwnorm_bias: nn.Parameter

    def make_shared(self, **kwargs) -> None:
        out_channels = kwargs["out_channels"]
        in_channels = kwargs["in_channels"]
        kernel_size = kwargs["kernel_size"]

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.conv2d_weight, self.norm_weight, self.norm_bias = \
            shared_conv2d_batchnorm2d(in_channels, 1, kernel_size)

        self.pwconv2d_weight, self.pwnorm_weight, self.pwnorm_bias = \
            shared_conv2d_batchnorm2d(out_channels, in_channels, (1, 1))

    def set_shared(self, module: Conv2dOp) -> None:
        super().set_shared(module)

        share_conv2d(module[LayerName.PWCONV2D][0], self.pwconv2d_weight, None)
        if has_norm(module[LayerName.PWCONV2D]):
            share_batchnorm2d(
                module[LayerName.PWCONV2D][1], self.pwnorm_weight, self.pwnorm_bias,
                None, None, None
            )

    def unset_shared(self, module: Conv2dOp) -> None:
        super().unset_shared(module)

        module[LayerName.PWCONV2D][0].weight = None
        if has_norm(module[LayerName.PWCONV2D]):
            module[LayerName.PWCONV2D][1].weight = None
            module[LayerName.PWCONV2D][1].bias = None

    def _weight_initialization(self) -> None:
        super()._weight_initialization()

        init_conv2d_batchnorm2d(
            self.pwconv2d_weight, self.pwnorm_weight, self.pwnorm_bias
        )


class BDWRConv2dSharer(SimpleSharer):
    expansion_weight: nn.Parameter
    expansion_norm_weight: nn.Parameter
    expansion_norm_bias: nn.Parameter

    pwconv2d_weight: nn.Parameter
    pwnorm_weight: nn.Parameter
    pwnorm_bias: nn.Parameter

    def make_shared(self, **kwargs) -> None:
        out_channels = kwargs["out_channels"]
        in_channels = kwargs["in_channels"]
        kernel_size = kwargs["kernel_size"]
        expansion_ratio = kwargs["expansion_ratio"]

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        hidden = int(round(in_channels * expansion_ratio))
        self.expansion_weight, self.expansion_norm_weight, self.expansion_norm_bias = \
            shared_conv2d_batchnorm2d(hidden, in_channels, (1, 1))

        self.conv2d_weight, self.norm_weight, self.norm_bias = \
            shared_conv2d_batchnorm2d(hidden, 1, kernel_size)

        self.pwconv2d_weight, self.pwnorm_weight, self.pwnorm_bias = \
            shared_conv2d_batchnorm2d(out_channels, hidden, (1, 1))

    def set_shared(self, module: Conv2dOp) -> None:
        super().set_shared(module)

        share_conv2d(module[LayerName.EXPANSION][0], self.expansion_weight, None)
        if has_norm(module[LayerName.EXPANSION]):
            share_batchnorm2d(
                module[LayerName.EXPANSION][1],
                self.expansion_norm_weight,
                self.expansion_norm_bias,
                None, None, None
            )

        share_conv2d(module[LayerName.PWCONV2D][0], self.pwconv2d_weight, None)
        if has_norm(module[LayerName.PWCONV2D]):
            share_batchnorm2d(
                module[LayerName.PWCONV2D][1], self.pwnorm_weight, self.pwnorm_bias,
                None, None, None
            )

    def unset_shared(self, module: Conv2dOp) -> None:
        super().unset_shared(module)

        module[LayerName.EXPANSION][0].weight = None
        if has_norm(module[LayerName.EXPANSION]):
            module[LayerName.EXPANSION][1].weight = None
            module[LayerName.EXPANSION][1].bias = None

        module[LayerName.PWCONV2D][0].weight = None
        if has_norm(module[LayerName.PWCONV2D]):
            module[LayerName.PWCONV2D][1].weight = None
            module[LayerName.PWCONV2D][1].bias = None

    def _weight_initialization(self) -> None:
        super()._weight_initialization()

        init_conv2d_batchnorm2d(
            self.expansion_weight,
            self.expansion_norm_weight,
            self.expansion_norm_bias
        )

        init_conv2d_batchnorm2d(
            self.pwconv2d_weight, self.pwnorm_weight, self.pwnorm_bias
        )


def _set_shared(dst: ChoiceLayer, src: BaseOp) -> None:
    sharer = cast(SimpleSharer, dst.sharer)
    set_conv2d_batchnorm2d(
        sharer.conv2d_weight,
        sharer.norm_weight,
        sharer.norm_bias,
        src[LayerName.CONV2D]
    )


def _set_expansion(dst: ChoiceLayer, src: BaseOp) -> None:
    if (
        not isinstance(dst.sharer, BDWRConv2dSharer)
        or LayerName.EXPANSION not in src
    ):
        return

    sharer = cast(BDWRConv2dSharer, dst.sharer)
    set_conv2d_batchnorm2d(
        sharer.expansion_weight,
        sharer.expansion_norm_weight,
        sharer.expansion_norm_bias,
        src[LayerName.EXPANSION]
    )


def _set_pwconv2d(dst: ChoiceLayer, src: BaseOp) -> None:
    if (
        not isinstance(dst.sharer, (DWConv2dSharer, BDWRConv2dSharer))
        or LayerName.PWCONV2D not in src
    ):
        return

    sharer = cast(DWConv2dSharer, dst.sharer)
    set_conv2d_batchnorm2d(
        sharer.pwconv2d_weight,
        sharer.pwnorm_weight,
        sharer.pwnorm_bias,
        src[LayerName.PWCONV2D]
    )


class FullWarmUpSetter(CommonWarmUpSetter):
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

                _set_shared(layer, src)
                _set_expansion(layer, src)
                _set_pwconv2d(layer, src)

        self._after_set(max_net)
