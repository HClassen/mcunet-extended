from typing import cast

import torch.nn as nn

from torchvision.ops import SqueezeExcitation

from ....searchspace import (
    KERNEL_CHOICES,
    EXPANSION_CHOICES,
    SE_CHOICES,
    LAYER_CHOICES,
    Model
)
from ....searchspace.layers import (
    squeeze_channels,
    LayerName,
    BaseModule,
    Conv2dModule,
    DWConv2dModule,
    BDWRConv2dModule
)
from ....oneshot.layers import ChoiceLayer, ChoiceBlock
from ....oneshot.share import (
    share_conv2d,
    share_batchnorm2d,
    make_shared_conv2d,
    make_shared_conv2d_batchnorm2d,
    init_conv2d,
    init_conv2d_batchnorm2d,
    set_conv2d,
    set_conv2d_batchnorm2d,
    ParameterSharer
)
from ....oneshot.helper import has_norm
from ....mobilenet import MobileSkeletonNet

from .. import SuperChoiceBlock, CommonWarmUpCtx


__all__ = [
    "Conv2dSharer", "DWConv2dSharer", "BDWRConv2dSharer", "FullWarmUpCtx"
]

def _make_shared_se(
    in_channels: int
) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter]:
    max_se = max(SE_CHOICES)
    se_channels = squeeze_channels(in_channels, max_se)

    shared = make_shared_conv2d(se_channels, in_channels, (1, 1))
    fc1_weight, fc1_bias = shared

    shared = make_shared_conv2d(in_channels, se_channels, (1, 1))
    fc2_weight, fc2_bias = shared

    return fc1_weight, fc1_bias, fc2_weight, fc2_bias


class SimpleSharer(ParameterSharer):
    conv2d_weight: nn.Parameter
    norm_weight: nn.Parameter
    norm_bias: nn.Parameter

    fc1_weight: nn.Parameter
    fc1_bias: nn.Parameter

    fc2_weight: nn.Parameter
    fc2_bias: nn.Parameter

    def set_shared(self, module: BaseModule) -> None:
        share_conv2d(module[LayerName.CONV2D][0], self.conv2d_weight, None)

        if has_norm(module[LayerName.CONV2D]):
            share_batchnorm2d(
                module[LayerName.CONV2D][1], self.norm_weight, self.norm_bias,
                None, None, None
            )

        if LayerName.SE not in module:
            return

        se: SqueezeExcitation = module[LayerName.SE]
        share_conv2d(se.fc1, self.fc1_weight, self.fc1_bias)
        share_conv2d(se.fc2, self.fc2_weight, self.fc2_bias)

    def unset_shared(self, module: BaseModule) -> None:
        module[LayerName.CONV2D][0].weight = None
        if has_norm(module[LayerName.CONV2D]):
            module[LayerName.CONV2D][1].weight = None
            module[LayerName.CONV2D][1].bias = None

        if LayerName.SE not in module:
            return

        se: SqueezeExcitation = module[LayerName.SE]
        se.fc1.weight = None
        se.fc1.bias = None
        se.fc2.weight = None
        se.fc2.bias = None

    def _weight_initialization(self) -> None:
        init_conv2d_batchnorm2d(
            self.conv2d_weight, self.norm_weight, self.norm_bias
        )

        init_conv2d(self.fc1_weight, self.fc1_bias)
        init_conv2d(self.fc2_weight, self.fc2_bias)


class Conv2dSharer(SimpleSharer):
    def _make_shared(self, max_in_channels: int, max_out_channels: int) -> None:
        max_kernel_size = (max(KERNEL_CHOICES), max(KERNEL_CHOICES))

        shared = make_shared_conv2d_batchnorm2d(
            max_out_channels, max_in_channels, max_kernel_size
        )
        self.conv2d_weight, self.norm_weight, self.norm_bias = shared

        shared = _make_shared_se(max_out_channels)
        self.fc1_weight, self.fc1_bias, self.fc2_weight, self.fc2_bias = shared

    def set_shared(self, module: BaseModule) -> None:
        if not isinstance(module, Conv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().set_shared(module)

    def unset_shared(self, module: BaseModule) -> None:
        if not isinstance(module, Conv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().unset_shared(module)


class DWConv2dSharer(SimpleSharer):
    pwconv2d_weight: nn.Parameter
    pwnorm_weight: nn.Parameter
    pwnorm_bias: nn.Parameter

    def _make_shared(self, max_in_channels: int, max_out_channels: int) -> None:
        max_kernel_size = (max(KERNEL_CHOICES), max(KERNEL_CHOICES))

        shared = make_shared_conv2d_batchnorm2d(max_in_channels, 1, max_kernel_size)
        self.conv2d_weight, self.norm_weight, self.norm_bias = shared

        shared = make_shared_conv2d_batchnorm2d(
            max_out_channels, max_in_channels, (1, 1)
        )
        self.pwconv2d_weight, self.pwnorm_weight, self.pwnorm_bias = shared

        shared = _make_shared_se(max_in_channels)
        self.fc1_weight, self.fc1_bias, self.fc2_weight, self.fc2_bias = shared

    def set_shared(self, module: BaseModule) -> None:
        if not isinstance(module, DWConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().set_shared(module)

        share_conv2d(module[LayerName.PWCONV2D][0], self.pwconv2d_weight, None)
        if has_norm(module[LayerName.PWCONV2D]):
            share_batchnorm2d(
                module[LayerName.PWCONV2D][1], self.pwnorm_weight, self.pwnorm_bias,
                None, None, None
            )

    def unset_shared(self, module: BaseModule) -> None:
        if not isinstance(module, DWConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

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

    def _make_shared(self, max_in_channels: int, max_out_channels: int) -> None:
        max_expansion_ratio = max(EXPANSION_CHOICES)
        max_kernel_size = (max(KERNEL_CHOICES), max(KERNEL_CHOICES))

        hidden = int(round(max_in_channels * max_expansion_ratio))
        shared = make_shared_conv2d_batchnorm2d(hidden, max_in_channels, (1, 1))
        self.expansion_weight, \
        self.expansion_norm_weight, \
        self.expansion_norm_bias = shared

        shared = make_shared_conv2d_batchnorm2d(hidden, 1, max_kernel_size)
        self.conv2d_weight, self.norm_weight, self.norm_bias = shared

        shared = make_shared_conv2d_batchnorm2d(max_out_channels, hidden, (1, 1))
        self.pwconv2d_weight, self.pwnorm_weight, self.pwnorm_bias = shared

        shared = _make_shared_se(hidden)
        self.fc1_weight, self.fc1_bias, self.fc2_weight, self.fc2_bias = shared

    def set_shared(self, module: BaseModule) -> None:
        if not isinstance(module, BDWRConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

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

    def unset_shared(self, module: BaseModule) -> None:
        if not isinstance(module, BDWRConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

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


def _set_shared(dst: ChoiceLayer, src: BaseModule) -> None:
    sharer = cast(SimpleSharer, dst.sharer)
    set_conv2d_batchnorm2d(
        sharer.conv2d_weight,
        sharer.norm_weight,
        sharer.norm_bias,
        src[LayerName.CONV2D]
    )


def _set_expansion(dst: ChoiceLayer, src: BaseModule) -> None:
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


def _set_pwconv2d(dst: ChoiceLayer, src: BaseModule) -> None:
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


def _set_se(dst: ChoiceLayer, src: BaseModule) -> None:
    if LayerName.SE not in src:
        return

    se: SqueezeExcitation = src[LayerName.SE]
    sharer = cast(SimpleSharer, dst.sharer)
    set_conv2d(sharer.fc1_weight, sharer.fc1_bias, se.fc1)
    set_conv2d(sharer.fc2_weight, sharer.fc2_bias, se.fc2)


class FullWarmUpCtx(CommonWarmUpCtx):
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
                _set_expansion(layer, src)
                _set_pwconv2d(layer, src)
                _set_se(layer, src)

        self._after_set(max_net)
