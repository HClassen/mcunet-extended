from typing import cast
from collections.abc import Callable

import torch
import torch.nn as nn

from torchvision.ops import SqueezeExcitation

from ....searchspace import (
    KERNEL_CHOICES,
    EXPANSION_CHOICES,
    SE_CHOICES,
    LAYER_CHOICES,
    ConvOp,
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
    SharedWeightsConv2d,
    SharedWeightsBatchNorm2d,
    SharedWeightsSqueezeExcitation,
    ParameterSharer
)
from ....oneshot.helper import (
    l1_reorder_conv2d,
    l1_reorder_batchnorm2d,
    l1_reorder_se,
    l1_reorder_main,
    has_norm
)
from ....mobilenet import MobileSkeletonNet

from .. import SuperChoiceBlock, CommonWarmUpCtx


__all__ = [
    "Conv2dSharer", "DWConv2dSharer", "BDWRConv2dSharer",
    "WarmUpCtxLocalReorder", "WarmUpCtxGlobalReorder"
]


class SimpleSharer(ParameterSharer):
    conv2d: SharedWeightsConv2d
    batchnorm2d: SharedWeightsBatchNorm2d

    se: SharedWeightsSqueezeExcitation

    def set_shared(self, module: BaseModule) -> None:
        self.conv2d.share(module[LayerName.CONV2D][0])

        if has_norm(module[LayerName.CONV2D]):
            self.batchnorm2d.share(module[LayerName.CONV2D][1])

        if LayerName.SE in module:
            self.se.share(module[LayerName.SE])

    def unset_shared(self, module: BaseModule) -> None:
        module[LayerName.CONV2D][0].weight = None
        if has_norm(module[LayerName.CONV2D]):
            batchnorm2d: nn.BatchNorm2d = module[LayerName.CONV2D][1]
            batchnorm2d.weight = None
            batchnorm2d.bias = None
            batchnorm2d.running_mean = None
            batchnorm2d.running_var = None
            batchnorm2d.num_batches_tracked = None

        if LayerName.SE not in module:
            return

        se: SqueezeExcitation = module[LayerName.SE]
        se.fc1.weight = None
        se.fc1.bias = None
        se.fc2.weight = None
        se.fc2.bias = None

    def _weight_initialization(self) -> None:
        self.conv2d._weight_initialization()
        self.batchnorm2d._weight_initialization()
        self.se._weight_initialization()


class Conv2dSharer(SimpleSharer):
    def _make_shared(self, max_in_channels: int, max_out_channels: int) -> None:
        max_kernel_size = max(KERNEL_CHOICES)

        self.conv2d = SharedWeightsConv2d(
            (max_out_channels, max_in_channels, max_kernel_size, max_kernel_size),
            False
        )
        self.batchnorm2d = SharedWeightsBatchNorm2d(max_out_channels, True)

        self.se = SharedWeightsSqueezeExcitation(
            max_out_channels,
            squeeze_channels(max_out_channels, max(SE_CHOICES))
        )

    def set_shared(self, module: BaseModule) -> None:
        if not isinstance(module, Conv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().set_shared(module)

    def unset_shared(self, module: BaseModule) -> None:
        if not isinstance(module, Conv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().unset_shared(module)

    def _weight_copy(self, module: nn.Module) -> None:
        if not isinstance(module, Conv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        self.conv2d._weight_copy(module[LayerName.CONV2D][0])
        if has_norm(module[LayerName.CONV2D]):
            self.batchnorm2d._weight_copy(module[LayerName.CONV2D][1])

        if LayerName.SE in module:
            self.se._weight_copy(module[LayerName.SE])


class DWConv2dSharer(SimpleSharer):
    pw_conv2d: SharedWeightsConv2d
    pw_batchnorm2d: SharedWeightsBatchNorm2d

    def _make_shared(self, max_in_channels: int, max_out_channels: int) -> None:
        max_kernel_size = max(KERNEL_CHOICES)

        self.conv2d = SharedWeightsConv2d(
            (max_in_channels, 1, max_kernel_size, max_kernel_size), False
        )
        self.batchnorm2d = SharedWeightsBatchNorm2d(max_in_channels, True)

        self.se = SharedWeightsSqueezeExcitation(
            max_in_channels, squeeze_channels(max_in_channels, max(SE_CHOICES))
        )

        self.pw_conv2d = SharedWeightsConv2d(
            (max_out_channels, max_in_channels, 1, 1), False
        )
        self.pw_batchnorm2d = SharedWeightsBatchNorm2d(max_out_channels, True)

    def set_shared(self, module: BaseModule) -> None:
        if not isinstance(module, DWConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().set_shared(module)

        self.pw_conv2d.share(module[LayerName.PWCONV2D][0])
        if has_norm(module[LayerName.PWCONV2D]):
            self.pw_batchnorm2d.share(module[LayerName.PWCONV2D][1])

    def unset_shared(self, module: BaseModule) -> None:
        if not isinstance(module, DWConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().unset_shared(module)

        module[LayerName.PWCONV2D][0].weight = None
        if has_norm(module[LayerName.PWCONV2D]):
            batchnorm2d: nn.BatchNorm2d = module[LayerName.PWCONV2D][1]

            batchnorm2d.weight = None
            batchnorm2d.bias = None
            batchnorm2d.running_mean = None
            batchnorm2d.running_var = None
            batchnorm2d.num_batches_tracked = None

    def _weight_initialization(self) -> None:
        super()._weight_initialization()

        self.pw_conv2d._weight_initialization()
        self.pw_batchnorm2d._weight_initialization()

    def _weight_copy(self, module: nn.Module) -> None:
        if not isinstance(module, DWConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        self.conv2d._weight_copy(module[LayerName.CONV2D][0])
        if has_norm(module[LayerName.CONV2D]):
            self.batchnorm2d._weight_copy(module[LayerName.CONV2D][1])

        if LayerName.SE in module:
            self.se._weight_copy(module[LayerName.SE])

        self.pw_conv2d._weight_copy(module[LayerName.PWCONV2D][0])
        if has_norm(module[LayerName.PWCONV2D]):
            self.pw_batchnorm2d._weight_copy(module[LayerName.PWCONV2D][1])


class BDWRConv2dSharer(SimpleSharer):
    exp_conv2d: SharedWeightsConv2d
    exp_batchnorm2d: SharedWeightsBatchNorm2d

    pw_conv2d: SharedWeightsConv2d
    pw_batchnorm2d: SharedWeightsBatchNorm2d

    def _make_shared(self, max_in_channels: int, max_out_channels: int) -> None:
        max_expansion_ratio = max(EXPANSION_CHOICES)
        max_kernel_size = max(KERNEL_CHOICES)

        hidden = int(round(max_in_channels * max_expansion_ratio))
        self.exp_conv2d = SharedWeightsConv2d(
            (hidden, max_in_channels, 1, 1), False
        )
        self.exp_batchnorm2d = SharedWeightsBatchNorm2d(hidden, True)

        self.conv2d = SharedWeightsConv2d(
            (hidden, 1, max_kernel_size, max_kernel_size), False
        )
        self.batchnorm2d = SharedWeightsBatchNorm2d(hidden, True)

        self.se = SharedWeightsSqueezeExcitation(
            hidden, squeeze_channels(hidden, max(SE_CHOICES))
        )

        self.pw_conv2d = SharedWeightsConv2d(
            (max_out_channels, hidden, 1, 1), False
        )
        self.pw_batchnorm2d = SharedWeightsBatchNorm2d(max_out_channels, True)

    def set_shared(self, module: BaseModule) -> None:
        if not isinstance(module, BDWRConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().set_shared(module)

        self.exp_conv2d.share(module[LayerName.EXPANSION][0])
        if has_norm(module[LayerName.EXPANSION]):
            self.exp_batchnorm2d.share(module[LayerName.EXPANSION][1])

        self.pw_conv2d.share(module[LayerName.PWCONV2D][0])
        if has_norm(module[LayerName.PWCONV2D]):
            self.pw_batchnorm2d.share(module[LayerName.PWCONV2D][1])

    def unset_shared(self, module: BaseModule) -> None:
        if not isinstance(module, BDWRConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().unset_shared(module)

        module[LayerName.EXPANSION][0].weight = None
        if has_norm(module[LayerName.EXPANSION]):
            batchnorm2d: nn.BatchNorm2d = module[LayerName.EXPANSION][1]

            batchnorm2d.weight = None
            batchnorm2d.bias = None
            batchnorm2d.running_mean = None
            batchnorm2d.running_var = None
            batchnorm2d.num_batches_tracked = None

        module[LayerName.PWCONV2D][0].weight = None
        if has_norm(module[LayerName.PWCONV2D]):
            batchnorm2d: nn.BatchNorm2d = module[LayerName.PWCONV2D][1]

            batchnorm2d.weight = None
            batchnorm2d.bias = None
            batchnorm2d.running_mean = None
            batchnorm2d.running_var = None
            batchnorm2d.num_batches_tracked = None

    def _weight_initialization(self) -> None:
        super()._weight_initialization()

        self.exp_conv2d._weight_initialization()
        self.exp_batchnorm2d._weight_initialization()

        self.pw_conv2d._weight_initialization()
        self.pw_batchnorm2d._weight_initialization()

    def _weight_copy(self, module: BaseModule) -> None:
        if not isinstance(module, BDWRConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        self.exp_conv2d._weight_copy(module[LayerName.EXPANSION][0])
        if has_norm(module[LayerName.EXPANSION]):
            self.exp_batchnorm2d._weight_copy(module[LayerName.EXPANSION][1])

        self.conv2d._weight_copy(module[LayerName.CONV2D][0])
        if has_norm(module[LayerName.CONV2D]):
            self.batchnorm2d._weight_copy(module[LayerName.CONV2D][1])

        if LayerName.SE in module:
            self.se._weight_copy(module[LayerName.SE])

        self.pw_conv2d._weight_copy(module[LayerName.PWCONV2D][0])
        if has_norm(module[LayerName.PWCONV2D]):
            self.pw_batchnorm2d._weight_copy(module[LayerName.PWCONV2D][1])


def _local_reorder_conv2d(
    src: BaseModule, prev_indices: torch.Tensor | None
) -> torch.Tensor | None:
    indices = l1_reorder_main(src[LayerName.CONV2D][0], prev_indices)
    if has_norm(src[LayerName.CONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.CONV2D][1], indices)

    if LayerName.SE in src:
        l1_reorder_se(src[LayerName.SE], indices)

    return indices


def _local_reorder_dwconv2d(
    src: BaseModule, prev_indices: torch.Tensor | None
) -> torch.Tensor | None:
    l1_reorder_conv2d(src[LayerName.CONV2D][0], 0, prev_indices)
    if has_norm(src[LayerName.CONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.CONV2D][1], prev_indices)

    if LayerName.SE in src:
        l1_reorder_se(src[LayerName.SE], prev_indices)

    indices = l1_reorder_main(src[LayerName.PWCONV2D][0], prev_indices)
    if has_norm(src[LayerName.PWCONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.PWCONV2D][1], indices)

    return indices


def _local_reorder_bdwrconv2d(
    src: BaseModule, prev_indices: torch.Tensor | None
) -> torch.Tensor | None:
    indices = l1_reorder_main(src[LayerName.EXPANSION][0], prev_indices)
    if has_norm(src[LayerName.EXPANSION]):
        l1_reorder_batchnorm2d(src[LayerName.EXPANSION][1], indices)

    l1_reorder_conv2d(src[LayerName.CONV2D][0], 0, indices)
    if has_norm(src[LayerName.CONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.CONV2D][1], indices)

    if LayerName.SE in src:
        l1_reorder_se(src[LayerName.SE], indices)

    indices = l1_reorder_main(src[LayerName.PWCONV2D][0], indices)
    if has_norm(src[LayerName.PWCONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.PWCONV2D][1], indices)

    return indices


class WarmUpCtxLocalReorder(CommonWarmUpCtx):
    _reorder: dict[
        ConvOp,
        Callable[[BaseModule, torch.Tensor | None], torch.Tensor | None]
    ]

    def __init__(self) -> None:
        self._reorder = {
            ConvOp.CONV2D: _local_reorder_conv2d,
            ConvOp.DWCONV2D: _local_reorder_dwconv2d,
            ConvOp.BDWRCONV2D: _local_reorder_bdwrconv2d
        }

    def set(
        self,
        supernet: MobileSkeletonNet,
        max_model: Model,
        max_net: MobileSkeletonNet
    ) -> None:
        op = max_model.blocks[0].layers[0].op

        indices = None
        for i, block in enumerate(supernet.blocks):
            block = cast(SuperChoiceBlock, block)

            choice = cast(ChoiceBlock, block.choices[op])
            for j, layer in enumerate(choice.layers):
                layer = cast(ChoiceLayer, layer)
                src = cast(
                    BaseModule, max_net.blocks[i * max(LAYER_CHOICES) + j]
                )

                indices = self._reorder[op](src, indices)
                layer.sharer._weight_copy(src)

        l1_reorder_conv2d(max_net.last[0], 1, indices)
        self._after_set(max_net)


def _global_reorder_conv2d(
    src: BaseModule,
    prev_indices: torch.Tensor | None,
    next_indices: torch.Tensor | None
) -> torch.Tensor | None:
    l1_reorder_conv2d(src[LayerName.CONV2D][0], 1, prev_indices)
    with torch.no_grad():
        reordered = torch.index_select(
            src[LayerName.CONV2D][0].weight, dim=0, index=next_indices
        )

        src[LayerName.CONV2D][0].weight = nn.Parameter(reordered)

    if has_norm(src[LayerName.CONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.CONV2D][1], next_indices)

    if LayerName.SE in src:
        l1_reorder_se(src[LayerName.SE], next_indices)

    return next_indices


def _global_reorder_dwconv2d(
    src: BaseModule,
    prev_indices: torch.Tensor | None,
    next_indices: torch.Tensor | None
) -> torch.Tensor | None:
    l1_reorder_conv2d(src[LayerName.CONV2D][0], 0, prev_indices)
    if has_norm(src[LayerName.CONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.CONV2D][1], prev_indices)

    if LayerName.SE in src:
        l1_reorder_se(src[LayerName.SE], prev_indices)

    l1_reorder_conv2d(src[LayerName.PWCONV2D][0], 1, prev_indices)
    with torch.no_grad():
        reordered = torch.index_select(
            src[LayerName.PWCONV2D][0].weight, dim=0, index=next_indices
        )

        src[LayerName.PWCONV2D][0].weight = nn.Parameter(reordered)

    if has_norm(src[LayerName.PWCONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.PWCONV2D][1], next_indices)

    return next_indices


def _global_reorder_bdwrconv2d(
    src: BaseModule,
    prev_indices: torch.Tensor | None,
    next_indices: torch.Tensor | None
) -> torch.Tensor | None:
    indices = l1_reorder_main(src[LayerName.EXPANSION][0], prev_indices)
    if has_norm(src[LayerName.EXPANSION]):
        l1_reorder_batchnorm2d(src[LayerName.EXPANSION][1], indices)

    l1_reorder_conv2d(src[LayerName.CONV2D][0], 0, indices)
    if has_norm(src[LayerName.CONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.CONV2D][1], indices)

    if LayerName.SE in src:
        l1_reorder_se(src[LayerName.SE], indices)

    l1_reorder_conv2d(src[LayerName.PWCONV2D][0], 1, indices)
    with torch.no_grad():
        reordered = torch.index_select(
            src[LayerName.PWCONV2D][0].weight, dim=0, index=next_indices
        )

        src[LayerName.PWCONV2D][0].weight = nn.Parameter(reordered)

    if has_norm(src[LayerName.PWCONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.PWCONV2D][1], next_indices)

    return next_indices


class WarmUpCtxGlobalReorder(CommonWarmUpCtx):
    _reorder: dict[
        ConvOp,
        Callable[
            [BaseModule, torch.Tensor | None, torch.Tensor | None],
            torch.Tensor | None
        ]
    ]
    _norms: list[torch.Tensor]

    def __init__(self, norms: list[torch.Tensor]) -> None:
        self._reorder = {
            ConvOp.CONV2D: _global_reorder_conv2d,
            ConvOp.DWCONV2D: _global_reorder_dwconv2d,
            ConvOp.BDWRCONV2D: _global_reorder_bdwrconv2d
        }
        self._norms = norms

    def set(
        self,
        supernet: MobileSkeletonNet,
        max_model: Model,
        max_net: MobileSkeletonNet
    ) -> None:
        op = max_model.blocks[0].layers[0].op

        indices = None
        for i, block in enumerate(supernet.blocks):
            block = cast(SuperChoiceBlock, block)

            choice = cast(ChoiceBlock, block.choices[op])
            for j, layer in enumerate(choice.layers):
                layer = cast(ChoiceLayer, layer)

                idx = i * max(LAYER_CHOICES) + j
                src = cast(BaseModule, max_net.blocks[idx])

                indices = self._reorder[op](src, indices, self._norms[idx])
                layer.sharer._weight_copy(src)

        l1_reorder_conv2d(max_net.last[0], 1, indices)
        self._after_set(max_net)
