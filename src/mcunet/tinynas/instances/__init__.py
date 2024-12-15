from typing import cast
from collections.abc import Callable, Iterator

import torch
import torch.nn as nn

from torchvision.ops import Conv2dNormActivation

from ..searchspace import (
    KERNEL_CHOICES,
    SE_CHOICES,
    SHORTCUT_CHOICES,
    EXPANSION_CHOICES
)
from ..searchspace.layers import (
    squeeze_channels,
    LayerName,
    BaseModule,
    Conv2dModule,
    DWConv2dModule,
    BDWRConv2dModule
)
from ..oneshot import WarmUpCtx
from ..oneshot.layers import LastChoiceLayer
from ..oneshot.share import (
    SharedWeightsConv2d,
    SharedWeightsBatchNorm2d,
    SharedWeightsSqueezeExcitation,
    ParameterSharer
)
from ..oneshot.helper import (
    l1_reorder_conv2d,
    l1_reorder_batchnorm2d,
    l1_reorder_se,
    l1_reorder_main,
    has_norm
)
from ..mobilenet import MobileSkeletonNet


class SimpleSharer(ParameterSharer):
    conv2d: SharedWeightsConv2d
    batchnorm2d: SharedWeightsBatchNorm2d

    se: SharedWeightsSqueezeExcitation

    def share(self, module: BaseModule) -> None:
        self.conv2d.share(module[LayerName.CONV2D][0])

        if has_norm(module[LayerName.CONV2D]):
            self.batchnorm2d.share(module[LayerName.CONV2D][1])

        if LayerName.SE in module:
            self.se.share(module[LayerName.SE])

    def unshare(self, module: BaseModule) -> None:
        self.conv2d.unshare(module[LayerName.CONV2D][0])
        if has_norm(module[LayerName.CONV2D]):
            self.batchnorm2d.unshare(module[LayerName.CONV2D][1])

        if LayerName.SE in module:
            self.se.unshare(module[LayerName.SE])

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

    def share(self, module: BaseModule) -> None:
        if not isinstance(module, Conv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().share(module)

    def unshare(self, module: BaseModule) -> None:
        if not isinstance(module, Conv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().unshare(module)

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

    def share(self, module: BaseModule) -> None:
        if not isinstance(module, DWConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().share(module)

        self.pw_conv2d.share(module[LayerName.PWCONV2D][0])
        if has_norm(module[LayerName.PWCONV2D]):
            self.pw_batchnorm2d.share(module[LayerName.PWCONV2D][1])

    def unshare(self, module: BaseModule) -> None:
        if not isinstance(module, DWConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().unshare(module)

        self.pw_conv2d.unshare(module[LayerName.PWCONV2D][0])
        if has_norm(module[LayerName.PWCONV2D]):
            self.pw_batchnorm2d.unshare(module[LayerName.PWCONV2D][1])

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

    def share(self, module: BaseModule) -> None:
        if not isinstance(module, BDWRConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().share(module)

        self.exp_conv2d.share(module[LayerName.EXPANSION][0])
        if has_norm(module[LayerName.EXPANSION]):
            self.exp_batchnorm2d.share(module[LayerName.EXPANSION][1])

        self.pw_conv2d.share(module[LayerName.PWCONV2D][0])
        if has_norm(module[LayerName.PWCONV2D]):
            self.pw_batchnorm2d.share(module[LayerName.PWCONV2D][1])

    def unshare(self, module: BaseModule) -> None:
        if not isinstance(module, BDWRConv2dModule):
            raise TypeError(f"wrong module type {type(module)}")

        super().unshare(module)

        self.pw_conv2d.unshare(module[LayerName.EXPANSION][0])
        if has_norm(module[LayerName.EXPANSION]):
            self.pw_batchnorm2d.unshare(module[LayerName.EXPANSION][1])

        self.pw_conv2d.unshare(module[LayerName.PWCONV2D][0])
        if has_norm(module[LayerName.PWCONV2D]):
            self.pw_batchnorm2d.unshare(module[LayerName.PWCONV2D][1])

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


class LastLayerSharer(ParameterSharer):
    conv2d: SharedWeightsConv2d
    batchnorm2d: SharedWeightsBatchNorm2d

    def _make_shared(self, max_in_channels: int, max_out_channels: int) -> None:
        self.conv2d = SharedWeightsConv2d(
            (max_out_channels, max_in_channels, 1, 1), False
        )
        self.batchnorm2d = SharedWeightsBatchNorm2d(max_out_channels, True)

    def share(self, choice: Conv2dNormActivation) -> None:
        self.conv2d.share(choice[0])

        if has_norm(choice):
            self.batchnorm2d.share(choice[1])

    def unshare(self, choice: Conv2dNormActivation) -> None:
        choice[0].weight = None

        if has_norm(choice):
            batchnorm2d: nn.BatchNorm2d = choice[1]

            batchnorm2d.weight = None
            batchnorm2d.bias = None
            batchnorm2d.running_mean = None
            batchnorm2d.running_var = None
            batchnorm2d.num_batches_tracked = None

    def _weight_initialization(self) -> None:
        self.conv2d._weight_initialization()
        self.batchnorm2d._weight_initialization()

    def _weight_copy(self, module: Conv2dNormActivation):
        self.conv2d._weight_copy(module[0])
        if has_norm(module):
            self.batchnorm2d._weight_copy(module[1])


def _permutations() -> Iterator[tuple[int, float, bool]]:
    for k in KERNEL_CHOICES:
        for se in SE_CHOICES:
            for shortcut in SHORTCUT_CHOICES:
                yield (k, se, shortcut)


def conv2d_choices(
    range_in_channels: tuple[int, ...],
    range_out_channels: tuple[int, ...],
    stride: int,
    norm_layer: Callable[..., nn.Module] | None,
    activation_layer: Callable[..., nn.Module] | None
) -> list[Conv2dModule]:
    choices: list[Conv2dModule] = []

    for i in range_in_channels:
        for j in range_out_channels:
            for k, se, shortcut in _permutations():
                choices.append(Conv2dModule(
                    i, j, se, shortcut, k, stride, norm_layer, activation_layer
                ))

    return choices


def dwconv2d_choices(
    range_in_channels: tuple[int, ...],
    range_out_channels: tuple[int, ...],
    stride: int,
    norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
    activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
) -> list[DWConv2dModule]:
    choices: list[DWConv2dModule] = []

    for i in range_in_channels:
        for j in range_out_channels:
            for k, se, shortcut in _permutations():
                choices.append(DWConv2dModule(
                    i, j, se, shortcut, k, stride, norm_layer, activation_layer
                ))

    return choices


def bdwrconv2d_choices(
    range_in_channels: tuple[int, ...],
    range_out_channels: tuple[int, ...],
    stride: int,
    norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
    activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
) -> list[BDWRConv2dModule]:
    choices: list[BDWRConv2dModule] = []

    for i in range_in_channels:
        for j in range_out_channels:
            for e in EXPANSION_CHOICES:
                for k, se, shortcut in _permutations():
                    choices.append(BDWRConv2dModule(
                        i, j, e, se, shortcut, k, stride,
                        norm_layer, activation_layer
                    ))

    return choices


class CommonWarmUpCtx(WarmUpCtx):
    _first_conv2d: SharedWeightsConv2d
    _first_batchnorm2d: SharedWeightsBatchNorm2d

    _last_conv2d: SharedWeightsConv2d
    _last_batchnorm2d: SharedWeightsBatchNorm2d

    _classifier_parameters: tuple[torch.Tensor, torch.Tensor]

    def pre(self, supernet: MobileSkeletonNet, device) -> None:
        first = supernet.first
        self._first_conv2d = SharedWeightsConv2d(first[0].weight.size(), False)
        self._first_batchnorm2d = SharedWeightsBatchNorm2d(
            first[1].weight.size(), True
        )
        self._first_conv2d.to(device)
        self._first_batchnorm2d.to(device)

        with torch.no_grad():
            self._first_conv2d.weight.zero_()
            self._first_batchnorm2d.weight.zero_()
            self._first_batchnorm2d.bias.zero_()
            self._first_batchnorm2d.running_mean.zero_()
            self._first_batchnorm2d.running_var.zero_()
            self._first_batchnorm2d.num_batches_tracked.zero_()

        last = cast(LastChoiceLayer, supernet.last)
        sharer = cast(LastLayerSharer, last.sharer)
        self._last_conv2d = SharedWeightsConv2d(
            sharer.conv2d.weight.size(), False
        )
        self._last_batchnorm2d = SharedWeightsBatchNorm2d(
            sharer.batchnorm2d.weight.size(), True
        )
        self._last_conv2d.to(device)
        self._last_batchnorm2d.to(device)

        with torch.no_grad():
            self._last_conv2d.weight.zero_()
            self._last_batchnorm2d.weight.zero_()
            self._last_batchnorm2d.bias.zero_()
            self._last_batchnorm2d.running_mean.zero_()
            self._last_batchnorm2d.running_var.zero_()
            self._last_batchnorm2d.num_batches_tracked.zero_()

        classifier = supernet.classifier
        self._classifier_parameters = (
            torch.zeros_like(classifier[1].weight, device=device),
            torch.zeros_like(classifier[1].bias, device=device)
        )

    def _after_set(self, max_net: MobileSkeletonNet) -> None:
        first = max_net.first
        with torch.no_grad():
            self._first_conv2d.weight.add_(first[0].weight)
            self._first_batchnorm2d.weight.add_(first[1].weight)
            self._first_batchnorm2d.bias.add_(first[1].bias)
            self._first_batchnorm2d.running_mean.add_(first[1].running_mean)
            self._first_batchnorm2d.running_var.add_(first[1].running_var)
            self._first_batchnorm2d.num_batches_tracked.add_(
                first[1].num_batches_tracked
            )

        last = max_net.last
        with torch.no_grad():
            self._last_conv2d.weight.add_(last[0].weight)
            self._last_batchnorm2d.weight.add_(last[1].weight)
            self._last_batchnorm2d.bias.add_(last[1].bias)
            self._last_batchnorm2d.running_mean.add_(last[1].running_mean)
            self._last_batchnorm2d.running_var.add_(last[1].running_var)
            self._last_batchnorm2d.num_batches_tracked.add_(
                last[1].num_batches_tracked
            )

        classifier = max_net.classifier
        self._classifier_parameters[0].add_(classifier[1].weight)
        self._classifier_parameters[1].add_(classifier[1].bias)

    def post(self, supernet: MobileSkeletonNet) -> None:
        with torch.no_grad():
            self._first_conv2d.weight.div_(3)
            self._first_batchnorm2d.weight.div_(3)
            self._first_batchnorm2d.bias.div_(3)
            self._first_batchnorm2d.running_mean.div_(3)
            self._first_batchnorm2d.running_var.div_(3)

            self._last_conv2d.weight.div_(3)
            self._last_batchnorm2d.weight.div_(3)
            self._last_batchnorm2d.bias.div_(3)
            self._last_batchnorm2d.running_mean.div_(3)
            self._last_batchnorm2d.running_var.div_(3)

            for shared in self._classifier_parameters:
                shared.div_(3)

            first = supernet.first
            first[0].weight.copy_(self._first_conv2d.weight)
            first[1].weight.copy_(self._first_batchnorm2d.weight)
            first[1].bias.copy_(self._first_batchnorm2d.bias)
            first[1].running_mean.copy_(self._first_batchnorm2d.running_mean)
            first[1].running_var.copy_(self._first_batchnorm2d.running_var)
            first[1].num_batches_tracked.copy_(
                self._first_batchnorm2d.num_batches_tracked
            )

            out_channels, in_channels, height, width = \
                self._last_conv2d.weight.size()
            tmp = Conv2dNormActivation(
                in_channels, out_channels, (height, width)
            )
            tmp[0].weight = self._last_conv2d.weight
            tmp[1].weight = self._last_batchnorm2d.weight
            tmp[1].bias = self._last_batchnorm2d.bias
            tmp[1].running_mean = self._last_batchnorm2d.running_mean
            tmp[1].running_var = self._last_batchnorm2d.running_var
            tmp[1].num_batches_tracked = self._last_batchnorm2d.num_batches_tracked

            last = cast(LastChoiceLayer, supernet.last)
            sharer = cast(LastLayerSharer, last.sharer)
            sharer._weight_copy(tmp)

            classifier = supernet.classifier
            classifier[1].weight.copy_(self._classifier_parameters[0])
            classifier[1].bias.copy_(self._classifier_parameters[1])


def local_reorder_conv2d(
    src: BaseModule, prev_indices: torch.Tensor | None
) -> torch.Tensor | None:
    indices = l1_reorder_main(src[LayerName.CONV2D][0], prev_indices)
    if has_norm(src[LayerName.CONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.CONV2D][1], indices)

    if LayerName.SE in src:
        l1_reorder_se(src[LayerName.SE], indices)

    return indices


def local_reorder_dwconv2d(
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


def local_reorder_bdwrconv2d(
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


def global_reorder_conv2d(
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


def global_reorder_dwconv2d(
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


def global_reorder_bdwrconv2d(
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
