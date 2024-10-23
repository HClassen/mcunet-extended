from typing import Callable

import torch.nn as nn

import numpy as np

from ..utils import make_divisible

from ..skeleton import MobileNetV2Skeleton

from .layers import IdentityOp, MixedOpSwitch, MixedOp

from ..mnasnet.searchspace import (
    layer_settings,
    layer_choices,
    conv_choices,
    kernel_choices,
    se_choices,
    skip_choices,
    ConvOp,
    SkipOp
)
from ..mnasnet.layers import BaseOp, Conv2dOp, DWConv2dOp, MBConv2dOp


# all possible combinations of in_channels/out_channels multiplier
# 2187 in total
_layer_combinations: np.ndarray = np.array(
    np.meshgrid(
        layer_choices,
        layer_choices,
        layer_choices,
        layer_choices,
        layer_choices,
        layer_choices,
        layer_choices
    )
).T.reshape(-1, 7)


# all possible combinations of ConvOp, kernel size, se ratio and SkipOp
# 36 in total
_op_combinations: np.ndarray = np.array(
    np.meshgrid(
        conv_choices,
        kernel_choices,
        se_choices,
        skip_choices
    )
).T.reshape(-1, 4)


def _build_layer(
    in_channels: int,
    out_channels: int,
    stride: int,
    expand_ratio: int,
    combinations: np.ndarray,
    norm_layer: Callable[..., nn.Module],
    activation_layer: Callable[..., nn.Module]
) -> list[BaseOp]:
    layer: list[nn.Module] = []

    for conv_op, kernel_size, se_ratio, skip_op in combinations:
        match ConvOp(conv_op):
            case ConvOp.CONV2D:
                op = Conv2dOp(
                    in_channels,
                    out_channels,
                    se_ratio=se_ratio,
                    skip_op=SkipOp(skip_op),
                    kernel_size=int(kernel_size),
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer
                )
            case ConvOp.DWCONV2D:
                op = DWConv2dOp(
                    in_channels,
                    out_channels,
                    se_ratio=se_ratio,
                    skip_op=SkipOp(skip_op),
                    kernel_size=int(kernel_size),
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer
                )
            case ConvOp.MBCONV2d:
                op = MBConv2dOp(
                    in_channels,
                    out_channels,
                    expand_ratio=expand_ratio,
                    se_ratio=se_ratio,
                    skip_op=SkipOp(skip_op),
                    kernel_size=int(kernel_size),
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer
                )
            case _:
                raise ValueError(f"unknown conv op: {conv_op}")

        layer.append(op)

    return layer


class SuperProxylessNAS(MobileNetV2Skeleton):
    _switch: MixedOpSwitch

    def __init__(
        self,
        width_mult: float,
        layer_width_mult: list[float],
        resolution: int,
        n_classes: int,
        dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        if len(layer_width_mult) != len(layer_settings):
            raise ValueError(
                "length of layer width mult invalid",
                len(layer_settings),
                len(layer_width_mult)
            )
        
        self._switch = MixedOpSwitch()

        round_nearest = 8
        in_channels = make_divisible(32 * width_mult, round_nearest)

        block_in = in_channels

        blocks: list[MixedOp] = []
        for t, c, n, s in layer_settings:
            block_features: list[MixedOp] = []

            out_channels = make_divisible(
                c * width_mult, round_nearest
            )

            n_layers = n + max(layer_choices)
            for i in range(n_layers):
                stride = s if i == 0 else 1
                
                layer = _build_layer(
                    in_channels,
                    out_channels,
                    stride,
                    t,
                    _op_combinations,
                    norm_layer,
                    activation_layer
                )

                in_channels = out_channels

                # MnasNet selects between [0, +1, -1] layers. To reflect that in
                # the MixedOp add an IdentityOp for the last two layers.
                if i == n_layers - 1 or i == n_layers:
                    layer.append(IdentityOp())

                block_features.append(MixedOp(layer, self._switch))

            blocks.extend(block_features)

        block_out = out_channels
        last_out = make_divisible(1280 * max(1.0, width_mult), round_nearest)

        super().__init__(
            resolution,
            n_classes,
            blocks,
            block_in,
            block_out,
            last_out,
            dropout,
            norm_layer,
            activation_layer
        )


class SuperMixedOp(MixedOp):
    candidates: list[SuperProxylessNAS]

    def __init__(
        self, candidates: list[MixedOp], switch: MixedOpSwitch
    ) -> None:
        if not all([isinstance(x, MixedOp) for x in candidates]):
            raise ValueError("candidates must be all MixedOp")
        
        super().__init__(candidates, switch)

    def set_chosen_candidate(self) -> None:
        super().set_chosen_candidate()

        for candidate in self.candidates:
            candidate.set_chosen_candidate()

    def binarize(self) -> None:
        super().binarize()

        for candiate in self.candidates:
            candiate.binarize()

    def set_arch_param_grad(self) -> None:
        super().set_arch_param_grad()

        for candidate in self.candidates:
            candidate.set_arch_param_grad()


class SuperSuperProxylessNAS(nn.Module):

