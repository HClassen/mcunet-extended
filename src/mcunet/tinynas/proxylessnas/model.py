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
from ..mnasnet.layers import Conv2dOp, DWConv2dOp, MBConv2dOp


class SuperProxylessNAS(MobileNetV2Skeleton):
    _switch: MixedOpSwitch

    def __init__(
        self,
        width_mult: float,
        resolution: int,
        n_classes: int,
        dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        self._switch = MixedOpSwitch()

        # all possible combinations of ConvOp, kernel size, se ratio and SkipOp
        # 36 in total
        cross = np.array(
            np.meshgrid(
                conv_choices,
                kernel_choices,
                se_choices,
                skip_choices
            )
        ).T.reshape(-1, 4)

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
                mixed: list[nn.Module] = []

                stride = s if i == 0 else 1
                in_channels = in_channels if i == 0 else out_channels

                for conv_op, kernel_size, se_ratio, skip_op in cross:
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
                                expand_ratio=t,
                                se_ratio=se_ratio,
                                skip_op=SkipOp(skip_op),
                                kernel_size=int(kernel_size),
                                stride=stride,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer
                            )
                        case _:
                            raise ValueError(f"unknown conv op: {conv_op}")

                    mixed.append(op)

                # MnasNet selects between [0, +1, -1] layers. To reflect that in
                # the MixedOp add an IdentityOp for the last two layers.
                if i == n_layers - 1 or i == n_layers:
                    mixed.append(IdentityOp())

                block_features.append(MixedOp(mixed, self._switch))

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
