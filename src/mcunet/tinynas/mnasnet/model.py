from typing import Callable

import torch.nn as nn

from .searchspace import Model, ConvOp
from .layers import Conv2dOp, DWConv2dOp, MBConv2dOp
from ..skeleton import MobileNetV2Skeleton


class MnasNet(MobileNetV2Skeleton):
    def __init__(
        self,
        model: Model,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        blocks: list[nn.Module] = []

        # Build intermediary blocks.
        for block in model.blocks:
            block_features: list[nn.Module] = []

            for i in range(block.n_layers):
                # The behavior of the first layer of a block is a bit different
                # than on the following blocks. It translates from block.in_channels
                # to block.out_channels and has a stride potentially different
                # than 1.
                if i == 0:
                    in_channels = block.in_channels
                    stride = block.first_stride
                else:
                    in_channels = block.out_channels
                    stride = 1

                match block.conv_op:
                    case ConvOp.CONV2D:
                        op = Conv2dOp(
                            in_channels,
                            block.out_channels,
                            se_ratio=block.se_ratio,
                            skip_op=block.skip_op,
                            kernel_size=block.kernel_size,
                            stride=stride,
                            norm_layer=norm_layer,
                            activation_layer=activation_layer
                        )
                    case ConvOp.DWCONV2D:
                        op = DWConv2dOp(
                            in_channels,
                            block.out_channels,
                            se_ratio=block.se_ratio,
                            skip_op=block.skip_op,
                            kernel_size=block.kernel_size,
                            stride=stride,
                            norm_layer=norm_layer,
                            activation_layer=activation_layer
                        )
                    case ConvOp.MBCONV2d:
                        op = MBConv2dOp(
                            in_channels,
                            block.out_channels,
                            expand_ratio=block.expand_ratio,
                            se_ratio=block.se_ratio,
                            skip_op=block.skip_op,
                            kernel_size=block.kernel_size,
                            stride=stride,
                            norm_layer=norm_layer,
                            activation_layer=activation_layer
                        )
                    case _:
                        raise ValueError(f"unknown conv op: {block.conv_op}")

                block_features.append(op)

            blocks.extend(block_features)

        block_in = model.blocks[0].in_channels
        block_out = model.blocks[-1].out_channels
        last_out = model.last_channels

        super().__init__(
            model.resolution,
            model.n_classes,
            blocks,
            block_in,
            block_out,
            last_out,
            model.dropout,
            norm_layer,
            activation_layer
        )
