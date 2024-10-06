from collections.abc import Callable

import torch.nn as nn

from ..mobilenet import (
    DROPOUT,
    build_first,
    build_last,
    build_pool,
    build_classifier,
    MobileSkeletonNet
)

from . import Model, ConvOp, SkipOp
from .layers import BaseOp, Conv2dOp, DWConv2dOp, MBConv2dOp


__all__ = ["build_model"]


def _build_op(
    op: ConvOp,
    in_channels: int,
    out_channels: int,
    expansion_ratio: int,
    se_ratio: float,
    skip_op: SkipOp,
    kernel_size: int,
    stride: int,
    norm_layer: Callable[..., nn.Module] | None,
    activation_layer: Callable[..., nn.Module] | None
) -> BaseOp:
    match op:
        case ConvOp.CONV2D:
            return Conv2dOp(
                in_channels,
                out_channels,
                se_ratio,
                skip_op == SkipOp.IDENTITY,
                kernel_size,
                stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        case ConvOp.DWCONV2D:
            return DWConv2dOp(
                in_channels,
                out_channels,
                se_ratio,
                skip_op == SkipOp.IDENTITY,
                kernel_size,
                stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        case ConvOp.MBCONV2D:
            return MBConv2dOp(
                in_channels,
                out_channels,
                expansion_ratio,
                se_ratio,
                skip_op == SkipOp.IDENTITY,
                kernel_size,
                stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )
        case _:
            raise ValueError(f"unknown convolution operation: {op}")


def build_model(
    model: Model,
    classes: int,
    dropout: float = DROPOUT,
    norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
    activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
) -> MobileSkeletonNet:
    """
    Build a ``MobileSkeletonNet`` from a given ``Model``.

    Args:
        model (Model):
            A model sampled from the MnasNet search space.
        classes (int).
            The amout of classes for the classifier to recognize.
        dropout (float):
            The percentage of dropout used in the classifier.
        norm_layer (Callable[..., nn.Module], None):
            The constructor for the norm layer.
        activation_layer (Callable[..., nn.Module], None):
            The constructor for the activation layer.

    Returns:
        MobileSkeletonNet:
            The created Pytorch model.
    """
    in_channels = model.blocks[0].in_channels

    first = build_first(in_channels, norm_layer, activation_layer)

    blocks: list[BaseOp] = []
    for block in model.blocks:
        for i in range(block.n_layers):
            # The behavior of the first layer of a block is a bit different
            # than on the following blocks. It translates from block.in_channels
            # to block.out_channels and has a stride potentially different
            # than 1.
            stride = block.first_stride if i == 0 else 1

            blocks.append(
                _build_op(
                    block.conv_op,
                    in_channels,
                    block.out_channels,
                    block.expansion_ratio,
                    block.se_ratio,
                    block.skip_op,
                    block.kernel_size,
                    stride,
                    norm_layer,
                    activation_layer
                )
            )

            in_channels = block.out_channels

    last = build_last(
        in_channels, norm_layer=norm_layer, activation_layer=activation_layer
    )

    pool = build_pool()
    classifier = build_classifier(classes=classes, dropout=dropout)

    return MobileSkeletonNet(first, blocks, last, pool, classifier)
