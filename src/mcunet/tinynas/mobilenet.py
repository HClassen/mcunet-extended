import time
from typing import Final
from functools import cache
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.ops import Conv2dNormActivation

from .utils import make_caption, Logger

from ..dep.torchprofile import profile_macs


__all__ = [
    "LAYER_SETTINGS", "FIRST_CONV_CHANNELS", "LAST_CONV_CHANNELS", "DROPOUT",
    "build_first", "build_last", "build_pool", "build_classififer",
    "MobileNetV2Skeleton", "skeletonnet_train", "skeletonnet_valid"
]


# MobileNetV2 settings for bottleneck layers. Copied from
# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
LAYER_SETTINGS: Final[list[list[int]]] = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]

# The out channels of the first convolution layer of MobileNetV2.
FIRST_CONV_CHANNELS: Final[int] = 32

# The out channels of the last convolution layer of MobileNetV2.
LAST_CONV_CHANNELS: Final[int] = 1280

# The default dropout rate of MobileNetV2.
DROPOUT: Final[float] = 0.2


def build_first(
    out_channels: int = FIRST_CONV_CHANNELS,
    norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
    activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
) -> Conv2dNormActivation:
    """
    A convenience method to build the first conv2d layer of MobileNetV2.

    Args:
        out_channels (int):
            The amount of output channels.
        norm_layer (Callable[..., nn.Module], None):
            The constructor for the norm layer.
        activation_layer (Callable[..., nn.Module], None):
            The constructor for the activation layer.
    Returns:
        Conv2dNormActivation:
            The first conv2d layer.
    """
    return Conv2dNormActivation(
        3,
        out_channels,
        kernel_size=3,
        stride=2,
        norm_layer=norm_layer,
        activation_layer=activation_layer
    )


def build_last(
    in_channels: int = LAYER_SETTINGS[-1][1],
    out_channels: int = LAST_CONV_CHANNELS,
    norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
    activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
) -> Conv2dNormActivation:
    """
    A convenience method to build the last conv2d layer of MobileNetV2.

    Args:
        int_channels (int):
            The output channels of the last block operation.
        out_channels (int):
            The amount of output channels.
        norm_layer (Callable[..., nn.Module], None):
            The constructor for the norm layer.
        activation_layer (Callable[..., nn.Module], None):
            The constructor for the activation layer.
    Returns:
        Conv2dNormActivation:
            The last conv2d layer.
    """
    return Conv2dNormActivation(
        in_channels,
        out_channels,
        1,
        norm_layer=norm_layer,
        activation_layer=activation_layer
    )


def build_pool() -> nn.Sequential:
    """
    A convenience method to build the pooling layer of MobileNetV2.

    Returns:
        nn.Sequential:
            The combination of avg pooling and flatten operation.
    """
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1)
    )


def build_classifier(
    classes: int, out_channels: int = LAST_CONV_CHANNELS, dropout: float = DROPOUT
) -> nn.Sequential:
    """
    A convenience method to build the classifier layer of MobileNetV2.

    Args:
        classes (int):
            The amount of classes to recognize.
        out_channels (int):
            The amount of aoutput channels of the last conv2d operation.
        dropout (float):
            The percentage of dropout.

    Returns:
        nn.Sequential:
            The combination of dropout and linear operation.
    """
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(out_channels, classes)
    )


class MobileSkeletonNet(nn.Module):
    """
    A skeleton of the MobileNetV2 architecture. Expects all layers to be passed
    to the constructor. Implements common methods for weight initialization,
    forward pass and computing FLOPs.

    The weight initialization is copied from
    https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
    """

    # Layers of the NN.
    first: nn.Module
    blocks: nn.ModuleList
    last: nn.Module
    pool: nn.Sequential  # avg pooling + flatten
    classifier: nn.Sequential  # dropout + linear

    def __init__(
        self,
        first: nn.Module,
        blocks: Iterable[nn.Module],
        last: nn.Module,
        pool: nn.Sequential,
        classifier: nn.Sequential,
        initialize_weights: bool = True
    ) -> None:
        super().__init__()

        self.first = first
        self.blocks = nn.ModuleList(blocks)
        self.last = last
        self.pool = pool
        self.classifier = classifier

        if initialize_weights:
            self._weight_initialization()

    def _weight_initialization(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.first(inputs)

        for block in self.blocks:
            inputs = block(inputs)

        inputs = self.last(inputs)
        inputs = self.pool(inputs)

        return self.classifier(inputs)

    @cache
    def flops(self, resolution: int, *, device=None) -> int:
        """
        The FLOPs/MACs of this model. They are calculated once and the result is
        cached for future uses.

        Args:
            resolution (int):
                The resolution of the input image.

        Returns:
            int:
                The computed FLOPs/MACs.
        """
        self.to(device)
        inputs = torch.randn(1, 3, resolution, resolution, device=device)

        return profile_macs(self, inputs)


def skeletonnet_train(
    skeletonnet: MobileSkeletonNet,
    dl: DataLoader,
    epochs: int,
    momentum: float = 0.9,
    weight_decay: float = 5e-5,
    *,
    logger: Logger,
    batches: int | None = None,
    device=None
) -> None:
    """
    Train a `MobileSkeletonNet` on the data set `ds` using the `SGD` optimizer.

    Args:
        skeletonnet (MobileSkeletonNet):
            The network to train.
        dl (torch.utils.data.DataLoader):
            The data loader of the data set to train on.
        epochs (int):
            The number of training epochs.
        momentum (float):
            The momenum for `SGD` to use.
        weight_decay (float):
            The weight decay for `SGD` to use.
        logger (Logger):
            The interface to pass logging information to.
        batches (int, None):
            The number of batches per epoch. If set to `None` use the whole data
            set.
        device:
            The Pytorch device the network is moved to.
    """
    train_start = time.time()

    batches = batches if batches is not None else len(dl)

    skeletonnet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        skeletonnet.parameters(), momentum=momentum, weight_decay=weight_decay
    )

    skeletonnet.train()
    for i in range(epochs):
        logger.log(make_caption(f"Epoch {i + 1}/{epochs}", 70, "-"))
        epoch_start = time.time()

        batch_start = time.time()  # to capture data load time
        for j, (images, labels) in enumerate(dl):
            logger.log(
                f"epoch={i + 1}, batch={j + 1:0{len(str(batches))}}/{batches}",
                end=""
            )

            images = images.to(device)
            labels = labels.to(device)

            outputs = skeletonnet(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_start

            logger.log(f", time={batch_time:.2f}s")
            batch_start = time.time()

            if batches == j + 1:
                break

        epoch_time = time.time() - epoch_start
        logger.log(f"\ntime={epoch_time:.2f}s\n")

    train_time = time.time() - train_start
    logger.log(f"\ntotal={train_time:.2f}s\n")


def skeletonnet_valid(
    skeletonnet: MobileSkeletonNet,
    dl: DataLoader,
    *,
    batches: int | None = None,
    device=None
) -> float:
    """
    Validate a `MobileSkeletonNet` on the data set `ds`.

    Args:
        skeletonnet (MobileSkeletonNet):
            The network to validate.
        dl (torch.utils.data.DataLoader):
            The data loader of the data set used to validate.
        batches (int, None):
            The number of batches per epoch. If set to `None` use the whole data
            set.
        device:
            The Pytorch device the network is moved to.

    Returns:
        float:
            The accuracy in [0, 1].
    """
    batches = batches if batches is not None else len(dl)

    skeletonnet.to(device)
    skeletonnet.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for i, (features, labels) in enumerate(dl):
            features = features.to(device)
            labels = labels.to(device)

            outputs = skeletonnet(features)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if batches == i + 1:
                break

    return float(correct) / float(total)
