import math
import random
from pathlib import Path
from typing import Callable
from functools import cache

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.common_types import _size_2_t
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision.io import read_image
from torchvision.transforms.functional import resize
from torchvision.ops import Conv2dNormActivation, SqueezeExcitation

from torchprofile import profile_macs

import pandas as pd

import matplotlib.pyplot as plt

from .searchspace import Model, ConvOp, SkipOp


def _add_se(
    layers: list[nn.Module],
    in_channels: int,
    se_ratio: float,
    activation_layer: Callable[..., torch.nn.Module] | None = torch.nn.ReLU6
) -> None:
    if se_ratio == 0:
        return

    # Copied from
    # https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
    squeeze_channels = max(1, int(in_channels * se_ratio))

    layers.append(
        SqueezeExcitation(
            in_channels,
            squeeze_channels,
            activation=activation_layer
        )
    )


class BaseOp(nn.Module):
    conv: nn.Sequential
    use_residual: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t,
        skip_op: SkipOp
    ) -> None:
        super().__init__()

        self.use_residual =  stride == 1 and \
                             in_channels == out_channels and \
                             skip_op == SkipOp.IDENTITY

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return input + self.conv(input)
        else:
            return self.conv(input)


class Conv2dOp(BaseOp):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip_op: SkipOp,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: _size_2_t | str | None = None,
        norm_layer: Callable[..., torch.nn.Module] | None = torch.nn.BatchNorm2d,
        activation_layer: Callable[..., torch.nn.Module] | None = torch.nn.ReLU6
    ) -> None:
        super().__init__(in_channels, out_channels, stride, skip_op)

        layers: list[nn.Module] = [
            Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                inplace=True
            )
        ]

        _add_se(layers,out_channels, se_ratio, activation_layer)

        self.conv = nn.Sequential(*layers)


class DWConv2dOp(BaseOp):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float,
        skip_op: SkipOp,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: _size_2_t | str | None = None,
        norm_layer: Callable[..., torch.nn.Module] | None = torch.nn.BatchNorm2d,
        activation_layer: Callable[..., torch.nn.Module] | None = torch.nn.ReLU6
    ) -> None:
        super().__init__(in_channels, out_channels, stride, skip_op)

        layers: list[nn.Module] = [
            Conv2dNormActivation(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                inplace=True
            )
        ]

        # The order of operations is guessed from the MnasNet paper and
        # implementation. Both are not clear how to handle SE for the depthwise
        # convolution op, but for the mobile inverted bottleneck SE is performed
        # before the final pointwise convolution. So do the same here.
        _add_se(layers, in_channels, se_ratio, activation_layer)

        layers.extend([
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            norm_layer(out_channels)
        ])

        self.conv = nn.Sequential(*layers)


class Expand2d(nn.Module):
    expand: Conv2dNormActivation

    def __init__(
        self,
        in_channels: int,
        expand_ratio: float,
        norm_layer: Callable[..., torch.nn.Module] | None = torch.nn.BatchNorm2d,
        activation_layer: Callable[..., torch.nn.Module] | None = torch.nn.ReLU6
    ) -> None:
        super().__init__()

        self.expand = Conv2dNormActivation(
            in_channels,
            int(round(in_channels * expand_ratio)),
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            inplace=True
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.expand(input)


class MBConv2dOp(BaseOp):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float,
        se_ratio: float,
        skip_op: SkipOp,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
        padding: _size_2_t | str | None = None,
        norm_layer: Callable[..., torch.nn.Module] | None = torch.nn.BatchNorm2d,
        activation_layer: Callable[..., torch.nn.Module] | None = torch.nn.ReLU6
    ) -> None:
        super().__init__(in_channels, out_channels, stride, skip_op)

        hidden = int(round(in_channels * expand_ratio))
        layers: list[nn.Module] = []

        if expand_ratio > 1:
            layers.append(
                Expand2d(
                    in_channels,
                    expand_ratio,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer
                )
            )

        layers.append(
            Conv2dNormActivation(
                hidden,
                hidden,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                inplace=True
            )
        )

        _add_se(layers, hidden, se_ratio, activation_layer)

        layers.extend([
            nn.Conv2d(
                hidden,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            norm_layer(out_channels)
        ])

        self.conv = nn.Sequential(*layers)


class Net(nn.Module):
    resolution: int
    n_classes: int
    features: nn.Sequential
    pool: nn.Sequential  # avg pooling + flatten
    classifier: nn.Sequential  # dropout + linear

    def __init__(self, model: Model) -> None:
        super().__init__()

        self.resolution = model.resolution
        self.n_classes = model.n_classes

        activation = nn.ReLU6
        norm = nn.BatchNorm2d

        # Build the first feature layer. Copied from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        layers: list[nn.Module] = [
            Conv2dNormActivation(
                3,
                model.blocks[0].in_channels,
                stride=2,
                norm_layer=norm,
                activation_layer=activation
            )
        ]

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
                    skip_op = block.first_skip
                else:
                    in_channels = block.out_channels
                    stride = 1
                    skip_op = block.rest_skip

                match block.conv_op:
                    case ConvOp.CONV2D:
                        op = Conv2dOp(
                            in_channels,
                            block.out_channels,
                            se_ratio=block.se_ratio,
                            skip_op=skip_op,
                            stride=stride
                        )
                    case ConvOp.DWCONV2D:
                        op = DWConv2dOp(
                            in_channels,
                            block.out_channels,
                            se_ratio=block.se_ratio,
                            skip_op=skip_op,
                            stride=stride
                        )
                    case ConvOp.BOTTLENECK:
                        op = MBConv2dOp(
                            in_channels,
                            block.out_channels,
                            expand_ratio=block.expand_ratio,
                            se_ratio=block.se_ratio,
                            skip_op=skip_op,
                            stride=stride
                        )
                    case _:
                        raise ValueError(f"unknown conv op: {block.conv_op}")

                block_features.append(op)

            layers += block_features

        # Build the last feature layer. Copied from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        layers.append(
            Conv2dNormActivation(
                model.blocks[-1].out_channels,
                model.last_channels,
                kernel_size=1,
                norm_layer=norm,
                activation_layer=activation
            )
        )

        self.features = nn.Sequential(*layers)

        # Copied and modified from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1)
        )

        # Copied from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        self.classifier = nn.Sequential(
            nn.Dropout(p=model.dropout),
            nn.Linear(model.last_channels, self.n_classes)
        )

        # Weight initialization. Copied from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.features(input)
        x = self.pool(x)
        return self.classifier(x)

    @property
    @cache
    def flops(self) -> int:
        """
        The FLOPs/MACs of this model. They are calculated once and the result is
        cached for future uses.

        Returns:
            int: The computed FLOPs/MACs.
        """
        input = torch.randn(1, 3, self.resolution, self.resolution)

        return profile_macs(self, input)


class _CustomDataset(Dataset):
    """
    A custom ``torch.utils.data.Dataset`` which implements the methods shared by
    ``x.pytorch.TrainingDataset`` and ``x.pytorch.TestDataset``.
    """
    def __init__(
        self,
        annotations: str | Path, images: str | Path,
        transform=None, target_transform=None
    ):
        if isinstance(annotations, str):
            annotations = Path(annotations)
        self._annotations = pd.read_csv(annotations, sep=",|;", engine="python")

        if isinstance(images, str):
            images = Path(images)
        self._images = images

        self._transform = transform
        self._target_transform = target_transform

    def __len__(self) -> int:
        return len(self._annotations)

    def show(self, amount: int) -> None:
        """
        Show ``amount`` many images of the data set. The shown images are randomly
        sampled. Tries to arange them in a square or nearly square rectangle.

        Args:
            amount (int): How many images should be shown.
        """
        a = int(math.sqrt(amount))
        b = amount // a
        if a * b < amount:
            a += 1

        _, ax = plt.subplots(a, b)
        for row in range(a):
            for col in range(b):
                idx = random.randint(0, len(self))
                image, _ = self[idx]

                image = image.squeeze().type(torch.uint8)
                ax[row, col].imshow(image.permute(1, 2, 0))

        plt.show()


class TrainingDataset(_CustomDataset):
    """
    A custom ``torch.utils.data.Dataset`` to load the training portion of the
    GTSRB data set. Expects the image to be in the format of JPEG, PNG or GIF.
    """
    def __getitem__(self, idx: int):
        label = self._annotations.iloc[idx]["ClassId"]
        name = self._annotations.iloc[idx]["Filename"]

        path = self._images / f"{label:05}" / name
        image = read_image(path)

        if self._transform:
            image = self._transform(image)

        if self._target_transform:
            label = self._target_transform(label)

        return image, label


class TestDataset(_CustomDataset):
    """
    A custom ``torch.utils.data.Dataset`` to load the separate test portion of
    the GTSRB data set. Expects the image to be in the format of JPEG, PNG or
    GIF.
    """
    def __getitem__(self, idx: int):
        label = self._annotations.iloc[idx]["ClassId"]
        name = self._annotations.iloc[idx]["Filename"]

        path = self._images / name
        image = read_image(path)

        if self._transform:
            image = self._transform(image)

        if self._target_transform:
            label = self._target_transform(label)

        return image, label


def transform(resolution: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    A transformation to be provided to ``x.pytorch.TrainingDataset`` or
    ``x.pytorch.TestDataset``. It resizes the GTSRB images and casts the tensors
    to  type ``torch.float32`` to be used during training and evaluation.

    Args:
        resolution (int): The resolution of the images.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: A transformation function to be used by a ``torch.utils.data.Dataset``.
    """
    def func(input: torch.Tensor) -> torch.Tensor:
        x = resize(input, [resolution, resolution])
        return x.type(torch.float32)

    return func


def get_train_dataloaders(
    csv: str | Path,
    images: str | Path,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    split: list[int | float] = [0.7, 0.3],
    batch_size: int = 64,
    shuffle: bool = True
) -> tuple[DataLoader, DataLoader]:
    """
    Convenience function to create a training and validation
    ``torch.utils.data.DataLoader`` for the GTSRB.

    Args:
        csv (str, Path): The CSV file containing the annotations.
        images (str, Path): The directory containing the image subdirectories.
        transform (Callable[[torch.Tensor], torch.Tensor], None): A transformation to be applied to the images.
        split (list[int], list[float]): Split the dataset either by fraction or by count.
        batch_size (int): The size of the batches.
        shuffle (bool): Shuffle the data at each epoch.

    Return:
        torch.utils.data.DataLoader: The training data loader.
        torch.utils.data.DataLoader: The validation data loader.
    """
    ds = TrainingDataset(csv, images, transform=transform)

    train_ds, valid_ds = random_split(ds, split)

    train_dl = DataLoader(train_ds, batch_size, shuffle=shuffle)
    valid_dl = DataLoader(valid_ds, batch_size, shuffle=shuffle)

    return train_dl, valid_dl


def get_test_dataloaders(
    csv: str | Path,
    images: str | Path,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    batch_size: int = 64
) -> DataLoader:
    """
    Convenience function to create a test ``torch.utils.data.DataLoader`` for
    the GTSRB.

    Args:
        csv (str, Path): The CSV file containing the annotations.
        images (str, Path): The directory containing the images.
        transform (Callable[[torch.Tensor], torch.Tensor], None): A transformation to be applied to the images.
        batch_size (int): The size of the batches.

    Return:
        torch.utils.data.DataLoader: The test data loader.
    """
    ds = TestDataset(csv, images, transform=transform)

    return DataLoader(ds, batch_size, shuffle=False)


def _test(net: Net, test_dl: DataLoader, device) -> float:
    correct = 0
    total = 0
    net.eval()

    with torch.no_grad():
        for features, labels in test_dl:
            features.to(device)
            labels.to(device)

            outputs = net(features)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return float(correct) / float(total) * 100


def train(
    net: Net,
    n_epochs: int,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    optimizer: Callable[..., optim.Optimizer] = optim.Adam,
    device=None
) -> None:
    """
    Implements the training loop for a ``x.pytorch.Net``. Prints out the accuracy
    per epoch.

    Args:
        net (x.pytorch.Net): The neural network to train.
        n_epochs (int): For haw many epochs.
        train_dl (torch.utils.data.DataLoader): The images to train on.
        valid_dl (torch.utils.data.DataLoader): The images to test the accuracy per epoch.
        optimizer (Callable[..., optim.Optimizer]): The optimizer to use. If none is given use ``torch.optim.Adam``.
        device: The device to run torch on.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(net.parameters())

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)

    for epoch in range(n_epochs):
        net.train()
        for features, labels in train_dl:
            features.to(device)
            labels.to(device)

            optimizer.zero_grad()

            outputs = net(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        accuracy = _test(net, valid_dl, device)
        print(f"Epoch {epoch}: accuracy of the model is {accuracy:.2f}%")


def test(net: Net, test_dl: DataLoader, device=None) -> float:
    """
    Test a ``x.pytorch.Net`` after training.

    Args:
        net (x.pytorch.Net): The neural network to train.
        n_epochs (int): For haw many epochs.
        test_dl (torch.utils.data.DataLoader): The images to test on.
        device: The device to run torch on.

    Returns:
        float: The accuracy.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)

    return _test(net, test_dl, device)
