import math
import random
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision.io import read_image
from torchvision.transforms.functional import resize

import pandas as pd

import matplotlib.pyplot as plt


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


def _test(net: nn.Module, test_dl: DataLoader, device) -> float:
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
    net: nn.Module,
    n_epochs: int,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    optimizer: Callable[..., optim.Optimizer] = optim.Adam,
    device=None
) -> None:
    """
    Implements the training loop for a ``torch.nn.Module``. Prints out the
    accuracy per epoch.

    Args:
        net (torch.nn.Module): The neural network to train.
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


def test(net: nn.Module, test_dl: DataLoader, device=None) -> float:
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
