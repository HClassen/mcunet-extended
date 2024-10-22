from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision.transforms.functional import resize

import numpy as np

import matplotlib.pyplot as plt


def get_device() -> torch.device:
    """
    Returns the available device for torch to run on. Prefers CUDA.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _find_div(amount: int, sqrt: int) -> int | None:
    for i in range(sqrt, 1, -1):
        if amount % i == 0:
            return i

    return None

def show(ds: Dataset, amount: int) -> None:
    """
    A convenience function to show ``amount`` many images of a dataset. The
    shown images are randomly sampled. Tries to arange them in a square or
    nearly square rectangle.

    Args:
        ds (torch.utils.data.Dataset):
            The dataset containing the images.
        amount (int):
            How many images should be shown.
    """
    sqrt = int(np.ceil(np.sqrt(amount)))
    div = _find_div(amount, sqrt)

    if div:
        rows = div
        columns = int(amount / div)
    else:
        columns = sqrt
        rows = int(np.ceil(amount / columns))

    rng = np.random.default_rng()
    _, ax = plt.subplots(rows, columns)
    for row in range(rows):
        diff = 0
        if (row + 1) * columns > amount:
            diff = (row + 1) * columns - amount
            columns -= diff

        for col in range(columns):
            idx = rng.integers(0, len(ds))
            image, _ = ds[idx]

            image = image.squeeze().type(torch.uint8)
            ax[row, col].imshow(image.permute(1, 2, 0))
            ax[row, col].axis("off")

        for col in range(columns, columns + diff):
            ax[row, col].remove()

    plt.show()


def transform_resize(
    size: int | tuple[int, int]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    This creates a transform function to resize images to ``size`` casts
    the tensors to  type ``torch.float32``. The returned transform is meant to
    be passed to the ``torch.utils.data.Dataset`` constructor.

    Args:
        size (int, tuple[int, int]):
            The size of the transformed image.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]:
            A transform function to be passed to ``torch.utils.data.Dataset``.
    """
    if isinstance(size, int):
        size = (size, size)
    def func(input: torch.Tensor) -> torch.Tensor:
        x = resize(input, list(size))
        return x.type(torch.float32)

    return func


def get_train_dataloaders(
    ds: Dataset,
    split: list[int | float] = [0.7, 0.3],
    batch_size: int = 64,
    shuffle: bool = True
) -> tuple[DataLoader, DataLoader]:
    """
    A convenience function to create a training and validation
    ``torch.utils.data.DataLoader`` from a ``torch.utils.data.Dataset``.

    Args:
        ds (torch.utils.data.Dataset):
            The dataset containing all images to be used during training.
        split (list[int], list[float]):
            Split the dataset either by fraction or by count.
        batch_size (int):
            The size of the batches.
        shuffle (bool):
            Shuffle the data at each epoch.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
            The training and validation data loader.
    """
    train_ds, valid_ds = random_split(ds, split)

    train_dl = DataLoader(train_ds, batch_size, shuffle=shuffle)
    valid_dl = DataLoader(valid_ds, batch_size, shuffle=shuffle)

    return train_dl, valid_dl


def _test(net: nn.Module, dl: DataLoader, device) -> float:
    correct = 0
    total = 0

    net.eval()

    with torch.no_grad():
        for features, labels in dl:
            features = features.to(device)
            labels = labels.to(device)

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
    Implements a simple training loop for a ``torch.nn.Module``. Prints out the
    accuracy per epoch.

    Args:
        net (torch.nn.Module):
            The neural network to train.
        n_epochs (int):
            For how many epochs to train.
        train_dl (torch.utils.data.DataLoader):
            The images to train on.
        valid_dl (torch.utils.data.DataLoader):
            The images to test the accuracy per epoch.
        optimizer (Callable[..., optim.Optimizer]):
            The optimizer to use. Default is ``torch.optim.Adam``.
        device:
            The device to run torch on.
    """
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(net.parameters())

    for epoch in range(n_epochs):
        net.train()
        for features, labels in train_dl:
            features = features.to(device)
            labels = labels.to(device)

            outputs = net(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        accuracy = _test(net, valid_dl, device)
        print(f"Epoch {epoch}: accuracy of the model is {accuracy:.2f}%")


def test(net: nn.Module, dl: DataLoader, device=None) -> float:
    """
    Implements a simple test of accuracy for a ``torch.nn.Module`` after training.

    Args:
        net (torch.nn.Module):
            The neural network to test.
        dl (torch.utils.data.DataLoader):
            The images to test on.
        device:
            The device to run torch on.

    Returns:
        float:
            The accuracy.
    """
    net.to(device)

    return _test(net, dl, device)
