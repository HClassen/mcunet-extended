from abc import abstractmethod, ABC
from collections.abc import Callable

import torch
from torch.utils.data import Dataset

from torchvision.transforms import Compose, Resize, ConvertImageDtype

import numpy as np

import matplotlib.pyplot as plt


__all__ = ["CustomDataset", "show", "transform_resize"]


class CustomDataset(ABC, Dataset):
    """
    Base class for all data sets to use. Creates a uniform interface.
    """
    @property
    @abstractmethod
    def classes(self) -> int:
        pass


def _find_div(amount: int, sqrt: int) -> int | None:
    for i in range(sqrt, 1, -1):
        if amount % i == 0:
            return i

    return None


def show(ds: CustomDataset, amount: int) -> None:
    """
    A convenience function to show `amount` many images of a `CustomDataset`.
    The shown images are randomly sampled. Tries to arange them in a square or
    nearly square rectangle.

    Args:
        ds (CustomDataset):
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


def transform_resize(size: int | tuple[int, int]) -> Compose:
    """
    This creates a transform function to resize images to `size` casts
    the tensors to  type `torch.float32`. The returned transform is meant to
    be passed to the `torch.utils.data.Dataset` constructor.

    Args:
        size (int, tuple[int, int]):
            The size of the transformed image.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]:
            A transform function to be passed to `torch.utils.data.Dataset`.
    """
    if isinstance(size, int):
        size = (size, size)

    return Compose([Resize(size), ConvertImageDtype(torch.float32)])
