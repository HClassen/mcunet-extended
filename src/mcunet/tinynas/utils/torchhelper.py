from typing import Callable

import torch

from torchvision.transforms.functional import resize


def get_device() -> torch.device:
    """
    Returns the available device for torch to run on. Prefers CUDA.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
