from typing import Any
from pathlib import Path
from collections.abc import Callable

import torch
from torch.utils.data import Dataset

from torchvision.io import read_image

import pandas as pd

import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    """
    A custom ``Dataset```to read in the German Traffic Sign Recognition Benchmark
    (GTSRB). The data can be found here: https://benchmark.ini.rub.de/gtsrb_dataset.html
    """
    _annotations: pd.DataFrame
    _images: Path
    _transform: Callable[[torch.Tensor], torch.Tensor] | None
    _target_transform: Callable[[Any], Any] | None
    _train: bool

    def __init__(
        self,
        annotations: str | Path,
        images: str | Path,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[Any], Any] | None = None
    ) -> None:
        if isinstance(annotations, str):
            annotations = Path(annotations)
        self._annotations = pd.read_csv(annotations, sep=",|;", engine="python")

        if isinstance(images, str):
            images = Path(images)
        self._images = images

        # The directory containing the test images is flat, while the directory
        # containing the train images has subdirectories.
        try:
            next(self._images.glob("*.png"))
        except StopIteration:
            self._train = True
        else:
            self._train = False

        self._transform = transform
        self._target_transform = target_transform

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Any]:
        label = self._annotations.iloc[idx]["ClassId"]
        name = self._annotations.iloc[idx]["Filename"]

        if self._train:
            path = self._images / f"{label:05}" / name
        else:
            path = self._images / name
        image = read_image(path)

        if self._transform:
            image = self._transform(image)

        if self._target_transform:
            label = self._target_transform(label)

        return image, label
