from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
from collections.abc import Callable

import torch

from torchvision.io import read_image, ImageReadMode

from . import CustomDataset


__all__ = ["ImageNetDataset"]


class ImageNetDataset(CustomDataset):
    _classes: int
    _images: list[tuple[str | Path, str]]  # (filename, synset)
    _transform: Callable[[torch.Tensor], torch.Tensor] | None
    _target_transform: Callable[[Any], Any] | None

    _synset_mapping: dict[str, tuple[int, str]]  # synset -> (class, description)

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[Any], Any] | None = None
    ) -> None:
        if isinstance(root, str):
            root = Path(root)

        data = root / "ILSVRC" / "Data" / "CLS-LOC"
        annotations = root / "ILSVRC" / "Annotations" / "CLS-LOC"

        match split:
            case "train":
                self._init_train(data)
            case "test":
                self._init_test(data, annotations)
            case _:
                raise RuntimeError(f"unknown split '{split}'")

        self._init_synset_mapping(root, data)

        self._transform = transform
        self._target_transform = target_transform

    def _init_synset_mapping(self, root: Path, data: Path) -> None:
        train = data / "train"
        walk = train.walk(top_down=True, follow_symlinks=False)
        _, classes, _ = next(walk)

        mapping: dict[str, int] = {}
        for i, name in enumerate(classes):
            if name in mapping:
                raise RuntimeError(f"class '{name}' exists twice")

            mapping[name] = i

        with open(root / "LOC_synset_mapping.txt", "r") as f:
            lines = f.readlines()

        self._synset_mapping = {}
        for line in lines:
            splitted = line.split(" ", 1)
            name = splitted[0]

            if name not in mapping:
                raise RuntimeError(f"unknown class '{name}'")

            self._synset_mapping[name] = (mapping[name], splitted[1])

        self._classes = len(self._synset_mapping.keys())

    def _init_train(self, data: Path) -> None:
        data = data / "train"
        walk = data.walk(top_down=True, follow_symlinks=False)

        self._images = [
            (path / file, path.name)
            for path, _, files in walk for file in files
        ]

    def _init_test(self, data: Path, annotations: Path) -> None:
        data = data / "val"
        annotations = annotations / "val"

        walk = annotations.walk(top_down=True, follow_symlinks=False)
        _, _, files = next(walk)

        images: list[tuple[Path, str]] = []
        for file in files:
            tree = ET.parse(annotations / file)
            root = tree.getroot()

            file_name = root.find("filename").text
            class_name = root.find("object/name").text

            images.append((data / f"{file_name}.JPEG", class_name))

        self._images = images

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Any]:
        mapping = self._images[idx]

        image = read_image(mapping[0], mode=ImageReadMode.RGB)
        # if image.size(dim=0) == 1:
        #     image = image.expand(3, -1, -1)

        label = self._synset_mapping[mapping[1]][0]

        if self._transform:
            image = self._transform(image)

        if self._target_transform:
            label = self._target_transform(label)

        return image, label

    @property
    def classes(self) -> int:
        return self._classes
