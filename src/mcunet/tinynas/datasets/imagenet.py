import math
import itertools
from pathlib import Path
from typing import Any, cast
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence

import torch

from torchvision.io import read_image, ImageReadMode

from . import CustomDataset


__all__ = ["subset", "strict_split", "ImageNetDataset"]


class ImageNetDataset(CustomDataset):
    _classes: int
    _images: list[tuple[str | Path, str]]  # (filename, synset)
    _transform: Callable[[torch.Tensor], torch.Tensor] | None
    _target_transform: Callable[[Any], Any] | None

    _synset_mapping: dict[str, tuple[int, str]]  # synset -> (class, description)
    _classes_mapping: dict[int, str]  # class -> synset

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
                raise ValueError(f"unknown split '{split}'")

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
        self._classes_mapping = {}
        for line in lines:
            splitted = line.split(" ", 1)
            name = splitted[0]

            if name not in mapping:
                raise RuntimeError(f"unknown class '{name}'")

            self._synset_mapping[name] = (mapping[name], splitted[1])
            self._classes_mapping[mapping[name]] = name

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
        label = self._synset_mapping[mapping[1]][0]

        if self._transform:
            image = self._transform(image)

        if self._target_transform:
            label = self._target_transform(label)

        return image, label

    @property
    def classes(self) -> int:
        return self._classes


class _Subset(ImageNetDataset):
    def __init__(
        self,
        classes: int,
        images: list[tuple[str | Path, str]],
        synset_mapping: dict[str, tuple[int, str]],
        classes_mapping: dict[int, str],
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[Any], Any] | None = None
    ) -> None:
        self._classes = classes
        self._images = images
        self._synset_mapping = synset_mapping
        self._classes_mapping = classes_mapping

        self._transform = transform
        self._target_transform = target_transform


def subset(
    ds: ImageNetDataset,
    classes: int | None,
    images: int | None
) -> ImageNetDataset:
    """
    Returns a subset of the `ImageNetDataset` `ds`. Can be applied to the number
    of classes, images per class or both. Randomly selects the subset of classes
    and images.

    Args:
        ds (ImageNetDataset):
            The starting point data set.
        classes (int, None):
            A subset of classes. If `None` or `classes > ds.classes` it reverts
            to `ds.classes`.
        images (int, None):
            A subset of images per class. If `None` or `images` is grater than
            the number of images of a class, it reverts to the number of images
            of a class in `ds`.

    Returns:
        ImageNetDataset:
            A new data set with a subset of images of ds.
    """
    data: list[tuple[str | Path, str]] = []
    synset_mapping: dict[str, tuple[int, str]] = {}
    classes_mapping: dict[int, str] = {}

    classes = ds.classes if classes is None else min(classes, ds.classes)
    subclasses, _ = torch.sort(
        torch.randperm(ds.classes)[:classes], dim=0, descending=False
    )

    for i, subclass in enumerate(subclasses):
        subclass = subclass.item()
        synset = ds._classes_mapping[subclass]
        classes_mapping[i] = synset
        synset_mapping[synset] = (i, ds._synset_mapping[synset][1])

        class_images: list[str | Path] = [
            entry[0] for entry in ds._images if entry[1] == synset
        ]
        length = len(class_images)

        images = length if images is None else min(images, length)
        subimages = torch.randperm(length)[:images]
        data.extend(
            [(class_images[j.item()], synset) for j in subimages]
        )

    return cast(
        ImageNetDataset,
        _Subset(
            classes,
            data,
            synset_mapping,
            classes_mapping,
            ds._transform,
            ds._target_transform
        )
    )


def strict_split(
    ds: ImageNetDataset,
    lengths: Sequence[int | float]
) -> list[ImageNetDataset]:
    """
    Splits the images of `ds` per class into `lengths` independent subsets.

    Args:
        ds (ImageNetDataset):
            The starting point data set.
        lengths (Sequence[int | float]):
            Either relative values to split or absolute. If the combined values
            of `lengths` don't make up for the whole images of a class in `ds`,
            an additional subset is generated, containing the remaining images.

    Returns:
        list[ImageNetDataset]:
            A list containing `len(lengths)` many independent data sets.
    """
    images: list[list[tuple[str | Path, str]]] = []

    for c in range(ds.classes):
        synset = ds._classes_mapping[c]

        class_images: list[tuple[str | Path, str]] = [
            entry for entry in ds._images if entry[1] == synset
        ]

        images.append(class_images)

    total = sum(lengths)
    items_per_class: list[list[int]] = []
    for class_images in images:
        subset_lengths: list[int] = []

        if math.isclose(total, 1) and total <= 1:
            for i, frac in enumerate(lengths):
                if frac < 0 or frac > 1:
                    raise ValueError(
                        f"fraction at index {i} is not between 0 and 1"
                    )

                n_items_in_split = int(math.floor(len(class_images) * frac))
                subset_lengths.append(n_items_in_split)
        else:
            if total > len(class_images):
                raise ValueError(
                    f"sum of lengths {total} is greater than avaialble images {sum(class_images)}"
                )

            subset_lengths.extend(lengths)

        combined = sum(subset_lengths)
        if combined < len(class_images):
            subset_lengths.append(len(class_images) - combined)

        items_per_class.append(subset_lengths)

    split: list[list[tuple[str, str]]] = [
        [
            items[offset - length : offset]
            for offset, length in zip(itertools.accumulate(n), n)
        ] for items, n in zip(images, items_per_class)
    ]

    transposed: list[list[tuple[str, str]]] = [
        [x for item in split for x in item[i]] for i in range(len(split[0]))
    ]

    return [
        cast(
            ImageNetDataset,
            _Subset(
                ds.classes,
                items,
                ds._synset_mapping,
                ds._classes_mapping,
                ds._transform,
                ds._target_transform
            )
        ) for items in transposed
    ]
