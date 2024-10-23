import math
import tempfile
import itertools
from PIL import Image
from pathlib import Path
from typing import Any, cast, Final
from urllib.request import urlretrieve
from collections.abc import Callable, Sequence
from shutil import copyfile, rmtree, unpack_archive

import torch

from torchvision.io import decode_image, ImageReadMode

import pandas as pd

from . import CustomDataset


__all__ = ["strict_split", "GTSRBDataset"]


_training_data: Final[str] = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
_test_data: Final[str] = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
_test_annotations: Final[str] = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"


def _download(url: str, zipped: Path) -> None:
    _, headers = urlretrieve(url, zipped)

    content = headers.get_content_type()
    if content != "application/zip":
        raise Exception(f"invalid file type for GTSRB data: {content}")


def _merge_csvs(unzipped: Path, path: Path) -> None:
    csvs = unzipped.glob("**/*.csv")

    first = next(csvs)
    csv = [first.read_text()]

    for name in csvs:
        with open(name, "r") as f:
            f.readline()
            csv.append(f.read())

    final = "".join(csv)
    with open(path, "w") as f:
        f.write(final)



def _convert_images(
    unzipped: Path, path: Path, to: Callable[[Path, Path], Path]
) -> None:
    for src in unzipped.glob("**/*.ppm"):
        dst = to(path, src)

        im = Image.open(src)
        im.save(dst / f"{src.stem}.png")


def _edit_csv(path: Path, edit: Callable[[str], str]) -> None:
    with open(path, "r") as f:
        content: str = f.read()

    content = edit(content)

    with open(path, "w") as f:
        f.write(content)


def _get_train(path: Path, download: Path, verbose: bool) -> None:
    training = path / "train"
    training.mkdir()

    if verbose:
        print("downloading train data...")
    zipped = download / "train.zip"
    _download(_training_data, zipped)

    if verbose:
        print("unpacking train data...")
    unzipped = download / "train"
    unpack_archive(zipped, unzipped)

    if verbose:
        print("merging csvs...")
    labels = training / "labels.csv"
    _merge_csvs(unzipped, labels)
    _edit_csv(labels, lambda x: x.replace(".ppm", ".png"))

    if verbose:
        print("converting images from 'ppm' to 'png'...")

    def to(path: Path, src: Path) -> Path:
        parts = src.parts
        dst = path / parts[-2]

        dst.mkdir(parents=True, exist_ok=True)
        return dst
    _convert_images(unzipped, training / "images", to)


def _get_test(path: Path, download: Path, verbose: bool) -> None:
    test = path / "test"
    test.mkdir()

    if verbose:
        print("downloading test data...")
    zipped = download / "test.zip"
    _download(_test_data, zipped)

    if verbose:
        print("unpacking test data...")
    unzipped = download / "test"
    unpack_archive(zipped, unzipped)

    if verbose:
        print("converting images from 'ppm' to 'png'...")

    def to(path: Path, _: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path
    _convert_images(unzipped, test / "images", to)

    if verbose:
        print("downloading test annotations...")
    zipped = download / "test-annotations.zip"
    _download(_test_annotations, zipped)

    if verbose:
        print("unpacking test annotations...")
    unzipped = download / "test-annotations"
    unpack_archive(zipped, unzipped)

    src = unzipped / "GT-final_test.csv"
    dst = test / "labels.csv"
    copyfile(src, dst)
    _edit_csv(dst, lambda x: x.replace(".ppm", ".png"))


def get(path: str | Path, verbose: bool = False) -> None:
    """
    Downloads and extracts the German Traffic Sign Recognition Benchmark data
    set. The resulting directory structure is
        path/
         + - train/
              + - images/
                   + - 00000/
                        + - *.png
                   + - ...
              + - labels.csv
         + - test/
              + - images/
                   + - *.png
              + - labels.csv

    Args:
        path (str, Path):
            The path to where to store the data set.
        verbose (bool):
            If ``True`` print progress infromation to stdout.
    """
    if isinstance(path, str):
        path = Path(path)
    path.mkdir()

    tmp = tempfile.TemporaryDirectory()
    download = Path(tmp.name)
    try:
        _get_train(path, download, verbose)
        _get_test(path, download, verbose)
    except Exception as e:
        rmtree(path)
        raise e
    finally:
        tmp.cleanup()


class GTSRBDataset(CustomDataset):
    """
    A custom ``Dataset`` to read in the German Traffic Sign Recognition Benchmark
    data set. Expects the same directory structure as produced in ``get``.

    The data can be found here: https://benchmark.ini.rub.de/gtsrb_dataset.html
    """
    _annotations: pd.DataFrame
    _classes: int
    _images: Path
    _transform: Callable[[torch.Tensor], torch.Tensor] | None
    _target_transform: Callable[[Any], Any] | None
    _train: bool

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[Any], Any] | None = None
    ) -> None:
        if isinstance(root, str):
            root = Path(root)

        if split not in ["train", "test"]:
            raise ValueError(f"unknown split '{split}'")

        path = root / split
        self._annotations = pd.read_csv(
            path / "labels.csv", sep=",|;", engine="python"
        )

        self._images = path / "images"

        self._train = split == "train"

        self._transform = transform
        self._target_transform = target_transform

        self._classes = max(self._annotations.iloc[:]["ClassId"]) + 1

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Any]:
        label = self._annotations.iloc[idx]["ClassId"]
        name = self._annotations.iloc[idx]["Filename"]

        label = int(label)

        if self._train:
            path = self._images / f"{label:05}" / name
        else:
            path = self._images / name
        image = decode_image(str(path), mode=ImageReadMode.RGB)

        if self._transform:
            image = self._transform(image)

        if self._target_transform:
            label = self._target_transform(label)

        return image, label

    @property
    def classes(self) -> int:
        return self._classes

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": self._classes,
            "annotations": self._annotations.to_dict(),
            "split": "train" if self._train else "test",
            "images": str(self._images)
        }

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[Any], Any] | None = None
    ) -> 'GTSRBDataset':
        return cast(
            CustomDataset,
            _Subset(
                config["classes"],
                config["split"],
                pd.DataFrame.from_dict(config["annotations"]),
                Path(config["images"]),
                transform,
                target_transform
            )
        )


class _Subset(GTSRBDataset):
    def __init__(
        self,
        classes: int,
        split: str,
        annotations: pd.DataFrame,
        images: Path,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[Any], Any] | None = None
    ) -> None:
        self._classes = classes
        self._annotations = annotations
        self._images = images

        if split not in ["train", "test"]:
            raise ValueError(f"unknown split '{split}'")
        self._train = split == "train"

        self._transform = transform
        self._target_transform = target_transform


def strict_split(
    ds: GTSRBDataset, lengths: Sequence[int | float]
) -> list[GTSRBDataset]:
    """
    Splits the images of `ds` per class into `lengths` independent subsets.

    Args:
        ds (GTSRBDataset):
            The starting point data set.
        lengths (Sequence[int | float]):
            Either relative values to split or absolute. If the combined values
            of `lengths` don't make up for the whole images of a class in `ds`,
            an additional subset is generated, containing the remaining images.

    Returns:
        list[GTSRBDataset]:
            A list containing `len(lengths)` many independent data sets.
    """
    class_dfs: list[pd.DataFrame] = [
        ds._annotations[ds._annotations["ClassId"] == c]
        for c in range(ds.classes)
    ]

    total = sum(lengths)
    items_per_class: list[list[int]] = []
    for df in class_dfs:
        subset_lengths: list[int] = []

        if math.isclose(total, 1) and total <= 1:
            for i, frac in enumerate(lengths):
                if frac < 0 or frac > 1:
                    raise ValueError(
                        f"fraction at index {i} is not between 0 and 1"
                    )

                n_items_in_split = int(math.floor(len(df) * frac))
                subset_lengths.append(n_items_in_split)
        else:
            if total > len(df):
                raise ValueError(
                    f"sum of lengths {total} is greater than available images {len(df)}"
                )

            subset_lengths.extend(lengths)

        combined = sum(subset_lengths)
        if combined < len(df):
            subset_lengths.append(len(df) - combined)

        items_per_class.append(subset_lengths)

    split: list[list[pd.DataFrame]] = [
        [
            df[offset - length : offset]
            for offset, length in zip(itertools.accumulate(n), n)
        ] for df, n in zip(class_dfs, items_per_class)
    ]

    transposed: list[pd.DataFrame] = [
        pd.concat([item[i] for item in split], ignore_index=True)
        for i in range(len(split[0]))
    ]

    return [
        cast(
            GTSRBDataset,
            _Subset(
                ds.classes,
                "train" if ds._train else "test",
                annotations,
                ds._images,
                ds._transform,
                ds._target_transform
            )
        ) for annotations in transposed
    ]