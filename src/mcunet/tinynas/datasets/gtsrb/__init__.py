import tempfile
from PIL import Image
from pathlib import Path
from typing import Any, Final
from collections.abc import Callable
from urllib.request import urlretrieve
from shutil import copyfile, rmtree, unpack_archive

import torch

from torchvision.io import read_image

import pandas as pd

from .. import CustomDataset


__all__ = ["GTSRBDataset"]


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


def _get_training(path: Path, download: Path, verbose: bool) -> None:
    training = path / "training"
    training.mkdir()

    if verbose:
        print("downloading training data...")
    zipped = download / "training.zip"
    _download(_training_data, zipped)

    if verbose:
        print("unpacking training data...")
    unzipped = download / "training"
    unpack_archive(zipped, unzipped)

    if verbose:
        print("merging csvs...")
    labels = training / "labels.csv"
    _merge_csvs(unzipped, labels)
    _edit_csv(labels, lambda x: x.replace(".ppm", ".png"))

    if verbose:
        print("converting images 'ppm' to 'png'...")

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
        print("converting images 'ppm' to 'png'...")

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
         + - training/
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
        _get_training(path, download, verbose)
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
        image = read_image(path)

        if self._transform:
            image = self._transform(image)

        if self._target_transform:
            label = self._target_transform(label)

        return image, label

    @property
    def classes(self) -> int:
        return self._classes
