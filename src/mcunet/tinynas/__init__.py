import time
from itertools import islice
from typing import Any, Final
from bisect import insort_left
from operator import itemgetter
from collections.abc import Callable, Iterable

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

from .searchspace import configurations, Model, SearchSpace
from .searchspace.model import build_model
from .oneshot import (
    initial_population,
    crossover,
    mutate,
    _make_caption,
    OneShotNet,
    SuperNet
)
from .datasets import CustomDataset
from .utils import EDF
from .utils.torchhelper import get_device, test


__all__ = ["SearchSpace", "SampleManager"]


class SampleManager():
    """
    Handle sampling models from different search spaces and extracting different
    statistics from them. Not all search spaces and/or models must be iterated
    over at once. An object from this class can be serialized/deserialized and
    then continued.
    """
    _spaces: list[SearchSpace]
    _classes: int
    _models: list[list[searchspace.Model]] | None
    _flops: list[list[int]] | None
    _edfs: list[EDF] | None
    _index: tuple[int, int] | None

    def __init__(
        self,
        classes: int,
        spaces: Iterable[SearchSpace] | Iterable[tuple[float, int]] | None = None
    ) -> None:
        if spaces is None:
            spaces = configurations()

        spaces = list(spaces)
        if isinstance(spaces[0], tuple):
            spaces = [SearchSpace(x, y) for x, y in spaces]

        self._spaces = spaces
        self._classes = classes

        self._models = None
        self._flops = None
        self._edfs = None
        self._index = None

    @property
    def edfs(self) -> list[EDF]:
        """
        Get the list of all EDFs for all FLOPs of sampled models per search
        space.

        Returns:
            list[EDF]:
                A list of ``EDF`` objects.
        """
        self.update_edfs()

        return self._edfs

    def sample(self, m: int = 1000) -> None:
        """
        Sample ``m`` models from the MnasNet search space per configuration. It
        only generate architecture descriptions and not any Pytorch models.

        Args:
            m (int):
                How many samples should be generated.
        """
        self._models = [list(islice(space, m)) for space in self._spaces]

    def _append_flops(self, i: int, j: int) -> None:
        resolution = self._spaces[i].resolution
        net = build_model(self._models[i][j], self._classes)

        self._flops[i].append(net.flops(resolution, get_device()))

    def flops(self, batch: int | None = None) -> None:
        """
        Computes the FLOPs for the models generated via ``sample()``. This
        method can be called several times until all models have their FLOPs
        computed.

        Args:
            batch (int, None):
                The amount of models for which the FLOPs should be computed. If
                ``None`` compute them for all models.
        """
        if self._flops is None:
            self._flops = []

        if self._index is None:
            self._index = (0, 0)

        idx0, idx1 = self._index
        n_samples = len(self._models[0])
        if idx0 >= len(self._models) or idx1 >= n_samples:
            raise StopIteration()

        if batch is None:
            batch = (len(self._models) - self._index[0]) * \
                    (len(self._models[0]) - self._index[1])

        move = idx1 + batch

        if idx0 == len(self._flops):
            self._flops.append([])

        for j in range(idx1, min(move, n_samples)):
            self._append_flops(idx0, j)

        if move < n_samples:
            self._index = (idx0, move)
            return

        batch -= n_samples - idx1

        idx0 += 1
        n_full = idx0 + batch // n_samples
        for i in range(idx0, n_full):
            if i == len(self._flops):
                self._flops.append([])

            for j in range(n_samples):
                self._append_flops(i, j)

        idx0 = n_full
        n_remain = batch % n_samples
        if n_remain > 0:
            self._flops.append([])
        for j in range(n_remain):
            self._append_flops(idx0, j)

        self._index = (idx0, n_remain)

    def update_edfs(self) -> None:
        """
        Create or update the EDFs for all sampled FLOPs.
        """
        if self._flops is None:
            return

        if self._edfs is None:
            self._edfs = []

        for flops in self._flops[len(self._edfs):]:
            self._edfs.append(EDF(flops))

        return self._edfs

    def to_dict(self) -> dict[str, Any]:
        """
        Converts this instance of ``SampleManager`` to a ``dict``. This can
        then be used to save and reload this instance for later use. This way
        the operations on the sampled models can be paused and later resumed.

        Returns:
            dict[str, Any]:
                A ``dict`` containing the content of this manager.
        """
        return {
            "spaces": [
                {
                    "width_mult": space.width_mult,
                    "resolution": space.resolution,
                    "samples": [m.to_dict() for m in lm]
                }
                for space, lm in zip(self._spaces, self._models)
            ],
            "flops": self._flops,
            "index": self._index
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'SampleManager':
        """
        Converts a ``dict`` to a ``SampleManager``.

        Args:
            config (dict[str, Any]):
                The ``dict`` containing the content of a manager to be loaded.

        Returns:
            SampleManager:
                The manager constructed from the ``dict``.
        """
        spaces, flops, index = itemgetter("spaces", "flops", "index")(config)

        models = [
            [Model.from_dict(m) for m in s.get("samples", [])] for s in spaces
        ]

        spaces = [(s["width_mult"], s["resolution"]) for s in spaces]

        manager = cls(spaces)
        manager._models = models
        manager._flops = flops
        manager._index = index

        return manager


def _test_and_add_to_top(
    top: list[tuple[OneShotNet, float]],
    population: Iterable[OneShotNet],
    dl: DataLoader,
    calib_dl: DataLoader,
    device=None
) -> None:
    for candidate in population:
        candidate.to(device)

        # Recalibrate batch norm statistics (running mean/var).
        candidate.train()
        for m in candidate.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

        for (images, _) in islice(calib_dl, 20):
            images = images.to(device)

            candidate(images)
        candidate.eval()

        accuracy = test(candidate, dl, device)

        # Insert based on descending accuracy.
        insort_left(top, (candidate, accuracy), key=lambda entry: -entry[1])


class SearchManager():
    _space: SearchSpace
    _classes: int
    _supernet: SuperNet

    _initial_lr: float

    _ds: CustomDataset
    _train_ds: Dataset
    _valid_ds: Dataset

    def __init__(
        self, space: SearchSpace, ds: CustomDataset, supernet: SuperNet
    ) -> None:
        self._space = space
        self._supernet = supernet
        self._initial_lr = 0.05

        self._ds = ds
        self._train_ds, self._valid_ds = random_split(ds, [0.8, 0.2])

    def train(
        self,
        epochs: int = 300,  # Twice ``warm_up_epochs``
        batches: int | None = None,
        batch_size: int = 256,
        models_per_batch: int = 4,
        warm_up: bool = True,
        warm_up_epochs: int = 150,
        warm_up_batches: int | None = None,
        warm_up_batch_size: int = 256,
        device=None
    ) -> None:
        """
        Train the `SuperNet` by sampling random models per batch of training
        data, setting the `SuperNet` to the sampled model and performing the
        backward pass.

        Args:
            epochs (int):
                The number of training epochs.
            batches (int, None):
                The number of batches per epoch. If set to `None` use the whole
                training data set.
            batch_size (int):
                The number of samples per batch for training.
            models_per_batch (int):
                The number of models sampled per batch of training data.
            warm_up (bool):
                If set to `True` train the largest sub-networks of the
                `SuperNet` first to start the training with bettwer weights.
            warm_up_epochs (int):
                The number of warm up epochs. Only relevant if `warm_up` is
                `True`.
            warm_up_batches (int, None):
                The number of batches per warm up epoch. If set to `None` use
                the whole training data set. Only relevant if `warm_up` is
                `True`.
            warm_up_batch_size (int):
                The number of samples per batch for warm up. Only relevant if
                `warm_up` is `True`.
        """
        if warm_up:
            self._supernet.run_warm_up(
                self._space,
                self._train_ds,
                warm_up_epochs,
                warm_up_batches,
                warm_up_batch_size,
                device
            )

        self._supernet.run_train(
            self._space,
            self._train_ds,
            self._initial_lr,
            epochs,
            batches,
            batch_size,
            models_per_batch,
            device
        )

    def evolution(
        self,
        topk: int = 20,
        batch_size: int = 256,
        calib_batch_size: int = 64,
        iterations: int = 30,
        population_size: int = 100,
        next_gen_split: tuple[float, float] = (0.5, 0.5),
        mutation_rate: float = 0.1,
        fitness: Callable[[Model], bool] = lambda _: True,
        device=None
    ) -> tuple[tuple[OneShotNet, float]]:
        """
        Perform evoltion search on the `SuperNet` to find the model with the
        highest accuracy. Per iteration, single-point crossover and mutation is
        performed on the current population. Before the accuracy of a model is
        tested, calibrate the batch-normalisation layers on the validation data.

        Args:
            topk (int):
                The number of models with the highest accuracy to keep track off.
            batch_size (int):
                The batch size to use during validation.
            calib_batch_size (int):
                The batch size to use during the calibration of the
                batch-normalisation layers.
            iterations (int):
                For how many generations the evaluation search should run.
            population_size (int):
                The size of the population per generation.
            next_gen_split (tuple[float, float]):
                The percentages the next population is made up for from crossover
                and mutation.
            mutation_rate (float):
                Threashold for mutation to take effect.
            fitness (Callable[[Model], bool]):
                Evaluate the fitness of a sampled model on some criteria. Only
                models where `fitness` returns `True` are selected for the
                population of a generation.

        Returns:
            tuple[tuple[OneShotNet, float]]:
                The final `topk` many models and their accuracy.
        """
        dl = DataLoader(self._valid_ds, batch_size, shuffle=False)
        calib_dl = DataLoader(self._valid_ds, calib_batch_size, shuffle=True)

        population = initial_population(
            iter(self._space), population_size, self._supernet, fitness
        )

        rng = np.random.default_rng()
        top: list[tuple[OneShotNet, float]] = []

        n_cross: Final[int] = int(population_size * next_gen_split[0])
        n_mut: Final[int] = int(population_size * next_gen_split[1])

        print(_make_caption("Evolution", 70, " "))
        for i in range(iterations):
            print(_make_caption(f"Iteration {i + 1}/{iterations}", 70, "-"))

            iteration_start = time.time()

            print(f"iteration={i + 1}, method=test-and-add", end="")

            test_start = time.time()
            _test_and_add_to_top(top, population, dl, calib_dl, device)
            top = top[:topk]
            test_time = time.time() - test_start

            print(f", time={test_time:.2f}s")

            print(f"iteration={i + 1}, method=crossover", end="")
            crossover_start = time.time()

            cross: list[OneShotNet] = []
            for _ in range(n_cross):
                while True:
                    p1 = top[rng.integers(0, topk)][0]
                    p2 = top[rng.integers(0, topk)][0]

                    child = crossover(p1, p2)
                    if fitness(child.to_model()):
                        break

                cross.append(child)

            crossover_time = time.time() - crossover_start
            print(f", time={crossover_time:.2f}s")

            print(f"iteration={i + 1}, method=mutate", end="")
            mutate_start = time.time()

            mut: list[OneShotNet] = [
                mutate(top[rng.integers(0, topk)][0], mutation_rate)
                for _ in range(n_mut)
            ]

            mutate_time = time.time() - mutate_start
            print(f", time={mutate_time:.2f}s")

            population = cross + mut

            iteration_time = time.time() - iteration_start

            print()
            print(f"time={iteration_time:.2f}s")
            print()

        print(_make_caption("Final Population", 70, "-"))
        print(f"method=test-and-add", end="")

        test_start = time.time()
        _test_and_add_to_top(top, population, dl, calib_dl, device)
        test_time = time.time() - test_start

        print(f", time={test_time:.2f}s")

        return tuple(top[:topk])

    def to_dict(self) -> dict[str, Any]:
        """
        Converts this instance of `SearchManager` to a `dict`. This can then be
        used to save and reload this instance for later use. This way the
        training and evolution search can be paused and later resumed. It
        doesn't save the state of the `Dataset`.

        Returns:
            dict[str, Any]:
                A `dict`` containing the content of this manager.
        """
        return {
            "width_mult": self._space.width_mult,
            "resolution": self._space.resolution,
            "classes": self._ds.classes,
            "weights": self._supernet.state_dict()
        }

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        space: Callable[[float, int], SearchSpace],
        ds: CustomDataset,
        supernet: SuperNet
    ) -> 'SearchManager':
        """
        Converts a `dict` to a `SearchManager`.

        Args:
            config (dict[str, Any]):
                The `dict`` containing the content of a manager to be loaded.
            ds (Dataset):
                The training and validation data.
            batch_size (int):
                The batch size to use during training.

        Returns:
            SearchManager:
                The manager constructed from the `dict`.
        """
        width_mult, resolution, classes, weights = \
            itemgetter("width_mult", "resolution", "classes", "weights")(config)

        if classes != ds.classes:
            raise Exception(
                f"classes mismatch: expected {classes} data set has {ds.classes}"
            )

        manager = cls(space(width_mult, resolution), ds, supernet)
        manager._supernet.load_state_dict(weights)

        return manager
