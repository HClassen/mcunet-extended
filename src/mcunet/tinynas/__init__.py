import time
from typing import Any, Final
from bisect import insort_left
from multiprocessing import get_start_method, set_start_method, Pool
from itertools import islice
from collections.abc import Callable, Iterable, Iterator

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

from .searchspace import Model, SearchSpace
from .oneshot import (
    initial_population,
    crossover,
    mutate,
    supernet_warm_up,
    supernet_train,
    OneShotNet,
    SuperNet,
    WarmUpCtx
)
from .datasets import CustomDataset
from .mobilenet import skeletonnet_valid
from .utils import make_caption


__all__ = ["SearchSpace", "SampleManager"]


def _sample(args: tuple[int, SearchSpace]) -> list[list[Model]]:
    m, space = args

    return list(islice(space, m))


def _apply(
    chunk: tuple[Model, SearchSpace],
    fn: Callable[[Model, float, int], tuple[Any, ...]],
    m: int | None
) -> tuple[SearchSpace, tuple[Any, ...]]:
    samples, space = chunk

    sampled = len(samples)
    end = min(m, sampled) if m is not None else sampled

    return (
        space,
        tuple(
            fn(model, space.width_mult, space.resolution)
            for model in samples[:end]
        )
    )


class SampleManager():
    """
    Handle sampling models from different search spaces and extracting different
    statistics from them.
    """
    _spaces: list[SearchSpace]
    _models: list[list[Model]] | None

    def __init__(self, spaces: list[SearchSpace]) -> None:
        self._spaces = spaces
        self._models = None

    def sample(self, m: int = 1000) -> None:
        """
        Sample `m` models from each `SearchSpace` provided in the constructor.

        Args:
            m (int):
                The amount of samples to be generated.
        """
        with Pool() as pool:
            results = pool.map(_sample, [(m, space) for space in self._spaces])

            self._models = [result for result in results]

    def apply(
        self,
        fn: Callable[[Model, float, int], tuple[Any, ...]],
        m: int | None = None
    ) -> Iterator[tuple[SearchSpace, tuple[Any, ...]]]:
        """
        Apply the function `fn` to previously sampled models of the different
        search spaces.

        Args:
            fn (Callable[[Model, float, int], tuple[Any, ...]]):
                This function is applied to all models sampled.
            m (int, None):
                The number of samples per search space to iterate over. If `None`
                iterate over all models.

        Returns:
            Iterator[tuple[SearchSpace, tuple[Any, ...]]]:
                The results for all models of a search space and the search space
                itself.

        Raises:
            RuntimeError:
                If no models were sampled before this function is called.
        """
        if not self._models:
            raise RuntimeError("no models sampled")

        saved = get_start_method()
        set_start_method("spawn", force=True)
        with Pool() as pool:
            results = [
                pool.apply_async(_apply, (chunk, fn, m))
                for chunk in zip(self._models, self._spaces)
            ]

            for result in results:
                yield result.get()

        set_start_method(saved, force=True)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts this instance of `SampleManager` to a `dict`. This can
        then be used to save and reload this instance for later use. This way
        the operations on the sampled models can be paused and later resumed.

        Returns:
            dict[str, Any]:
                A `dict` containing the content of this manager.
        """
        return {
            "spaces": [
                {
                    "width_mult": space.width_mult,
                    "resolution": space.resolution,
                    "samples": [model.to_dict() for model in models]
                }
                for space, models in zip(self._spaces, self._models)
            ]
        }

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        space_factory: Callable[[float, int], SearchSpace]
    ) -> 'SampleManager':
        """
        Converts a `dict` to a `SampleManager`.

        Args:
            config (dict[str, Any]):
                The `dict` containing the content of a manager to be loaded.

        Returns:
            SampleManager:
                The manager constructed from the `dict`.
        """
        spaces = [
            space_factory(entry["width_mult"], entry["resolution"])
            for entry in config["spaces"]
        ]
        manager = cls(spaces)

        manager._models = [
            [Model.from_dict(sample) for sample in entry.get("samples", [])]
            for entry in config["spaces"]
        ]

        return manager


def _test_and_add_to_top(
    top: list[tuple[OneShotNet, float]],
    population: Iterable[OneShotNet],
    ds: Dataset,
    batch_size: int,
    calib_batch_size: int,
    *,
    batches: int | None = None,
    device=None
) -> None:
    calib_dl = DataLoader(ds, calib_batch_size, shuffle=True)

    for candidate in population:
        candidate.to(device)

        # Recalibrate batch norm statistics (running mean/var).
        candidate.eval()
        for m in candidate.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

        for (images, _) in islice(calib_dl, 20):
            images = images.to(device)
            candidate(images)

        # Insert based on descending accuracy.
        accuracy = skeletonnet_valid(
            candidate, ds, batch_size, batches=batches, device=device
        )
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
        epochs: int,
        batch_size: int,
        models_per_batch: int,
        warm_up: bool,
        warm_up_ctx: WarmUpCtx,
        warm_up_epochs: int,
        warm_up_batch_size: int,
        *,
        batches: int | None = None,
        warm_up_batches: int | None = None,
        device=None
    ) -> None:
        """
        Train the `SuperNet` by sampling random models per batch of training
        data, setting the `SuperNet` to the sampled model and performing the
        backward pass.

        Args:
            epochs (int):
                The number of training epochs.
            batch_size (int):
                The number of samples per batch for training.
            models_per_batch (int):
                The number of models sampled per batch of training data.
            warm_up (bool):
                If set to `True` train the largest sub-networks of the
                `SuperNet` first to start the training with bettwer weights.
            warm_up_ctx (WarmUpCtx):
                The context passed to the warm up function of the super-network.
                Only relevant if `warm_up` is `True`.
            warm_up_epochs (int):
                The number of warm up epochs. Only relevant if `warm_up` is
                `True`.
            warm_up_batch_size (int):
                The number of samples per batch for warm up. Only relevant if
                `warm_up` is `True`.
            batches (int, None):
                The number of batches per epoch. If set to `None` use the whole
                training data set.
            warm_up_batches (int, None):
                The number of batches per warm up epoch. If set to `None` use
                the whole training data set. Only relevant if `warm_up` is
                `True`.
            device:
                The Pytorch device the network is moved to.
        """
        if warm_up:
            supernet_warm_up(
                self._supernet,
                self._space,
                warm_up_ctx,
                self._train_ds,
                warm_up_epochs,
                warm_up_batch_size,
                batches=warm_up_batches,
                device=device
            )

        supernet_train(
            self._supernet,
            self._space,
            self._train_ds,
            self._initial_lr,
            epochs,
            batch_size,
            models_per_batch,
            batches=batches,
            device=device
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
        *,
        batches: int | None = None,
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
            batches (int, None):
                The number of batches used during validation per candidate. If
                set to `None` use the whole validation data set.
            device:
                The Pytorch device the network is moved to.

        Returns:
            tuple[tuple[OneShotNet, float]]:
                The final `topk` many models and their accuracy.
        """
        population = initial_population(
            iter(self._space), population_size, self._supernet, fitness
        )

        rng = np.random.default_rng()
        top: list[tuple[OneShotNet, float]] = []

        n_cross: Final[int] = int(population_size * next_gen_split[0])
        n_mut: Final[int] = int(population_size * next_gen_split[1])

        print(make_caption("Evolution", 70, " "))
        for i in range(iterations):
            print(make_caption(f"Iteration {i + 1}/{iterations}", 70, "-"))

            iteration_start = time.time()

            print(f"iteration={i + 1}, method=test-and-add", end="")

            test_start = time.time()
            _test_and_add_to_top(
                top,
                population,
                self._valid_ds,
                batch_size,
                calib_batch_size,
                batches=batches,
                device=device
            )
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
            print(f"\ntime={iteration_time:.2f}s\n")

        print(make_caption("Final Population", 70, "-"))
        print(f"method=test-and-add", end="")

        test_start = time.time()
        _test_and_add_to_top(
                top,
                population,
                self._valid_ds,
                batch_size,
                calib_batch_size,
                batches=batches,
                device=device
            )
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
        classes = config["classes"]
        if classes != ds.classes:
            raise Exception(
                f"classes mismatch: expected {classes} data set has {ds.classes}"
            )

        width_mult = config["width_mult"]
        resolution = config["resolution"]
        manager = cls(space(width_mult, resolution), ds, supernet)
        manager._supernet.load_state_dict(config["weights"])

        return manager
