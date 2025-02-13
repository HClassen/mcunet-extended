import time
import json
from pathlib import Path
from itertools import islice
from typing import Any, Final
from bisect import insort_left
from abc import abstractmethod, ABC
from collections.abc import Callable, Iterator

import torch
from torch.utils.data import DataLoader

import numpy as np

from .searchspace import Block, Model, SearchSpace
from .oneshot import (
    supernet_warm_up,
    supernet_train,
    initial_population,
    crossover,
    mutation,
    evaluate,
    SuperNet,
    WarmUpCtx,
    TrainCtx
)
from .datasets import CustomDataset
from .utils import make_caption, Logger, SilentLogger


__all__ = ["SearchSpace", "SampleManager"]


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
        self._models = [list(islice(space, m)) for space in self._spaces]

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

        for samples, space in zip(self._models, self._spaces):
            sampled = len(samples)
            end = min(m, sampled) if m is not None else sampled

            yield (
                space,
                (
                    fn(model, space.width_mult, space.resolution)
                    for model in samples[:end]
                )
            )

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


class SaveTrainCtx(TrainCtx):
    _path: Path
    _every: int
    _first: bool

    def __init__(
        self, path: str | Path, every: int, first: bool = False
    ) -> None:
        if isinstance(path, str):
            path = Path(path)

        self._path = path
        self._every = every
        self._first = first

        self._path.mkdir(parents=True, exist_ok=True)

    def epoch(self, epoch: int, supernet: SuperNet, epochs: int) -> None:
        if epoch % self._every != 0 and not (epoch == 1 and self._first):
            return

        path = self._path / f"weights-{epoch}.pth"
        with open(path, "wb") as f:
            torch.save(supernet.state_dict(), f)


class EvolutionCtx(ABC):
    @abstractmethod
    def iteration(
        self,
        iteration: int,
        population: list[Model],
        top: list[tuple[Model, float]]
    ) -> None:
        pass


class SaveEvolutionCtx(EvolutionCtx):
    _path: Path
    _every: int
    _first: bool

    def __init__(
        self, path: str | Path, every: int, first: bool = False
    ) -> None:
        if isinstance(path, str):
            path = Path(path)

        self._path = path
        self._every = every
        self._first = first

        self._path.mkdir(parents=True, exist_ok=True)

    def iteration(
        self,
        iteration: int,
        population: list[Model],
        top: list[tuple[Model, float]]
    ) -> None:
        if iteration % self._every != 0 and not (iteration == 1 and self._first):
            return

        base = self._path / f"iteration-{iteration}"

        path_population = base / "population"
        path_population.mkdir(parents=True)
        for i, model in enumerate(population):
            with open(path_population / f"{i}.json", "w") as f:
                json.dump(model.to_dict(), f)

        path_topk = base / f"top-{len(top)}"
        path_topk.mkdir(parents=True)
        for i, (model, accuracy) in enumerate(top):
            with open(path_topk / f"{i}.json", "w") as f:
                data = {"model": model.to_dict(), "accuracy": accuracy}

                json.dump(data, f)


def _add_to_top(
    top: list[tuple[Model, float]],
    population: list[Model],
    accuracies: list[float]
) -> None:
    for candidate, accuracy in zip(population, accuracies):
        insort_left(
            top, (candidate, accuracy), key=lambda entry: -entry[1]
        )


class SearchManager():
    _space: SearchSpace
    _classes: int
    _supernet: SuperNet

    _initial_lr: float

    _train_ds: CustomDataset
    _valid_ds: CustomDataset

    _logger: Logger

    def __init__(
        self,
        space: SearchSpace,
        train_ds: CustomDataset,
        valid_ds: CustomDataset,
        supernet: SuperNet,
        logger: Logger | None = None
    ) -> None:
        self._space = space
        self._supernet = supernet
        self._initial_lr = 0.05

        if train_ds.classes != valid_ds.classes:
            raise ValueError(
                f"train classes {train_ds.classes} != valid classes {valid_ds.classes}"
            )
        self._train_ds = train_ds
        self._valid_ds = valid_ds

        self._logger = logger if logger is not None else SilentLogger()

    def train(
        self,
        epochs: int,
        dl_config: dict[str, Any],
        models_per_batch: int,
        warm_up: bool,
        warm_up_ctx: WarmUpCtx | None = None,
        warm_up_epochs: int = 0,
        warm_up_dl_config: dict[str, Any] = {},
        *,
        ctx: TrainCtx | None = None,
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
            dl_config (dict[str, Any]):
                Configure the data loader for the training data. Gets passes
                directly to the constructor of `torch.utils.data.DataLoader`.
            models_per_batch (int):
                The number of models sampled per batch of training data.
            warm_up (bool):
                If set to `True` train the largest sub-networks of the
                `SuperNet` first to start the training with bettwer weights.
            warm_up_ctx (WarmUpCtx, None):
                The context passed to the warm up function of the super-network.
                Only relevant if `warm_up` is `True`.
            warm_up_epochs (int):
                The number of warm up epochs. Only relevant if `warm_up` is
                `True`.
            warm_up_dl_config (dict[str, Any]):
                Configure the data loader for the warm up data. Gets passes
                directly to the constructor of `torch.utils.data.DataLoader`.
                Only relevant if `warm_up` is `True`.
            ctx (TrainCtx, None):
                The context passed to the train function of the super-network.
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
        if warm_up and warm_up_ctx is not None:
            warm_up_dl = DataLoader(self._train_ds, **warm_up_dl_config)
            supernet_warm_up(
                self._supernet,
                self._space,
                warm_up_ctx,
                warm_up_dl,
                warm_up_epochs,
                logger=self._logger,
                batches=warm_up_batches,
                device=device
            )

        train_dl = DataLoader(self._train_ds, **dl_config)
        supernet_train(
            self._supernet,
            self._space,
            train_dl,
            self._initial_lr,
            epochs,
            models_per_batch,
            logger=self._logger,
            ctx=ctx,
            batches=batches,
            device=device
        )

    def evolution(
        self,
        valid_dl_config: dict[str, Any],
        calib_dl_config: dict[str, Any],
        topk: int,
        iterations: int,
        population_size: int,
        next_gen_split: tuple[float, float] = (0.5, 0.5),
        mutation_rate: float = 0.1,
        mutator: Callable[[int, float, Block], Block] = lambda a, b, c: c,
        fitness: Callable[[Model], bool] = lambda _: True,
        *,
        ctx: EvolutionCtx | None = None,
        valid_batches: int | None = None,
        device=None
    ) -> list[tuple[Model, float]]:
        """
        Perform evolution search on the `SuperNet` to find the model with the
        highest accuracy. Per iteration, single-point crossover and mutation is
        performed on the current population. Before the accuracy of a model is
        tested, calibrate the batch-normalisation layers on the validation data.

        Args:
            valid_dl_config (dict[str, Any]):
                Configure the data loader for the validation data. Gets passes
                directly to the constructor of `torch.utils.data.DataLoader`.
            calib_dl_config (dict[str, Any]):
                Configure the data loader for the calibration data. Gets passes
                directly to the constructor of `torch.utils.data.DataLoader`.
            topk (int):
                The number of models with the highest accuracy to keep track off.
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
            mutator (Callable[[int, float, Block], Block]):
                The mutator function passed to the `mutation` function of the
                evolution search.
            fitness (Callable[[Model], bool]):
                Evaluate the fitness of a sampled model on some criteria. Only
                models where `fitness` returns `True` are selected for the
                population of the next generation.
            ctx (EvolutionCtx, None):
                The evolution context.
            valid_batches (int, None):
                The number of batches used during validation per candidate. If
                set to `None` use the whole validation data set.
            device:
                The Pytorch device the network is moved to.

        Returns:
            list[tuple[Model, float]]:
                The final `topk` many models and their accuracy.
        """
        evo_start = time.time()

        valid_dl = DataLoader(self._valid_ds, **valid_dl_config)
        calib_dl = DataLoader(self._valid_ds, **calib_dl_config)

        rng = np.random.default_rng()
        top: list[tuple[Model, float]] = []

        n_cross: Final[int] = int(population_size * next_gen_split[0])
        n_mut: Final[int] = int(population_size * next_gen_split[1])

        self._logger.log(make_caption("Evolution", 70, " "))

        population = initial_population(
            iter(self._space), population_size, fitness
        )

        for i in range(iterations):
            self._logger.log(
                make_caption(f"Iteration {i + 1}/{iterations}", 70, "-")
            )

            iteration_start = time.time()

            self._logger.log(f"iteration={i + 1}, method=evaluate", end="")

            test_start = time.time()
            accuracies = evaluate(
                population,
                self._supernet,
                valid_dl,
                calib_dl,
                valid_batches=valid_batches,
                calib_batches=20,
                device=device
            )
            test_time = time.time() - test_start

            self._logger.log(f", time={test_time:.2f}s")

            _add_to_top(top, population, accuracies)
            top = top[:topk]

            if ctx is not None:
                ctx.iteration(i + 1, population, top)

            self._logger.log(f"iteration={i + 1}, method=crossover", end="")
            crossover_start = time.time()

            cross: list[Model] = []
            while len(cross) < n_cross:
                p1 = top[rng.integers(0, topk)][0]
                p2 = top[rng.integers(0, topk)][0]

                c1, c2 = crossover(p1, p2)

                if fitness(c1):
                    cross.append(c1)

                if fitness(c2):
                    cross.append(c2)

            # If n_cross % 2 = 1 then adding two models per iteration results
            # in len(cross) == n_cross + 1.
            cross = cross[:n_cross]

            crossover_time = time.time() - crossover_start
            self._logger.log(f", time={crossover_time:.2f}s")

            self._logger.log(f"iteration={i + 1}, method=mutate", end="")
            mutate_start = time.time()

            mut: list[Model] = []
            while len(mut) != n_mut:
                mutated = mutation(
                    top[rng.integers(0, topk)][0], mutation_rate, mutator
                )

                if fitness(mutated):
                    mut.append(mutated)

            mutate_time = time.time() - mutate_start
            self._logger.log(f", time={mutate_time:.2f}s")

            population = cross + mut

            iteration_time = time.time() - iteration_start
            self._logger.log(f"\ntime={iteration_time:.2f}s\n")

        self._logger.log(make_caption("Final Population", 70, "-"))
        self._logger.log(f"method=evaluate", end="")

        test_start = time.time()
        accuracies = evaluate(
            population,
            self._supernet,
            valid_dl,
            calib_dl,
            valid_batches=valid_batches,
            calib_batches=20,
            device=device
        )
        test_time = time.time() - test_start
        self._logger.log(f", time={test_time:.2f}s")

        _add_to_top(top, population, accuracies)
        top = top[:topk]

        if ctx is not None:
            ctx.iteration(-1, population, top)

        evo_time = time.time() - evo_start
        self._logger.log(f"\ntotal={evo_time:.2f}s\n")

        return top

    def to_dict(self) -> dict[str, Any]:
        """
        Converts this instance of `SearchManager` to a `dict`. This can then be
        used to save and reload this instance for later use. This way the
        training and evolution search can be paused and later resumed. It
        doesn't save the state of the `Dataset`.

        Returns:
            dict[str, Any]:
                A `dict` containing the content of this manager.
        """
        self._supernet.unset()
        return {
            "width_mult": self._space.width_mult,
            "resolution": self._space.resolution,
            "classes": self._train_ds.classes,
            "weights": self._supernet.state_dict()
        }

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        space: Callable[[float, int], SearchSpace],
        train_ds: CustomDataset,
        valid_ds: CustomDataset,
        supernet: SuperNet,
        logger: Logger | None = None
    ) -> 'SearchManager':
        """
        Converts a `dict` to a `SearchManager`.

        Args:
            config (dict[str, Any]):
                The `dict` containing the content of a manager to be loaded.
            space: (Callable[[float, int], SearchSpace]):
                A function creating a `SearchSpace`.
            train_ds (CustomDataset):
                The training data set.
            valid_ds (CustomDataset):
                The validation data set.
            supernet (SuperNet):
                The supernet.
            logger (Logger, None):
                An optional logger.

        Returns:
            SearchManager:
                The manager constructed from the `dict`.
        """
        classes = config["classes"]
        if classes != train_ds.classes:
            raise ValueError(
                f"train classes {train_ds.classes} != config classes {classes}"
            )

        if classes != valid_ds.classes:
            raise ValueError(
                f"valid classes {valid_ds.classes} != config classes {classes}"
            )

        width_mult = config["width_mult"]
        resolution = config["resolution"]
        manager = cls(
            space(width_mult, resolution), train_ds, valid_ds, supernet, logger
        )
        manager._supernet.load_state_dict(config["weights"])

        return manager
