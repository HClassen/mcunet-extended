import time
from itertools import islice
from bisect import insort_left
from operator import itemgetter
from typing import cast, Any, Final
from collections.abc import Callable, Iterator, Iterable

import torch
import torch.nn as nn
import torch.linalg as la
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

from .mnasnet import (
    KERNEL_CHOICES,
    SE_CHOICES,
    LAYER_CHOICES,
    CHANNEL_CHOICES,
    EXPANSION_CHOICES,
    sample_model,
    uniform_model,
    ConvOp,
    SkipOp,
    Model
)
from .mnasnet.layers import BaseOp
from .mnasnet.model import build_model
from .oneshot import (
    initial_population,
    crossover,
    mutate,
    OneShotNet,
    SuperNet
)
from .oneshot.layers import SuperBlock, BaseChoiceOp
from .datasets import CustomDataset
from .utils import EDF
from .utils.torchhelper import get_device, test


__all__ = [
    "WIDTH_CHOICES", "RESOLUTION_CHOICES",
    "configurations",
    "SearchSpace", "SampleManager"
]


# The selected choices for the width multiplier.
_width_step = 0.1
WIDTH_CHOICES: Final[list[float]] = [0.2 + i * _width_step for i in range(9)]

# The selected choices for the resolution.
_resolution_step = 16
RESOLUTION_CHOICES: Final[list[int]] = [
    48 + i * _resolution_step for i in range(12)
]


def configurations() -> Iterator[tuple[float, int]]:
    """
    Creates all 108 combinations of alpha (width multiplier) and rho (resolution)
    as per the MCUNet paper.

    Returns:
        Iterator[tuple[float, int]]:
            An iterator of all combinations.
    """
    for with_mult in WIDTH_CHOICES:
        for resolution in RESOLUTION_CHOICES:
            yield (with_mult, resolution)


class SearchSpace():
    """
    Represents a search space with a given width multiplier and resolution. It
    implements the ``Iterator`` pattern to randomly sample models.
    """
    width_mult: float
    resolution: int

    _rng: np.random.Generator | None

    def __init__(self, width_mult: float, resolution: int) -> None:
        self.width_mult = width_mult
        self.resolution = resolution

        self._rng = None

    def oneshot(self) -> Model:
        """
        Samples one model from the search space. Meant to be used in a oneshot
        way, where only one model is needed.

        Returns:
            Model:
                A randomly sampled model from the MnasNet searchspace.
        """
        rng = np.random.default_rng()
        return sample_model(rng, self.width_mult)

    def __iter__(self) -> 'SearchSpace':
        if self._rng is None:
            self._rng = np.random.default_rng()

        return self

    def __next__(self) -> Model:
        if self._rng is None:
            raise StopIteration()

        return sample_model(self._rng, self.width_mult)


class SampleManager():
    """
    Handle sampling models from different search spaces and extracting different
    statistics from them. Not all search spaces and/or models must be iterated
    over at once. An object from this class can be serialized/deserialized and
    then continued.
    """
    _spaces: list[SearchSpace]
    _classes: int
    _models: list[list[mnasnet.Model]] | None
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


def _sort_channels(weight: torch.Tensor) -> torch.Tensor:
    # Sort channels by L1 norm. The higher the better.
    importance: list[tuple[int, float]] = []
    for channel in range(weight.size()[0]):
        l1 = la.norm(weight[channel].view(-1), ord=1)
        insort_left(
            importance, (channel, l1), key=lambda entry: -entry[1]
        )

    reordered = torch.empty_like(weight)
    with torch.no_grad():
        for k, (channel, _) in enumerate(importance):
            reordered[k].copy_(weight[channel])

    return reordered


def _get_warm_up_common(
    src: BaseOp
) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter]:
    weight_conv2d = _sort_channels(src.layers["conv2d"][0].weight)
    weight_fc1 = _sort_channels(src.layers["se"].fc1.weight)
    weight_fc2 = _sort_channels(src.layers["se"].fc2.weight)

    return (weight_conv2d, weight_fc1, weight_fc2)


def _set_warm_up_conv2d(src: BaseOp, layer: BaseChoiceOp) -> None:
    weight_conv2d, weight_fc1, weight_fc2 = _get_warm_up_common(src)

    layer.set_weights(weight_conv2d, weight_fc1, weight_fc2)


def _set_warm_up_dwconv2d(src: BaseOp, layer: BaseChoiceOp) -> None:
    weight_conv2d, weight_fc1, weight_fc2 = _get_warm_up_common(src)
    weight_spconv2d = _sort_channels(src.layers["spconv2d"].weight)

    layer.set_weights(weight_conv2d, weight_fc1, weight_fc2, weight_spconv2d)


def _set_warm_up_mbconv2d(src: BaseOp, layer: BaseChoiceOp) -> None:
    weight_conv2d, weight_fc1, weight_fc2 = _get_warm_up_common(src)
    weight_expand = _sort_channels(src.layers["expand"][0].weight)
    weight_spconv2d = _sort_channels(src.layers["spconv2d"].weight)

    layer.set_weights(
        weight_expand, weight_conv2d, weight_fc1, weight_fc2, weight_spconv2d
    )


def _adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epochs: int,
    batch: int,
    batches: int,
    initial_lr: float
) -> float:
    # Adjust the learning rate based on cosine annealing.
    total = epochs * batches
    cur = epoch * batches + batch
    adjusted_lr = 0.5 * initial_lr * (1 + np.cos(np.pi * cur / total))

    for group in optimizer.param_groups:
        group["lr"] = adjusted_lr

    return adjusted_lr


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
        self,
        space: SearchSpace | tuple[float, int],
        ds: CustomDataset,
        supernet: SuperNet | None = None
    ) -> None:
        if isinstance(space, tuple):
            space = SearchSpace(*space)

        self._space = space

        if supernet is None:
            supernet = SuperNet(ds.classes, self._space.width_mult)

        self._supernet = supernet
        self._initial_lr = 0.05

        self._ds = ds
        self._train_ds, self._valid_ds = random_split(ds, [0.85, 0.15])

    def _warm_up(
        self,
        op: ConvOp,
        warm_up_epochs: int,
        warm_up_batches: int | None,
        batch_size: int,
        device=None
    ) -> None:
        max_model = uniform_model(
            op,
            max(KERNEL_CHOICES),
            max(SE_CHOICES),
            SkipOp.NOSKIP,
            max(LAYER_CHOICES),
            max(CHANNEL_CHOICES),
            max(EXPANSION_CHOICES),
            self._space.width_mult
        )

        net = build_model(max_model, self._ds.classes)

        net.to(device)

        dl = DataLoader(self._train_ds, batch_size, shuffle=True)
        batches = len(dl) if warm_up_batches is None else warm_up_batches

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            net.parameters(), momentum=0.9, weight_decay=5e-5
        )

        net.train()

        print(f"{' ' * 30} Warmup {str(op)}")
        for i in range(warm_up_epochs):
            print(f"{'-' * 30} Epoch {i + 1}/{warm_up_epochs} {'-' * 30}")

            epoch_start = time.time()

            for k, (images, labels) in enumerate(dl):
                print(f"epoch={i + 1}, batch={k + 1:03}/{batches}", end="")
                batch_start = time.time()

                images = images.to(device)
                labels = labels.to(device)

                outputs = net(images)
                loss = criterion(outputs, labels)

                net.zero_grad()

                loss.backward()
                optimizer.step()

                batch_time = time.time() - batch_start

                print(f", time={batch_time:.2f}s")

                if batches == k + 1:
                    break

            epoch_time = time.time() - epoch_start

            print()
            print(f"time={epoch_time:.2f}s")
            print()

        for i, block in enumerate(self._supernet.blocks):
            block = cast(SuperBlock, block)

            choice = block.get_choice_block(op)
            for j, layer in enumerate(choice.layers):
                layer = cast(BaseChoiceOp, layer)
                src = cast(BaseOp, net.blocks[i * max(LAYER_CHOICES) + j])

                match op:
                    case ConvOp.CONV2D:
                        _set_warm_up_conv2d(src, layer)
                    case ConvOp.DWCONV2D:
                        _set_warm_up_dwconv2d(src, layer)
                    case ConvOp.MBCONV2D:
                        _set_warm_up_mbconv2d(src, layer)

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
        Train the ``SuperNet`` by sampling random models per batch of training
        data, setting the ``SuperNet`` to the sampled model and performing the
        backward pass.

        Args:
            epochs (int):
                The number of training epochs.
            batches (int, None):
                The number of batches per epoch. If set to ``None`` use the whole
                training data set.
            batch_size (int):
                The number of samples per batch for training.
            models_per_batch (int):
                The number of models sampled per batch of training data.
            warm_up (bool):
                If set to ``True`` train the largest sub-networks of the
                ``SuperNet`` first to start the training with bettwer weights.
            warm_up_batches (int, None):
                The number of batches per warm up epoch. If set to ``None`` use
                the whole training data set. Only relevant if ``warm_up`` is
                ``True``.
            warm_up_epochs (int):
                The number of warm up epochs. Only relevant if ``warm_up`` is
                ``True``.
            warm_up_batch_size (int):
                The number of samples per batch for warm up. Only relevant if
                ``warm_up`` is ``True``.
        """
        if warm_up:
            self._warm_up(
                ConvOp.CONV2D, warm_up_epochs, warm_up_batches, warm_up_batch_size, device
            )
            self._warm_up(
                ConvOp.DWCONV2D, warm_up_epochs, warm_up_batches, warm_up_batch_size, device
            )
            self._warm_up(
                ConvOp.MBCONV2D, warm_up_epochs, warm_up_batches, warm_up_batch_size, device
            )

        models = iter(self._space)

        self._supernet.to(device)

        dl = DataLoader(self._train_ds, batch_size, shuffle=True)
        batches = len(dl) if batches is None else batches

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self._supernet.parameters(), lr=self._initial_lr
        )

        self._supernet.train()

        print(f"{' ' * 30} Training")
        for i in range(epochs):
            print(f"{'-' * 30} Epoch {i + 1}/{epochs} {'-' * 30}")

            epoch_start = time.time()

            for k, (images, labels) in enumerate(dl):
                lr = _adjust_learning_rate(
                    optimizer, i, epochs, k, batches, self._initial_lr
                )

                print(f"epoch={i + 1}, batch={k + 1:03}/{batches}, lr={lr:.5f}", end="")

                batch_start = time.time()

                images = images.to(device)
                labels = labels.to(device)

                for _ in range(models_per_batch):
                    model = next(models)
                    self._supernet.set(model)

                    outputs = self._supernet(images)
                    loss = criterion(outputs, labels)

                    self._supernet.zero_grad()

                    loss.backward()
                    optimizer.step()

                batch_time = time.time() - batch_start

                print(f", time={batch_time:.2f}s")

                if batches == k + 1:
                    break

            epoch_time = time.time() - epoch_start

            print()
            print(f"time={epoch_time:.2f}s")
            print()

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
    ) -> Iterable[tuple[OneShotNet, float]]:
        dl = DataLoader(self._valid_ds, batch_size, shuffle=True)
        calib_dl = DataLoader(self._valid_ds, calib_batch_size, shuffle=True)

        population = initial_population(
            iter(self._space), population_size, self._supernet, fitness
        )

        rng = np.random.default_rng()
        top: list[tuple[OneShotNet, float]] = []

        n_cross: Final[int] = int(population_size * next_gen_split[0])
        n_mut: Final[int] = int(population_size * next_gen_split[1])

        print(f"{' ' * 30} Evolution")
        for i in range(iterations):
            print(f"{'-' * 30} Iteration {i + 1}/{iterations} {'-' * 30}")

            iteration_start = time.time()

            print(f"iteration={i + 1}, method=test-and-add", end=", ")
            test_start = time.time()

            _test_and_add_to_top(top, population, dl, calib_dl, device)
            top = top[:topk]

            test_time = time.time() - test_start
            print(f"time={test_time:.2f}s")

            print(f"iteration={i + 1}, method=crossover", end=", ")
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
            print(f"time={crossover_time:.2f}s")

            print(f"iteration={i + 1}, method=mutate", end=", ")
            mutate_start = time.time()

            mut: list[OneShotNet] = [
                mutate(top[rng.integers(0, topk)][0], mutation_rate)
                for _ in range(n_mut)
            ]

            mutate_time = time.time() - mutate_start
            print(f"time={mutate_time:.2f}s")

            population = cross + mut

            iteration_time = time.time() - iteration_start
            print()
            print(f"time={iteration_time:.2f}s")
            print()

        _test_and_add_to_top(top, population, dl, calib_dl, device)
        return top[:topk]

    def to_dict(self) -> dict[str, Any]:
        """
        Converts this instance of ``SearchManager`` to a ``dict``. This can
        then be used to save and reload this instance for later use. This way
        the training and evolution search can be paused and later resumed. It
        doesn't save the state of the ``Dataset``.

        Returns:
            dict[str, Any]:
                A ``dict`` containing the content of this manager.
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
        ds: CustomDataset,
        batch_size: int = 256
    ) -> 'SearchManager':
        """
        Converts a ``dict`` to a ``SearchManager``.

        Args:
            config (dict[str, Any]):
                The ``dict`` containing the content of a manager to be loaded.
            ds (Dataset):
                The training and validation data.
            batch_size (int):
                The batch size to use during training.

        Returns:
            SearchManager:
                The manager constructed from the ``dict``.
        """
        width_mult, resolution, classes, weights = \
            itemgetter("width_mult", "resolution", "classes", "weights")(config)

        if classes != ds.classes:
            raise Exception(
                f"classes mismatch: expected {classes} data set has {ds.classes}"
            )

        manager = cls((width_mult, resolution), ds, batch_size, None)
        manager._supernet.load_state_dict(weights)

        return manager
