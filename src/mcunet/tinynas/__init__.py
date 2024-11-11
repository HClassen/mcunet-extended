from itertools import islice
from typing import Any, Final
from operator import itemgetter
from collections.abc import Iterator, Iterable

import numpy as np

from .searchspace import sample_model, Model
from .searchspace.model import build_model
from .utils import EDF
from .utils.torchhelper import get_device


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
        return sample_model(rng, self.width_mult, self.resolution)

    def __iter__(self) -> 'SearchSpace':
        if self._rng is None:
            self._rng = np.random.default_rng()

        return self

    def __next__(self) -> Model:
        if self._rng is None:
            raise StopIteration()

        return sample_model(self._rng, self.width_mult, self.resolution)


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
            "models": [[m.to_dict() for m in lm] for lm in self._models],
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
        models, flops, index = itemgetter("models", "flops", "index")(config)

        models = [[Model.from_dict(m) for m in lm] for lm in models]

        spaces = [(lm[0].width_mult, lm[0].resolution) for lm in models]

        manager = cls(spaces)
        manager._models = models
        manager._flops = flops
        manager._index = index

        return manager
