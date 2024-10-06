from typing import Final
from collections.abc import Iterator

import numpy as np

from .mnasnet import sample_model, Model


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
