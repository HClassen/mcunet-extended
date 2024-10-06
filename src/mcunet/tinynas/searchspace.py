from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np

import matplotlib.pyplot as plt

from .mnasnet import searchspace as mnasnet

def sample(
    n_classes: int,
    width_mult: float = 1.0,
    resolution: int = 224,
    dropout: float = 0.2,
    m: int = 1000
) -> Iterator[mnasnet.Model]:
    """
    Provides ``m`` samples for the search space given by ``width_mult`` and
    ``resolution``.

    Args:
        n_classes (int): The number of classes the networks use.
        width_mult (float): Modifies the input and output channels of the hidden layers.
        resolution (int): The resolution of the input image.
        dropout (float): Dropout rate for the classifier of MobileNetV2.
        m (int): The amount of samples generated.

    Returns:
        Iterator[x.searchspace.Model]: A generator to fetch the sampled models.
    """
    size = range(m)
    rng = np.random.default_rng()

    for _ in size:
        yield mnasnet.sample_model(
            rng, n_classes, width_mult, resolution, dropout
        )


@dataclass(frozen=True)
class Configuration():
    width_mult: float
    resolution: int


def configurations() -> list[Configuration]:
    width_step = 16
    width_choices = [48 + i * width_step for i in range(12)]

    resolution_step = 0.1
    resolution_choices = [0.2 + i * resolution_step for i in range(9)]

    # cross product to get all combinations
    cross = np.array(
        np.meshgrid(width_choices, resolution_choices)
    ).T.reshape(-1, 2)

    return [Configuration(float(w), int(r)) for r, w in cross]



class EDF():
    _sorted: np.ndarray
    _buckets: np.ndarray
    _events: np.ndarray

    def __init__(self, samples: list[int] | np.ndarray) -> None:
        if isinstance(samples, list):
            samples = np.asarray(samples)

        self._sorted = np.sort(samples)
        self._buckets, counts = np.unique(self._sorted, return_counts=True)
        self._events = np.cumsum(counts)

    def __call__(
        self, x: int | list[int] | np.ndarray
    ) -> float | list[float] | np.ndarray:
        """
        Evaluates the EDF (Estimated Distribution Function) for ``x``. The function
        is defined as:
        .. math::
            F(x) = 1/n * \sum_{i = 0}^{n} 1[x_i < x]

        :math:`F(x)` returns the fraction of values in samples less than x.
        The return value mirrors the input type.

        Args:
            x (float, list[float], np.ndarray): The x values(s) to evaluate the function for.

        Returns:
            float, list[float], np.ndarray: The y values(s) of the function.
        """
        if isinstance(x, int):
            xs = np.asarray([x])
        elif isinstance(x, list):
            xs = np.asarray(x)
        else:
            xs = x

        ys = self._evaluate(xs)

        if isinstance(x, int):
            return float(ys[0])
        elif isinstance(x, list):
            return float(list(ys))
        else:
            return ys

    def _evaluate(self, xs: np.ndarray) -> np.ndarray:
        # Expand the spaced input to make use of broadcasting. Switch the
        # dimensions of the matrix since we want the transpose.
        expanded = np.broadcast_to(xs, (self._buckets.size, xs.size)).T

        # An array containing the indices to self._events for each entry of
        # spaced.
        idxs = np.argmax(self._buckets >= expanded, axis=1)

        # Special case: if x is larger than the max sample, self._buckets >= x
        # returns [False, ...] and thus np.argmax() returns 0. But it needs to
        # be self._buckets.size - 1, as 100% of samples are below this x.
        for i, idx in enumerate(idxs):
            if idx == 0 and xs[i] > self._buckets[-1]:
                idxs[i] = self._buckets.size - 1

        return self._events[idxs] / self._sorted.size

    def show(self, sample: int | None = None, ax: plt.Axes | None = None) -> None:
        """
        Plots this EDF.

        Args:
            sample (int, None): The number of x values to plot for. The default
                                is the amount of samples provided in the constructor.
            ax (matplotlib.pyplot.Axes, None): A pyplot Axes to draw on.
        """
        if sample is None:
            sample = self._sorted.size

        spaced = np.linspace(
            start=self._buckets[0],
            stop=self._buckets[-1],
            num=sample,
            dtype=np.int32
        )

        if ax is not None:
            ax.plot(spaced, self._evaluate(spaced))
        else:
            plt.plot(spaced, self._evaluate(spaced))

    def percentile(self, p: float) -> int:
        """
        Gets the FLOPs for which ``p``% models are equal or below.

        Args:
                p (float): The percentile.

        Returns:
                int: The FLOPs.
        """
        steps = self._events / self._sorted.size

        for i in range(steps.size):
            idx = steps.size - i - 1

            if p <= steps[idx]:
                return self._buckets[idx]

        return self._buckets[0]
