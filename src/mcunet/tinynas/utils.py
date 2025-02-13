from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np

import matplotlib.pyplot as plt


__all__ = [
    "make_divisible",
    "make_caption", "Logger", "SilentLogger", "ConsoleLogger", "FileLogger",
    "EDF"
]


def make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    """
    Change ``v`` so that it is divisble by ``divisor``. Copied from
    https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py.

    Args:
        v (float):
            The value.
        divisor (int):
            The newe divisor for ``v``.
        min_value (int, None):
            Lower bound for ``v`` to not go below.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def make_caption(text: str, line_length: int, divider: str) -> str:
    diff = line_length - len(text)
    dashes = int((diff - 2) / 2) * divider

    caption = f"{dashes} {text} {dashes}"
    return f"{caption}{divider * (line_length - len(caption))}"


class Logger(ABC):
    @abstractmethod
    def log(self, msg: str, end: str = "\n") -> None:
        pass


class SilentLogger(Logger):
    def log(self, msg: str, end: str = "\n") -> None:
        pass


class ConsoleLogger(Logger):
    def log(self, msg: str, end: str = "\n") -> None:
        print(msg, end=end)


class FileLogger(Logger):
    _path: Path

    def __init__(self, path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)

        self._path = path
        self._file = open(self._path, "w")

    def __del__(self) -> None:
        self._file.close()

    def log(self, msg: str, end: str = "\n") -> None:
        self._file.write(msg)
        self._file.write(end)
        self._file.flush()


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
            x (float, list[float], np.ndarray):
                The x values(s) to evaluate the function for.

        Returns:
            float, list[float], np.ndarray:
                The y values(s) of the function.
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

    def show(
        self, sample: int | None = None, ax: plt.Axes | None = None
    ) -> None:
        """
        Plots this EDF.

        Args:
            sample (int, None):
                The number of x values to plot for. The default is the amount of
                samples provided in the constructor.
            ax (matplotlib.pyplot.Axes, None):
                A pyplot Axes to draw on.
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
            p (float):
                The percentile.

        Returns:
            int:
                The FLOPs.
        """
        steps = self._events / self._sorted.size

        for i in range(steps.size):
            idx = steps.size - i - 1

            if p <= steps[idx]:
                return self._buckets[idx]

        return self._buckets[0]
