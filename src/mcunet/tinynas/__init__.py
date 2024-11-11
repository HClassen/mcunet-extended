from typing import Any
from itertools import islice
from collections.abc import Callable, Iterator

from .searchspace import Model, SearchSpace


__all__ = ["SampleManager"]


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
