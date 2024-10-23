from abc import ABC, abstractmethod

from torch.utils.data import Dataset


__all__ = ["CustomDataset"]


class CustomDataset(ABC, Dataset):
    """
    Base class for all data sets to use. Creates a uniform interface.
    """
    @property
    @abstractmethod
    def classes(self) -> int:
        pass
