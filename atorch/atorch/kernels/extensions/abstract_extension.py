from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class AbstractExtension(ABC):
    def __init__(self):
        self._name = str(self.__class__.__name__).replace("Extension", "")

    @property
    def name(self):
        return self._name

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def load(self) -> Optional[Callable[..., Any]]:
        pass
