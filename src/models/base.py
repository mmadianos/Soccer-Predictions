import torch
import numpy as np
from abc import ABC, abstractmethod


class ScikitModel(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def fit(self, x, y) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x) -> np.array:
        raise NotImplementedError
    
    def save(self) -> None:
        pass
    
    def load(self) -> None:
        pass

    def __str__(self) -> str:
        return f"{self.name}"
    
class TorchModel(torch.nn.Module, ABC):
    def __init__(self) -> None:
        super(TorchModel, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError
    