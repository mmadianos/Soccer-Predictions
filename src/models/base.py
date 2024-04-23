import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class BaseModel(BaseEstimator, ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self._model = None

    @abstractmethod
    def fit(self, x, y) -> np.array:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise NotFittedError("Model has not been trained yet. Please call fit() before predict().")
        return self._model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise NotFittedError("Model has not been trained yet. Please call fit() before predict_proba().")
        return self._model.predict_proba(X)
    
    @property
    def classes_(self) -> np.array:
        if self._model is None:
            raise NotFittedError("Model has not been trained yet. No classes available.")
        return self._model.classes_
    
    @property
    def feature_importances_(self) -> np.array:
        if self._model is None:
            raise NotFittedError("Model has not been trained yet. No feature importances available.")
        return self._model.feature_importances_
    
    def save(self) -> None:
        pass
    
    def load(self) -> None:
        pass

    def __str__(self) -> str:
        return f"{self.name}"
