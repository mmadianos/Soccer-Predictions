import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.calibration import CalibratedClassifierCV


class BaseModel(ClassifierMixin, BaseEstimator, ABC):
    def __init__(self, name: str, calibrate_probabilities: bool) -> None:
        self.name = name
        self.calibrate_probabilities = calibrate_probabilities
        self._model = None

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def _get_model(self) -> BaseEstimator:
        estimator = self._get_estimator()
        if self.calibrate_probabilities:
            return CalibratedClassifierCV(estimator)
        else:
            return estimator
    
    @abstractmethod
    def _get_estimator(self):
        ...

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
