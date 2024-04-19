from numpy import array
from ..base import ScikitModel
from sklearn.linear_model import RidgeClassifier


class RidgeRegClassifier(ScikitModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(name='Ridge Classifier')
        self._model = RidgeClassifier(**kwargs)

    def fit(self, X, y):
        self._model.fit(X, y)
    
    def predict(self, x) -> array:
        return self._model.predict(x)