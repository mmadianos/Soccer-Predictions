from ..base import BaseModel
from sklearn.linear_model import RidgeClassifier


class KNNClassifier(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(name='Ridge Classifier')
        self.kwargs = kwargs

    def fit(self, X, y) -> None:
        self._model = RidgeClassifier(**self.kwargs)
        self._model.fit(X, y)
