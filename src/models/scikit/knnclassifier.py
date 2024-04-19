from numpy import array
from ..base import ScikitModel
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier(ScikitModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(name='KNN Classifier')
        self._model = KNeighborsClassifier(**kwargs)

    def fit(self, X, y):
        self._model.fit(X, y)
    
    def predict(self, x) -> array:
        return self._model.predict(x)