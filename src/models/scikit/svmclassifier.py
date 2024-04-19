from numpy import array
from ..base import ScikitModel
from sklearn.svm import SVC


class SVMClassifier(ScikitModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(name='Support Vector Machine Classifier')
        self._model = SVC(**kwargs)

    def fit(self, X, y):
        self._model.fit(X, y)
    
    def predict(self, x) -> array:
        return self._model.predict(x)