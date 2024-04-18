from numpy import array
from ..base import ScikitModel
from sklearn.tree import DecisionTreeClassifier

class DecisionTree(ScikitModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(name='DTC')
        self._model = DecisionTreeClassifier(**kwargs)

    def fit(self, X, y):
        self._model.fit(X, y)
    
    def predict(self, x) -> array:
        return self._model.predict(x)