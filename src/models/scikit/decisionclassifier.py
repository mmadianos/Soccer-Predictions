from numpy import array
from ..base import ScikitModel
from sklearn.tree import DecisionTreeClassifier


class DecisionClassifier(ScikitModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(name='Decision Tree Classifier')
        self._model = DecisionTreeClassifier(**kwargs)

    def fit(self, X, y):
        self._model.fit(X, y)
    
    def predict(self, x) -> array:
        return self._model.predict(x)
    
    def predict_proba(self, x):
        return self._model.predict_proba(x)
    
    @property
    def classes_(self):
        return self._model.classes_