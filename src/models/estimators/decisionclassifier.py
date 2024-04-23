from ..base import BaseModel
from sklearn.tree import DecisionTreeClassifier


class DecisionClassifier(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(name='Decision Tree Classifier')
        self.kwargs = kwargs

    def fit(self, X, y) -> None:
        self._model = DecisionTreeClassifier(**self.kwargs)
        self._model.fit(X, y)
