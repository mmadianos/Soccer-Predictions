from ..base import BaseModel
from sklearn.svm import SVC

    
class KNNClassifier(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(name='Support Vector Machine Classifier')
        self.kwargs = kwargs

    def fit(self, X, y) -> None:
        self._model = SVC(**self.kwargs)
        self._model.fit(X, y)
