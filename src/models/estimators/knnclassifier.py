from ..base import BaseModel
from sklearn.neighbors import KNeighborsClassifier

    
class KNNClassifier(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(name='KNN Classifier')
        self.kwargs = kwargs

    def fit(self, X, y) -> None:
        self._model = KNeighborsClassifier(**self.kwargs)
        self._model.fit(X, y)
