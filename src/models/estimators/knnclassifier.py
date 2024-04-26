from ..base import BaseModel
from sklearn.neighbors import KNeighborsClassifier

    
class KNNClassifier(BaseModel):
    _parameter_space = {

        }
    def __init__(self, **kwargs) -> None:
        super().__init__(name='KNN Classifier')
        self._model = KNeighborsClassifier(**kwargs)

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    @classmethod
    def get_parameter_space(cls):
        return cls._parameter_space
    