from ..base import BaseModel
from sklearn.tree import DecisionTreeClassifier


class DecisionClassifier(BaseModel):
    _parameter_space = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [1, 20],
            'min_samples_leaf': [1, 20],
            'min_samples_split': [1, 20],
            'max_features': [0.1, 1.0]
        }
    def __init__(self, **kwargs) -> None:
        super().__init__(name='Decision Tree Classifier')
        self._model = DecisionTreeClassifier(**kwargs)   

    def fit(self, X, y) -> None:
        self._model.fit(X, y)
        return self
    
    @classmethod
    def get_parameter_space(cls):
        return cls._parameter_space