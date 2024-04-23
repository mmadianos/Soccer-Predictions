import lightgbm as lgb
from ..base import BaseModel


class LightGBMClassifier(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(name='LightGBM Classifier')
        self.kwargs = kwargs

    def fit(self, X, y) -> None:
        self._model = lgb.LightGBMClassifier(**self.kwargs)
        self._model.fit(X, y)