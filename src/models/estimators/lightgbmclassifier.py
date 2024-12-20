from lightgbm import LGBMClassifier
from typing import Optional


class LightGBMClassifier(LGBMClassifier):
    _parameter_space = {
        'num_leaves': [5, 100],
        'max_depth': [3, 20],
        'learning_rate': [1e-4, 1e-1],
        'min_child_samples': [10, 200],
        'min_child_weight': [1e-5, 1e4],
        'subsample': [0.3, 1.],
        'colsample_bytree': [0.3, 1.],
        'reg_alpha': [1e-2, 10.],
        'reg_lambda': [1e-2, 10.],
        'n_estimators': [50, 1000],
    }

    def __init__(self,
                 name='LightGBMClassifier',
                 random_state: Optional[int] = None,
                 num_leaves: int = 31,
                 max_depth: Optional[int] = -1,
                 learning_rate: float = 0.1,
                 min_child_samples: int = 20,
                 min_child_weight: float = 1e-3,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 0.0,
                 n_estimators: int = 100,
                 objective: 'str' = 'multiclass',
                 verbose: int = -1) -> None:
        super().__init__(random_state=random_state,
                         num_leaves=num_leaves,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         min_child_samples=min_child_samples,
                         min_child_weight=min_child_weight,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         reg_alpha=reg_alpha,
                         reg_lambda=reg_lambda,
                         n_estimators=n_estimators,
                         objective=objective,
                         verbose=verbose)
        self.name = name

    @classmethod
    def get_parameter_space(cls):
        return cls._parameter_space
