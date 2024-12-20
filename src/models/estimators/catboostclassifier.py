from catboost import CatBoostClassifier


class CatboostClassifier(CatBoostClassifier):
    _parameter_space = {
        'iterations': [100, 1000],  # Number of boosting iterations
        'learning_rate': [1e-4, 0.1],  # Learning rate (typically small)
        'depth': [3, 10],  # Depth of the trees
        'l2_leaf_reg': [1e-5, 10.0],  # L2 regularization
        # Fraction of features used for each tree split
        'colsample_bylevel': [0.3, 1.0],
        # Maximum number of bins to use for numerical features
        'max_bin': [32, 255],
        'grow_policy': ['SymmetricTree', 'Depthwise']  # Tree growth policy
    }

    def __init__(self,
                 name='CatBoostClassifier',
                 iterations=100,
                 learning_rate=0.1,
                 depth=6,
                 l2_leaf_reg=3.0,
                 colsample_bylevel=1.0,
                 max_bin=255,
                 grow_policy='SymmetricTree',
                 random_state=None,
                 verbose=False) -> None:
        super().__init__(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            colsample_bylevel=colsample_bylevel,
            max_bin=max_bin,
            grow_policy=grow_policy,
            random_state=random_state,
            verbose=verbose
        )
        self.name = name

    @classmethod
    def get_parameter_space(cls):
        return cls._parameter_space
