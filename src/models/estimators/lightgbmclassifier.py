from lightgbm import LGBMClassifier


class LightGBMClassifier(LGBMClassifier):
    _parameter_space = {
        'num_leaves': [3, 40],
        'min_child_samples': [20, 200],
        'min_child_weight': [1e-5, 1e4],
        'subsample': [0.3, 1.],
        'colsample_bytree': [0.3, 1.],
        'reg_alpha': [1e-2, 10.],
        'reg_lambda': [1e-2, 10.]
    }

    def __init__(self,
                 name='LightGBMClassifier',
                 boosting_type='gbdt',
                 random_state=None,
                 num_leaves=31,
                 min_child_samples=20,
                 min_child_weight=0.001,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 reg_alpha=0.,
                 reg_lambda=0.) -> None:
        super().__init__(random_state=random_state,
                         boosting_type=boosting_type,
                         num_leaves=num_leaves,
                         min_child_samples=min_child_samples,
                         min_child_weight=min_child_weight,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         reg_alpha=reg_alpha,
                         reg_lambda=reg_lambda)
        self.name = name

    @classmethod
    def get_parameter_space(cls):
        return cls._parameter_space
