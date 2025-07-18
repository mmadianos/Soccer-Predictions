from lightgbm import LGBMClassifier


class LightGBMClassifier(LGBMClassifier):
    _parameter_space = {
        'num_leaves': {
            'type': 'int',
            'low': 3,
            'high': 40
        },
        'min_child_samples': {
            'type': 'int',
            'low': 20,
            'high': 200
        },
        'min_child_weight': {
            'type': 'float',
            'low': 1e-5,
            'high': 1e4,
            'log': True
        },
        'subsample': {
            'type': 'float',
            'low': 0.3,
            'high': 1.0,
            'log': False
        },
        'colsample_bytree': {
            'type': 'float',
            'low': 0.3,
            'high': 1.0,
            'log': False
        },
        'reg_alpha': {
            'type': 'float',
            'low': 1e-2,
            'high': 10.0,
            'log': True
        },
        'reg_lambda': {
            'type': 'float',
            'low': 1e-2,
            'high': 10.0,
            'log': True
        }
    }

    def __init__(self, **kwargs):
        self.name = 'LightGBMClassifier'
        verbose = kwargs.pop('verbose', -1)
        super().__init__(**kwargs, verbose=verbose)

    @classmethod
    def get_parameter_space(cls):
        return cls._parameter_space
