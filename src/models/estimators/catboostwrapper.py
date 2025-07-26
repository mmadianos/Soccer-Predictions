from catboost import CatBoostClassifier


class CatBoostWrapper(CatBoostClassifier):
    _parameter_space = {
        'learning_rate': {
            'type': 'float',
            'low': 0.001,
            'high': 0.3,
            'log': True
        },
        'depth': {
            'type': 'int',
            'low': 3,
            'high': 10
        },
        'l2_leaf_reg': {
            'type': 'float',
            'low': 1.0,
            'high': 10.0,
            'log': True
        },
        'min_child_samples': {
            'type': 'int',
            'low': 1,
            'high': 20
        },
        'border_count': {
            'type': 'int',
            'low': 32,
            'high': 255
        },
        'bootstrap_type': {
            'type': 'categorical',
            'choices': ['Bayesian', 'Bernoulli', 'MVS']
        }
    }

    def __init__(self, **kwargs):
        self.name = 'CatBoostWrapper'
        kwargs.setdefault('verbose', 0)
        kwargs.setdefault('random_seed', 42)
        kwargs.setdefault('iterations', 400)
        super().__init__(**kwargs)

    @classmethod
    def get_parameter_space(cls):
        return cls._parameter_space
