from sklearn.svm import SVC


class SVClassifier(SVC):
    _parameter_space = {
        'C': {
            'type': 'float',
            'low': 1e-1,
            'high': 1e3,
            'log': True
        },
        'gamma': {
            'type': 'float',
            'low': 1e-4,
            'high': 1.0,
            'log': True
        },
        'kernel': {
            'type': 'categorical',
            'choices': ['linear', 'poly', 'rbf', 'sigmoid']
        }
    }

    def __init__(self,
                 name='SVClassifier',
                 *,
                 C=1.,
                 gamma='scale',
                 kernel='rbf') -> None:
        super().__init__(C=C,
                         gamma=gamma,
                         kernel=kernel,
                         probability=True)
        self.name = name

    @classmethod
    def get_parameter_space(cls):
        return cls._parameter_space
