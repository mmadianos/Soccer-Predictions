from sklearn.neural_network import MLPClassifier


class NNClassifier(MLPClassifier):
    _parameter_space = {
        'hidden_layer_sizes': {
            'type': 'categorical',
            'choices': ['100', '50-50', '25-50-25']
        },
        'activation': {
            'type': 'categorical',
            'choices': ['relu', 'tanh', 'logistic']
        },
        'solver': {
            'type': 'categorical',
            'choices': ['adam', 'sgd', 'lbfgs']
        },
        'alpha': {
            'type': 'float',
            'low': 1e-5,
            'high': 1e-1,
            'log': True
        },
        'learning_rate': {
            'type': 'categorical',
            'choices': ['constant', 'adaptive', 'invscaling']
        },
        'learning_rate_init': {
            'type': 'float',
            'low': 1e-4,
            'high': 1e-1,
            'log': True
        }
    }

    def __init__(self,
                 name='NNClassifier',
                 hidden_layer_sizes=(100,),
                 activation='relu',
                 solver='adam',
                 alpha=0.0001,
                 learning_rate='constant',
                 learning_rate_init=0.001,
                 max_iter=400) -> None:
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation,
                         solver=solver,
                         alpha=alpha,
                         learning_rate=learning_rate,
                         learning_rate_init=learning_rate_init,
                         max_iter=max_iter)
        self.name = name

    @classmethod
    def get_parameter_space(cls):
        return cls._parameter_space
