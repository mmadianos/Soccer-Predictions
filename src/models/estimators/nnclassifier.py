from sklearn.neural_network import MLPClassifier

    
class NNClassifier(MLPClassifier):
    _parameter_space = {
        'hidden_layer_sizes': [(100,), (50, 50), (25, 50, 25)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [1e-4, 1e-1],
        'learning_rate': ['constant', 'adaptive'],
    }

    def __init__(self,
                 name='NNClassifier',
                 hidden_layer_sizes=(100,),
                 activation='relu',
                 solver='adam',
                 alpha=0.0001,
                 learning_rate='constant',
                 max_iter=200) -> None:
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation,
                         solver=solver,
                         alpha=alpha,
                         learning_rate=learning_rate)
        self.name = name

    @classmethod
    def get_parameter_space(cls):
        return cls._parameter_space
