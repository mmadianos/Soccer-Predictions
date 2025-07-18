from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier(KNeighborsClassifier):
    _parameter_space = {
        'n_neighbors': {
            'type': 'int',
            'low': 1,
            'high': 30
        },
        'weights': {
            'type': 'categorical',
            'choices': ['uniform', 'distance']
        },
        'metric': {
            'type': 'categorical',
            'choices': ['minkowski', 'euclidean', 'manhattan']
        }
    }

    def __init__(self,
                 name='KNNClassifier',
                 *,
                 n_neighbors=5,
                 weights="uniform",
                 algorithm="auto",
                 leaf_size=30,
                 p=2,
                 metric="minkowski") -> None:
        super().__init__(n_neighbors=n_neighbors,
                         weights=weights,
                         algorithm=algorithm,
                         leaf_size=leaf_size,
                         p=p,
                         metric=metric)
        self.name = name

    @classmethod
    def get_parameter_space(cls):
        return cls._parameter_space
