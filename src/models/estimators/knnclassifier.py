
from sklearn.neighbors import KNeighborsClassifier

    
class KNNClassifier(KNeighborsClassifier):
    _parameter_space = {
        'n_neighbors': [1, 10],
        'weights': ['uniform', 'distance']
        }
    def __init__(self,name='KNNClassifier', calibrate_probabilities=False,
                n_neighbors=5,
                *,
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
        self.name=name
        self.calibrate_probabilities=calibrate_probabilities
        #self._model = self._get_model()

    def _get_estimator(self, **kwargs):
        return KNeighborsClassifier(**kwargs)
    
    @classmethod
    def get_parameter_space(cls):
        return cls._parameter_space
    