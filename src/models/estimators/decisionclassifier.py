from ..base import BaseModel
from sklearn.tree import DecisionTreeClassifier


class DecisionClassifier(DecisionTreeClassifier):
    _parameter_space = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [1, 20],
            'min_samples_leaf': [1, 20],
            'min_samples_split': [2, 20],
            'max_features': [0.1, 1.0]
        }
    def __init__(self, name='DecisionClassifier',
                *,
                criterion="gini",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=None,
                random_state=None,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                class_weight='balanced',
                ccp_alpha=0.0) -> None:
        super().__init__(criterion=criterion,
                        splitter=splitter,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                        max_features=max_features,
                        random_state=random_state,
                        max_leaf_nodes=max_leaf_nodes,
                        min_impurity_decrease=min_impurity_decrease,
                        class_weight=class_weight,
                        ccp_alpha=ccp_alpha)
        self.name = name
    
    @classmethod
    def get_parameter_space(cls):
        return cls._parameter_space