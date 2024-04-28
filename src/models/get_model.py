import importlib
import joblib
from typing import List, Tuple, Union, Sequence
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV


def create_ensemble(estimators: List[Tuple[str, BaseEstimator]],
                    voting: str = 'soft',
                    weights: Union[None, List[float]] = None,
                    verbose: bool = True,
                    **kwargs) -> VotingClassifier:
    """Create a voting ensemble classifier."""
    ensemble = VotingClassifier(estimators=estimators, voting=voting,
                                weights=weights, verbose=verbose, **kwargs)
    return ensemble

def load_model(model_name: str, params_path: str, calibrate_probabilities=False) -> BaseEstimator:
    """Load a trained model from file."""
    try:
        model_params = joblib.load(params_path)
    except FileNotFoundError:
        print(f"Model parameters file '{params_path}' not found. Using default parameters.")
        model_params = {}
    model_package = f'models.estimators.{model_name.lower()}'
    mod = importlib.import_module(model_package)
    model = getattr(mod, model_name)(**model_params)
    if calibrate_probabilities:
       model = CalibratedClassifierCV(model)
    return model

def build_model(model_names: Union[str, Sequence[str]], 
              params_paths: str) -> Union[BaseEstimator, VotingClassifier]:
    """Get a single model or create an ensemble of models."""
    if isinstance(model_names, str):
        params_paths += model_names.lower()+'.pkl'
        model = load_model(model_names, params_paths)
    else:
        print('Building ensemble model...')
        estimators = [(model_name, load_model(model_name, param_path+model_name.lower()+'.pkl')) 
                      for model_name, param_path in zip(model_names, params_paths)]
        print(estimators)
        model = create_ensemble(estimators)
    print('type', type(model))
    return model
