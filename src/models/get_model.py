import importlib
import joblib
import os
from typing import List, Tuple, Union
from sklearn.base import ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
import logging
logger = logging.getLogger(__name__)


def create_ensemble(estimators: List[Tuple[str, ClassifierMixin]],
                    voting: str = 'soft',
                    weights: Union[None, List[float]] = None,
                    verbose: bool = True) -> VotingClassifier:
    """Create a voting ensemble classifier."""
    ensemble = VotingClassifier(estimators=estimators, voting=voting,
                                weights=weights, verbose=verbose)
    return ensemble


def load_model(model_name: str, params_path: str,
               calibrate_probabilities: bool = False) -> ClassifierMixin:
    """Load a trained model from file."""
    try:
        model_params = joblib.load(params_path)
    except FileNotFoundError:
        logger.warning(
            "Model parameters file '%s' not found. Using default parameters.",
            params_path
        )
        model_params = {}
    model_package = f'src.models.estimators.{model_name.lower()}'
    mod = importlib.import_module(model_package)
    model = getattr(mod, model_name)(**model_params)
    if calibrate_probabilities:
        logger.info(
            "Calibrating model probabilities using CalibratedClassifierCV."
        )
        model = CalibratedClassifierCV(model)
    return model


def build_model(config: dict) -> Union[ClassifierMixin, VotingClassifier]:
    """Get a single model or create an ensemble of models."""
    model_names = config['MODEL']
    parameters_vault = os.path.dirname(os.path.abspath(__file__))
    parameters_vault = os.path.join(
        parameters_vault, '../../vault/tuned_params/')

    if isinstance(model_names, str):
        parameters_vault = os.path.join(
            parameters_vault, model_names.lower()+'.pkl')
        logger.info('%s: Loading model parameters...', config["MODEL"])
        model = load_model(model_names, parameters_vault)
    else:
        logger.info("Building ensemble model...")
        estimators = [
            (model_name, load_model(model_name, os.path.join(
                parameters_vault+model_name.lower()+'.pkl')))
            for model_name in model_names
        ]
        logger.info("Estimators loaded: %s", [name for name, _ in estimators])
        model = create_ensemble(estimators)
    return model
