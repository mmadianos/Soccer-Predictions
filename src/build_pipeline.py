from imblearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks, EditedNearestNeighbours, ClusterCentroids)
from .preprocess import PreprocessorPipeline
import logging
logger = logging.getLogger(__name__)


def _get_resampler(resampler_type):
    """
    Returns the appropriate resampler object based on the resampler_type.
    """
    if resampler_type is None:
        logger.info("No resampling applied. Using original data distribution.")
        return None

    resamplers = {
        'SMOTE': SMOTE(),
        'ADASYN': ADASYN(),
        'RandomOverSampler': RandomOverSampler(),
        'RandomUnderSampler': RandomUnderSampler(),
        'TomekLinks': TomekLinks(),
        'EditedNearestNeighbours': EditedNearestNeighbours(),
        'ClusterCentroids': ClusterCentroids(),
        'SMOTEENN': SMOTEENN(),
        'SMOTETomek': SMOTETomek(),
    }
    if resampler_type not in resamplers:
        raise ValueError(f"Invalid resampler type: {resampler_type}. "
                         f"Available options: {list(resamplers.keys())}")
    logger.info("Using %s for resampling.", resampler_type)
    return resamplers[resampler_type]


def build_pipeline(config, model):
    """
    Builds a machine learning pipeline based on the configuration and model.

    Args:
        config (dict): Configuration dictionary.
        model (Union[ClassifierMixin, VotingClassifier]): The classifier or ensemble to use.

    Returns:
        Pipeline: Configured pipeline.
    """
    steps = []
    scaler_type = config.get('SCALER_TYPE', None)
    encoder_type = config.get('ENCODER_TYPE', None)
    imputer_type = config.get('IMPUTER_TYPE', None)
    resampler_type = config.get('RESAMPLER_TYPE', None)

    preprocessor = PreprocessorPipeline(
        scaler_type, encoder_type, imputer_type)
    steps.append(('Preprocessor', preprocessor))

    resampler = _get_resampler(resampler_type)

    if resampler is not None:
        steps.append(('Resampler', resampler))

    # Add the model
    if isinstance(model, VotingClassifier):
        steps.append(('Ensemble', model))
    else:
        steps.append(('Classifier', model))

    # Remove any steps with None for compatibility with imblearn/sklearn
    steps = [step for step in steps if step[1] is not None]
    return Pipeline(steps)
