from imblearn.pipeline import Pipeline
from .preprocess import PreprocessorPipeline
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
#from imblearn.under_sampling import ClusterCentroids


def get_resampler(resampler_type):
    """

    """
    if resampler_type is None:
        # or not self.config.get('RESAMPLER', False):
        print("No resampling applied. Using original data distribution.")
        return None

    resamplers = {
        'SMOTE': SMOTE(),
        'SMOTEENN': SMOTEENN(),
    }
    if resampler_type not in resamplers:
        raise ValueError(f"Invalid resampler type: {resampler_type}. "
                         f"Available options: {list(resamplers.keys())}")
    print(f"Using {resampler_type} for resampling.")
    return resamplers[resampler_type]


def build_pipeline(config, model):
    """
Builds a machine learning pipeline based on the configuration and model.

Args:
    config (dict): Configuration dictionary.
    model (Union[ClassifierMixin, VotingClassifier]):
                                    The classifier or ensemble to use.

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
    resampler = get_resampler(resampler_type)
    steps.append(('Resampler', resampler))

    # Add the model
    if isinstance(model, VotingClassifier):
        steps.append(('Ensemble', model))
    else:
        steps.append(('Classifier', model))
    return Pipeline(steps)
