from typing import Union
from sklearn.preprocessing import (
  StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
  OrdinalEncoder, OneHotEncoder, FunctionTransformer)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.pipeline import Pipeline
import logging
logger = logging.getLogger(__name__)


class PreprocessorPipeline(TransformerMixin, BaseEstimator):
    """

    """

    def __init__(self, scaler_type: Union[str, None] = None,
                 encoder_type: Union[str, None] = None,
                 imputer_type: Union[str, None] = None) -> None:

        self.scaler_type = scaler_type if isinstance(
            scaler_type, str) else None
        self.encoder_type = encoder_type if isinstance(
            encoder_type, str) else None
        self.imputer_type = imputer_type if isinstance(
            imputer_type, str) else None
        self._processor = None

        logger.info(
            "Preprocessor initialized with scaler=%s, encoder=%s, imputer=%s",
            self.scaler_type, self.encoder_type, self.imputer_type
        )

    def fit(self, X, y=None):
        """

        """
        numerical_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(
            include=['object', 'category']).columns

        # Build numerical pipeline
        num_steps = []
        imputer = self._get_imputer(self.imputer_type)
        if imputer is not None:
            num_steps.append(('imputer', imputer))
        scaler = self._get_scaler(self.scaler_type)
        if scaler is not None:
            num_steps.append(('scaler', scaler))
        if not num_steps:
            num_steps = [('identity', FunctionTransformer(lambda x: x))]

        numerical_transformer = Pipeline(steps=num_steps)

        # Build categorical pipeline
        cat_steps = []
        encoder = self._get_encoder(self.encoder_type)
        if encoder is not None:
            cat_steps.append(('encoder', encoder))
        # Do not scale categorical features after encoding
        if not cat_steps:
            cat_steps = [('identity', FunctionTransformer(lambda x: x))]
        categorical_transformer = Pipeline(steps=cat_steps)

        self._processor = ColumnTransformer(
            transformers=[
                ('numerical_pipeline', numerical_transformer, numerical_cols),
                ('categorical_pipeline', categorical_transformer, categorical_cols)
            ]
        )

        self._processor.fit(X, y)
        return self

    def transform(self, X):
        if self._processor:
            return self._processor.transform(X)
        else:
            return X

    @staticmethod
    def _get_scaler(scaler_type):
        """

        """
        if scaler_type is None:
            return None

        scalers = {
            'Standard': StandardScaler(),
            'Minmax': MinMaxScaler(),
            'Robust': RobustScaler(),
            'Maxabs': MaxAbsScaler()
        }
        if scaler_type not in scalers:
            raise ValueError(
                f"Scaler type must be one of {list(scalers.keys())}")
        return scalers[scaler_type]

    @staticmethod
    def _get_encoder(encoder_type):
        """
        Return an encoder for categorical features.
        """
        if encoder_type is None:
            return None

        encoders = {
            'Ordinal': OrdinalEncoder(),
            'Onehot': OneHotEncoder()
        }
        if encoder_type not in encoders:
            raise ValueError(
                f"Encoder type must be one of {list(encoders.keys())}")
        return encoders[encoder_type]

    @staticmethod
    def _get_imputer(imputer_type):
        """

        """
        if imputer_type is None:
            return None

        imputers = {
            'Simple': SimpleImputer(),
            'Knn': KNNImputer()
        }
        if imputer_type not in imputers:
            raise ValueError(
                f"Imputer type must be one of {list(imputers.keys())}")
        return imputers[imputer_type]
