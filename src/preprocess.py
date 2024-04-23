from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Union

#feature selection
#calibration
#sampling
class PreprocessorPipeline(TransformerMixin, BaseEstimator):
    def __init__(self, scaler_type: Union[str, None] = None, 
                encoder_type: Union[str, None] = None,
                imputer_type: Union[str, None] = None) -> None:
        self.scaler_type = scaler_type if isinstance(scaler_type, str) else None
        self.encoder_type = encoder_type if isinstance(encoder_type, str) else None
        self.imputer_type = imputer_type if isinstance(imputer_type, str) else None
        self._processor = None
        print('Initializing preprocessor with (scaler, encoder, imputer = '
        f'{self.scaler_type, self.encoder_type, self.imputer_type}')
        
    def fit(self, X, y=None):
        numerical_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        numerical_transformer = Pipeline(steps=[
            ('imputer', self._get_imputer(self.imputer_type)),
            ('scaler', self._get_scaler(self.scaler_type))
        ])

        categorical_transformer = Pipeline(steps=[
            ('encoder', self._get_encoder(self.encoder_type)),
            ('scaler', self._get_scaler(self.scaler_type))
        ])

        self._processor = ColumnTransformer(
            transformers=[
                ('numerical_pipeline', numerical_transformer, numerical_cols),
                ('categorical_pipeline', categorical_transformer, categorical_cols)
            ])
        
        self._processor.fit(X)
        return self
    
    def transform(self, X):
        if self._processor:
            return self._processor.transform(X)
        else:
            return X

    @staticmethod
    def _get_scaler(scaler_type):
        if scaler_type is None:
            return None
        
        scalers = {
            'Standard': StandardScaler(),
            'Minmax': MinMaxScaler(),
            'Robust': RobustScaler(),
            'Maxabs': MaxAbsScaler()
        }
        if scaler_type not in scalers:
            raise ValueError(f"Scaler type must be one of {list(scalers.keys())}")
        return scalers[scaler_type]
    
    @staticmethod
    def _get_encoder(encoder_type):
        if encoder_type is None:
            return None
        
        encoders = {
            'Label': LabelEncoder(),
            'Onehot': OneHotEncoder()
        }
        if encoder_type not in encoders:
            raise ValueError(f"Encoder type must be one of {list(encoders.keys())}")
        return encoders[encoder_type]
    
    @staticmethod
    def _get_imputer(imputer_type):
        if imputer_type is None:
            return None
        
        imputers = {
            'Simple': SimpleImputer(),
            'Knn': KNNImputer()
        }
        if imputer_type not in imputers:
            raise ValueError(f"Imputer type must be one of {list(imputers.keys())}")
        return imputers[imputer_type]