from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.base import BaseEstimator, TransformerMixin
#feature selection
#calibration
#encoding
#sampling

class Preprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, scaler_type: str ='standard') -> None:
        self.scaler_type = scaler_type if isinstance(scaler_type, str) else None
        self.scaler = None
        
    def fit(self, X, y=None):
        if self.scaler_type:
            self.scaler = self._get_scaler(self.scaler_type)
            self.scaler.fit(X)
        return self
    
    def transform(self, X):
        if self.scaler:
            return self.scaler.transform(X)
        else:
            return X
    
    @staticmethod
    def _get_scaler(scaler_type):
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'maxabs': MaxAbsScaler()
        }
        if scaler_type.lower() not in scalers:
            raise ValueError(f"Scaler type must be one of {list(scalers.keys())}")
        return scalers[scaler_type]
    