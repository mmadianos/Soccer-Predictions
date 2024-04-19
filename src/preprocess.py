from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, valid_ratio=None, test_ratio=0.2, scaler='standard') -> None:
        self.scaler = self._get_scaler(scaler) if scaler else None
        self.valid_ratio, self.test_ratio = valid_ratio, test_ratio

    @staticmethod
    def _get_scaler(type):
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'maxabs': MaxAbsScaler()
        }
        if type not in scalers:
            raise ValueError(f"Scaler type must be one of {list(scalers.keys())}")
        return scalers[type]
    
    def _split_data(self, X, y):
        seed = 42
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_ratio, random_state=seed)
        
        X_valid, y_valid = None, None
        if self.valid_ratio:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=self.valid_ratio / (self.test_ratio + self.valid_ratio), random_state=seed)
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test


    def fit_transform(self, X, y):
        X_train, X_valid, X_test, y_train, y_valid, y_test = self._split_data(X, y)
        
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_valid = self.scaler.transform(X_valid) if X_valid is not None else None
            X_test = self.scaler.transform(X_test)
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test, self.scaler

    def transform(self, X):
        if self.scaler:
            return self.scaler.transform(X)
        else:
            return X