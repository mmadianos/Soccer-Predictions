import importlib
import pandas as pd
from sklearn import metrics
from preprocess import Preprocessor

class Engine:
    def __init__(self, config, model=None) -> None:
        self.config = config
        self.test_size = config['TEST_SIZE']
        self._model = self._get_model(config['MODEL'])()
        self.preprocessor = Preprocessor()

    def _get_data(self):
        keep_col=['HomeTeam','AwayTeam', 'B365H', 'IWH', 'FTR']
        df = pd.read_csv(self.config['TRAINING_FILE'], usecols=keep_col)
        df = df.drop(columns=['HomeTeam','AwayTeam'])
        return self.preprocessor.fit_transform(X = df.drop(columns='FTR'), y = df.FTR)
        #return train_test_split(df.drop(columns='FTR'), df.FTR, test_size=self.test_size, random_state=42)

    @property
    def model(self):
        return self._model
    
    @staticmethod
    def _get_model(model_name):
        model_package = f'models.scikit.{model_name.lower()}'
        mod = importlib.import_module(model_package)
        return getattr(mod, model_name)

    @staticmethod
    def _evaluate(truth, prediction):
        accuracy = metrics.accuracy_score(truth, prediction)
        confusion_matrix = metrics.confusion_matrix(truth, prediction)
        return accuracy, confusion_matrix

    def train(self, random_state=42):
        X_train, X_valid, X_test, y_train, y_valid, y_test, scaler = self._get_data()
        self._model.fit(X_train, y_train)
        y_pred = self._model.predict(X_test)
        return self._evaluate(y_test, y_pred)