import importlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics


class Engine:
    def __init__(self, config, model=None) -> None:
        self.config = config
        self.test_size = config['TEST_SIZE']
        self.clf = self._get_model(config['MODEL'])()

    def _get_data(self):
        keep_col=['HomeTeam','AwayTeam', 'B365H', 'IWH', 'FTR']
        df = pd.read_csv(self.config['TRAINING_FILE'], usecols=keep_col)
        df = df.drop(columns=['HomeTeam','AwayTeam'])
        return train_test_split(df.drop(columns='FTR'), df.FTR, test_size=self.test_size, random_state=42)

    @staticmethod
    def _get_model(model_name):
        model_package = f'models.scikit.{model_name.lower()}'
        mod = importlib.import_module(model_package)
        return getattr(mod, model_name)

    def train(self, random_state=42):
        X_train, X_test, y_train, y_test = self._get_data()
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        m = metrics.accuracy_score(y_test, y_pred)
        mm = metrics.confusion_matrix(y_test, y_pred)
        return m, mm