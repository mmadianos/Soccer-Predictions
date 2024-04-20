import importlib
import pandas as pd
from sklearn import metrics
from preprocess import Preprocessor
import matplotlib.pyplot as plt


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

    @property
    def model(self):
        return self._model
    
    @staticmethod
    def _get_model(model_name):
        model_package = f'models.scikit.{model_name.lower()}'
        mod = importlib.import_module(model_package)
        return getattr(mod, model_name)

    @staticmethod
    def _evaluate(data, truth, model):
        prediction = model.predict(data)
        pred_proba = model.predict_proba(data)
        metrics_dict = pd.Series({
            'accuracy': round(metrics.accuracy_score(truth, prediction), 2),
            'precision': round(metrics.precision_score(truth, prediction, average='weighted'), 2),
            'recall': round(metrics.recall_score(truth, prediction, average='weighted'), 2),
            'f1 score': round(metrics.f1_score(truth, prediction, average='weighted'), 2),
            'roc auc': round(metrics.roc_auc_score(truth, pred_proba, multi_class='ovr'), 2)
        })
        
        confusion_matrix = metrics.confusion_matrix(truth, prediction, labels=model.classes_)
        return metrics_dict, confusion_matrix

    def train(self):
        try:
            X_train, X_valid, X_test, y_train, y_valid, y_test, scaler = self._get_data()
            self._model.fit(X_train, y_train)
        except Exception as e:
            print(f"An error occurred during training: {e}")
        
        metrics, confusion = self._evaluate(X_test, y_test, self._model)
        if self.config['PLOT_EVAL']:
            self._plot_eval(confusion, labels=self._model.classes_)
        return metrics
    
    @staticmethod
    def _plot_eval(confusion_matrix, labels):
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)
        disp.plot()
        plt.title('Confusion Matrix')
        plt.show()
    