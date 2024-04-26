import importlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from preprocess import PreprocessorPipeline
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from typing import Union, List, Tuple
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier

class Engine:
    def __init__(self, model:Union[BaseEstimator, List[Tuple[str, BaseEstimator]]], config) -> None:
        self.config = config
        self.test_size = 0.2
        self._model = model
        self._preprocessor = None

    def _get_data(self):
        keep_col=['HomeTeam','AwayTeam', 'B365H', 'IWH', 'FTR']
        df = pd.read_csv(self.config['TRAINING_FILE'], usecols=keep_col)
        df = df.drop(columns=['HomeTeam','AwayTeam'])
        X, y = df.drop(columns='FTR'), df.FTR
        return X, y
    
    @property
    def model(self):
        return self._model

    @staticmethod
    def _evaluate(data, truth, model):
        prediction = model.predict(data)
        #pred_proba = model.predict_proba(data)
        '''
        metrics_df = pd.Series({
            'accuracy': round(metrics.accuracy_score(truth, prediction), 2),
            'precision': round(metrics.precision_score(truth, prediction, average='weighted'), 2),
            'recall': round(metrics.recall_score(truth, prediction, average='weighted'), 2),
            'f1 score': round(metrics.f1_score(truth, prediction, average='weighted'), 2),
            'roc auc': round(metrics.roc_auc_score(truth, pred_proba, multi_class='ovr'), 2)
        })
        '''

        report = metrics.classification_report(truth, prediction, target_names=model.classes_)
        confusion_matrix = metrics.confusion_matrix(truth, prediction, labels=model.classes_)
        return report, confusion_matrix
    
    def train(self, cv=False):
        X, y = self._get_data()
        scaler_type = self.config.get('SCALER_TYPE', None)
        encoder_type = self.config.get('ENCODER_TYPE', None)
        imputer_type = self.config.get('IMPUTER_TYPE', None)

        self._preprocessor = PreprocessorPipeline(scaler_type, encoder_type, imputer_type)

        if isinstance(self._model, VotingClassifier):
            pipeline = Pipeline([('Preprocessor', self._preprocessor), ('Ensemble', self._model)])
        else:
            pipeline = Pipeline([('Preprocessor', self._preprocessor), (self._model.name, self._model)])
        
        if cv:
            strategy = self.config.get('CV_STRATEGY', 'StratifiedKFold')
            assert strategy in ['StratifiedKFold', 'KFold'], f'Invalid CV strategy: {strategy}'
            metrics = self._cross_validation(pipeline, X, y, strategy)

        else: 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
            pipeline.fit(X_train, y_train)
            metrics, confusion = self._evaluate(X_test, y_test, self._model)
            if self.config['PLOT_CONFUSION']: self._plot_confusion(confusion, labels=self._model.classes_)

        return metrics

    def _cross_validation(self, model, X, y, strategy, cv_splits=5):
        scoring = ['precision_macro', 'recall_macro', 'f1_weighted']

        if strategy == 'StratifiedKFold':
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        elif strategy == 'KFold':
            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        
        return cross_validate(model, X, y, cv=cv, scoring=scoring)

    @staticmethod
    def _plot_confusion(confusion_matrix, labels):
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)
        disp.plot()
        plt.title('Confusion Matrix')
        plt.show()
    