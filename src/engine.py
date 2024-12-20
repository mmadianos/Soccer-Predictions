import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from typing import Union, List, Tuple
from sklearn.base import ClassifierMixin
from .feature_engineering.get_features import FeaturesEngine
from .build_pipeline import build_pipeline


class Engine:
    def __init__(self, config: dict) -> None:
        self.config = config
        self._preprocessor = None
        self._feature_engine = FeaturesEngine()

    def _get_data(self):
        """

        """
        absolute_data_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_data_dir = os.path.join(
            absolute_data_dir, f'../vault/{self.config["TRAINING_FILE"]}')

        keep_col = ['HomeTeam', 'AwayTeam', 'B365H',
                    'B365D', 'B365A', 'FTR', 'FTHG', 'FTAG']
        df = pd.read_csv(absolute_data_dir, usecols=keep_col)

        df = self._feature_engine.generate_features(df)
        df = df.drop(columns=['HomeTeam', 'AwayTeam'])
        X, y = df.drop(columns=['FTR', 'FTHG', 'FTAG']), df.FTR
        return X, y

    @staticmethod
    def _evaluate(data, truth, model):
        prediction = model.predict(data)
        # pred_proba = model.predict_proba(data)
        '''
        metrics_df = pd.Series({
            'accuracy': round(metrics.accuracy_score(truth, prediction), 2),
            'precision': round(metrics.precision_score(truth, prediction, average='weighted'), 2),
            'recall': round(metrics.recall_score(truth, prediction, average='weighted'), 2),
            'f1 score': round(metrics.f1_score(truth, prediction, average='weighted'), 2),
            'roc auc': round(metrics.roc_auc_score(truth, pred_proba, multi_class='ovr'), 2)
        })
        '''

        report = metrics.classification_report(
            truth, prediction,
            target_names=model.classes_,
            output_dict=True)
        confusion_matrix = metrics.confusion_matrix(
            truth, prediction, labels=model.classes_)
        return report, confusion_matrix

    def train(self,
              model: Union[ClassifierMixin, List[Tuple[str, ClassifierMixin]]],
              cv=False):
        """

        """
        X, y = self._get_data()

        pipeline = build_pipeline(self.config, model)

        if cv:
            print('Performing cross-validation')
            strategy = self.config.get('CV_STRATEGY', 'StratifiedKFold')
            assert strategy in ['StratifiedKFold',
                                'KFold'], f'Invalid CV strategy: {strategy}'
            metrics = self._cross_validation(pipeline, X, y,
                                             scoring=self.config['SCORING'],
                                             cv_strategy=strategy,
                                             cv_splits=self.config['CV_SPLITS'])

        else:
            print('Performing train-test split')
            test_size = self.config['TEST_SIZE']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size)
            pipeline.fit(X_train, y_train)
            metrics, confusion = self._evaluate(X_test, y_test, model)
            if self.config['PLOT_CONFUSION']:
                self._plot_confusion(confusion, labels=model.classes_)

        # ff = pd.DataFrame(self._model.feature_importances_, index=X.columns, columns=["Importance"])
        return metrics

    def _cross_validation(self, model, X, y, scoring, cv_strategy, cv_splits: int = 5):
        if cv_strategy == 'StratifiedKFold':
            cv = StratifiedKFold(n_splits=cv_splits,
                                 shuffle=True, random_state=42)
        elif cv_strategy == 'KFold':
            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        else:
            raise ValueError('Invalid CV_STRATEGY specified')

        return cross_validate(model, X, y, cv=cv, scoring=scoring)

    @staticmethod
    def _plot_confusion(confusion_matrix, labels):
        disp = metrics.ConfusionMatrixDisplay(
            confusion_matrix, display_labels=labels)
        disp.plot()
        plt.title('Confusion Matrix')
        plt.show()
