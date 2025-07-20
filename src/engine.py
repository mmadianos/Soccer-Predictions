from typing import Union, List, Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_validate, train_test_split)
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    accuracy_score, recall_score, f1_score, roc_auc_score, make_scorer)
from .build_pipeline import build_pipeline


class Engine:
    def __init__(self, config: dict) -> None:
        self.config = config
        self._preprocessor = None
        self.X_test = None
        self.y_test = None
        self.scorers = self._get_scorers()

    @staticmethod
    def _evaluate(data, truth, model):
        prediction = model.predict(data)
        # pred_proba = model.predict_proba(data)
        report = classification_report(
            truth, prediction, target_names=model.classes_,
            output_dict=True, zero_division=0)
        matrix = confusion_matrix(
            truth, prediction, labels=model.classes_)
        return report, matrix

    def train(self,
              X, y,
              model: Union[ClassifierMixin,
                           List[Tuple[str, ClassifierMixin]]]):
        """
        Fit the model using a train/test split.
        Returns the fitted pipeline.
        """
        pipeline = build_pipeline(self.config, model)
        test_size = self.config['TEST_SIZE']
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=test_size)
        pipeline.fit(X_train, y_train)
        return pipeline

    def test(self, pipeline):
        """
        Evaluate the fitted pipeline on the stored test data.
        Returns a dictionary of metrics.
        """
        report, confusion = self._evaluate(self.X_test, self.y_test, pipeline)
        if self.config['PLOT_CONFUSION']:
            self._plot_confusion(confusion, labels=pipeline.classes_)
        return {
            "classification_report": report,
            "confusion_matrix": confusion
        }

    def cross_validate(self,
                       X, y,
                       model: Union[ClassifierMixin,
                                    List[Tuple[str, ClassifierMixin]]],
                       random_state: Union[int, None] = None):
        """
        Perform cross-validation and return metrics.
        """
        pipeline = build_pipeline(self.config, model)
        strategy = self.config.get('CV_STRATEGY', 'StratifiedKFold')
        if strategy == 'StratifiedKFold':
            cv = StratifiedKFold(n_splits=self.config['CV_SPLITS'],
                                 shuffle=True,
                                 random_state=random_state)
        elif strategy == 'KFold':
            cv = KFold(
                n_splits=self.config['CV_SPLITS'],
                shuffle=True,
                random_state=random_state)
        else:
            raise ValueError('Invalid CV_STRATEGY specified')

        cv_results = cross_validate(
            pipeline, X, y, cv=cv, scoring=self.scorers,
        )
        return cv_results

    @staticmethod
    def _plot_confusion(conf_matrix, labels):
        disp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
        disp.plot()
        plt.title('Confusion Matrix')
        plt.show()

    def _get_scorers(self):
        """
        Get the scoring metrics for cross-validation.
        """
        scorers = {
            'accuracy': make_scorer(accuracy_score),
            'recall_macro': make_scorer(recall_score, average='macro'),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'roc_auc_ovr': make_scorer(
                roc_auc_score,
                multi_class='ovr',
                response_method='predict_proba'
            )
        }

        scoring = self.config.get('SCORING', ['f1_weighted'])
        if isinstance(scoring, str):
            scoring = [scoring]

        return {metric: scorers[metric]
                for metric in scoring if metric in scorers}
