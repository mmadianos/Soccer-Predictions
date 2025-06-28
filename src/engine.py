import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from typing import Union, List, Tuple
from sklearn.base import ClassifierMixin
from .build_pipeline import build_pipeline


class Engine:
    def __init__(self, config: dict) -> None:
        self.config = config
        self._preprocessor = None
        self.X_test = None
        self.y_test = None

    @staticmethod
    def _evaluate(data, truth, model):
        prediction = model.predict(data)
        # pred_proba = model.predict_proba(data)
        report = metrics.classification_report(
            truth, prediction, target_names=model.classes_)
        confusion_matrix = metrics.confusion_matrix(
            truth, prediction, labels=model.classes_)
        return report, confusion_matrix

    def train(self, X, y,
              model: Union[ClassifierMixin, List[Tuple[str, ClassifierMixin]]]):
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

    def cross_validate(self, X, y,
                       model: Union[ClassifierMixin, List[Tuple[str, ClassifierMixin]]]):
        """
        Perform cross-validation and return metrics.
        """
        random_state = 42
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
            pipeline, X, y, cv=cv, scoring=self.config['SCORING']
        )
        return cv_results

    @staticmethod
    def _plot_confusion(confusion_matrix, labels):
        disp = metrics.ConfusionMatrixDisplay(
            confusion_matrix, display_labels=labels)
        disp.plot()
        plt.title('Confusion Matrix')
        plt.show()
