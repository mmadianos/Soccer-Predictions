import optuna
import joblib
import numpy as np
from .engine import Engine
from optuna.samplers import CmaEsSampler, RandomSampler, GridSampler, TPESampler
from optuna.samplers._base import BaseSampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_contour
from optuna.pruners import MedianPruner
from sklearn.ensemble import VotingClassifier
from typing import Union
from sklearn.base import ClassifierMixin
from sklearn.ensemble import VotingClassifier


class Tuner:
    def __init__(self,
                 engine: Engine,
                 model: Union[ClassifierMixin, VotingClassifier],
                 n_trials: int = 100,
                 metric: str = 'test_roc_auc_ovr',
                 sampler_type: str = 'TPESampler') -> None:

        self.n_trials = n_trials
        self.metric = metric
        self.sampler_type = self._get_sampler(sampler_type)
        self._engine = engine
        self._model = model

    def objective(self, trial: optuna.Trial):
        """
        Objective function for optimization.

        Args:
            trial (optuna.Trial): The trial object.

        Returns:
            float: The objective value.
        """
        if isinstance(self._model, VotingClassifier):
            tune_params = self._suggest_params_ensemble(trial)
            self._model.set_params(**tune_params)
        else:
            tune_params = self._suggest_params_single(trial, self._model)
            self._model.set_params(**tune_params)
        metrics = self._engine.train(model=self._model, cv=True)
        return np.mean(metrics[self.metric])

    def tune(self, save: bool, plot_tuning_results: bool) -> optuna.study.Study:
        """
        Perform hyperparameter tuning.

        Returns:
            optuna.study.Study: The optuna study object.
        """
        study = optuna.create_study(
            study_name='optimization', direction='maximize', pruner=MedianPruner())
        study.optimize(self.objective, n_trials=self.n_trials,
                       show_progress_bar=True)

        if save:
            joblib.dump(
                study.best_params, f'vault/tuned_params/{self._model.name.replace(" ", "").lower()}.pkl')
        if plot_tuning_results:
            self.plot_results(study=study)
        return study

    def _suggest_params_single(self, trial: optuna.Trial, model) -> dict:
        """
        Suggest tuning hyperparameters for a single model.

        Args:
            trial (optuna.Trial): The trial object.

        Returns:
            dict: Dictionary containing suggested hyperparameters.
        """
        params = {}
        hyperparams = model.get_parameter_space()

        for parameter, values in hyperparams.items():
            if isinstance(values, (list, tuple)):

                assert all(isinstance(e, type(values[0])) for e in values), \
                    f'All elements must be of same object.'

                if isinstance(values[0], int):
                    params[parameter] = trial.suggest_int(
                        parameter, values[0], values[-1])
                elif isinstance(values[0], float):
                    log = False
                    if values[-1]/values[0] > 100:
                        log = True
                    params[parameter] = trial.suggest_float(
                        parameter, values[0], values[-1], log=log)
                elif isinstance(values[0], str):
                    params[parameter] = trial.suggest_categorical(
                        parameter, values)
                elif isinstance(values[0], tuple):
                    params[parameter] = trial.suggest_categorical(
                        parameter, values)
                else:
                    raise ValueError(
                        f'Only int, float, str, tuple are supported for hyperparameter tuning, got {values[0]}')
            else:
                raise ValueError(
                    f'Only list, or tuple of values are supported, got {repr(values)}')
        return params

    def _suggest_params_ensemble(self, trial) -> dict:
        params_ensemble = {}
        params_ensemble['voting'] = trial.suggest_categorical(
            'voting', ['soft', 'hard'])

        for name, estimator in self._model.estimators:
            params = self._suggest_params_single(trial, estimator)
            params = {name + '__' + key: value for key,
                      value in params.items()}
            params_ensemble.update(params)
        return params_ensemble

    @staticmethod
    def _get_sampler(sampler_type: str) -> BaseSampler:
        """
        Get the sampler object based on the sampler type.

        Args:
            sampler_type (str): The sampler type.

        Returns:
            BaseSampler: The sampler object.
        """
        samplers = {
            'CmaEsSampler': CmaEsSampler(),
            'RandomSampler': RandomSampler(),
            'GridSampler': GridSampler({}),
            'TPESampler': TPESampler()
        }
        if sampler_type not in samplers:
            raise ValueError(
                f"Scaler type must be one of {list(samplers.keys())}")
        return samplers[sampler_type]

    @staticmethod
    def plot_results(study: optuna.study.Study) -> None:
        """
        Plot optimization history.

        Args:
            study (optuna.study.Study): The optuna study object.
        """
        fig = plot_optimization_history(study, error_bar=True)
        fig.show(config={"staticPlot": True})
        fig = plot_param_importances(study)
        fig.show(config={"staticPlot": True})

        fig_slice = plot_slice(study)
        fig_slice.show(config={"staticPlot": True})

        fig_contour = plot_contour(study)
        fig_contour.show(config={"staticPlot": True})

    @staticmethod
    def get_parameter_importance(study: optuna.study.Study) -> dict:
        """
        Compute parameter importances with FanovaImportanceEvaluator.

        Args:
            study (optuna.study.Study): The optuna study object.

        Returns:
            dict: Dictionary containing parameter importances.
        """
        return optuna.importance.get_param_importances(study=study)
