import optuna
import numpy as np
from engine import Engine
from optuna.samplers import CmaEsSampler, RandomSampler, GridSampler, TPESampler
from optuna.samplers._base import BaseSampler
from optuna.visualization import plot_optimization_history, plot_param_importances

class Tuner:
    def __init__(self, engine: Engine, n_trials: int=100, sampler_type: str='TPESampler') -> None:
        self.n_trials = n_trials
        self.sampler_type = self._get_sampler(sampler_type)
        self._engine = engine

    def objective(self, trial: optuna.Trial):
        """
        Objective function for optimization.

        Args:
            trial (optuna.Trial): The trial object.

        Returns:
            float: The objective value.
        """
        tune_params = self._suggest_params(trial)
        self._engine._model._model.set_params(**tune_params)
        metrics = self._engine.train(cv=True)
        return np.mean(metrics['test_f1_weighted'])

    def tune(self) -> optuna.study.Study:
        """
        Perform hyperparameter tuning.

        Returns:
            optuna.study.Study: The optuna study object.
        """
        study = optuna.create_study(study_name='optimization', direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        return study

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """
        Suggest hyperparameters for a trial.

        Args:
            trial (optuna.Trial): The trial object.

        Returns:
            dict: Dictionary containing suggested hyperparameters.
        """
        params = {}
        hyperparams = self._engine._model.get_parameter_space()

        for parameter, values in hyperparams.items():
            if isinstance(values, (list, tuple)):

                assert all(isinstance(e, type(values[0])) for e in values), \
                f'All elements must be of same object'

                if isinstance(values[0], int):
                    params[parameter] = trial.suggest_int(parameter, values[0], values[-1])
                elif isinstance(values[0], float):
                    params[parameter] = trial.suggest_float(parameter, values[0], values[-1])
                elif isinstance(values[0], str):
                    params[parameter] = trial.suggest_categorical(parameter, values)
                else:
                    raise ValueError(f'Only int, float, str are supported for hyperparameter tuning, got {values[0]}')
            else:
                raise ValueError(f'Only list, or tuple of values are supported, got {repr(values)}')
        return params
        
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
            raise ValueError(f"Scaler type must be one of {list(samplers.keys())}")
        return samplers[sampler_type]

    @staticmethod
    def plot_optimization_history(study: optuna.study.Study) -> None:
        """
        Plot optimization history.

        Args:
            study (optuna.study.Study): The optuna study object.
        """
        fig = plot_optimization_history(study, error_bar=True)
        fig.show(config={"staticPlot": True})
        fig = plot_param_importances(study)
        fig.show(config={"staticPlot": True})

    @staticmethod
    def get_parameter_importances(study: optuna.study.Study) -> dict:
        """
        Compute parameter importances with FanovaImportanceEvaluator.

        Args:
            study (optuna.study.Study): The optuna study object.

        Returns:
            dict: Dictionary containing parameter importances.
        """
        return optuna.importance.get_param_importances(study=study)