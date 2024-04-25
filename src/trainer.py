import argparse
import joblib
from config import params
from engine import Engine
from tuner import Tuner
import importlib


def get_model(model_name, params_path):
        """
        
        """
        model_params = joblib.load(params_path)
        model_package = f'models.estimators.{model_name.lower()}'
        mod = importlib.import_module(model_package)
        return getattr(mod, model_name)(**model_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', type=str)
    parser.add_argument('--config', type=str, help='Evaluation config file path')
    args = parser.parse_args()

    model = get_model(params['MODEL'], params['PARAMS_PATH'])
    engine = Engine(model, params)

    if params.get('TUNE', False):
        tuner = Tuner(engine=engine, n_trials=10)
        study = tuner.tune(save=True, plot_tuning_results=True)
    
        importance = tuner.get_parameter_importances(study)
        print(study.best_params)
        print(f'{study.best_value:.3f}')
        print(importance)
    else:
        metrics = engine.train(params.get('CV', True))
        print(metrics)
