import argparse
from engine import Engine
from tuner import Tuner
from models.get_model import build_model
from config.get_config import get_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', type=str)
    #parser.add_argument('--config', type=str, help='Config file path')
    args = parser.parse_args()

    config = get_config()
    model = build_model(config)
    engine = Engine(model, config)

    if config.get('TUNE', False):
        tuner = Tuner(engine=engine,
                      n_trials=config['N_TRIALS'],
                      sampler_type=config['SAMPLER_TYPE'])
        
        study = tuner.tune(save=config['SAVE_BEST_PARAMS'],
                           plot_tuning_results=config['PLOT_RESULTS'])
    
        importance = tuner.get_parameter_importance(study)
        print('BEST PARAMS:', study.best_params)
        print(f'BEST METRIC: {study.best_value:.3f}')
        print('IMPORTANCE WEIGHTS', importance)
    else:
        metrics = engine.train(config['CV'])
        print('CV RESULT:', metrics)
