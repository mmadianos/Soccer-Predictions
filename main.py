import argparse
from src import Engine, Tuner, build_model, get_config


def main(args):
    config = get_config(args=args)
    model = build_model(config)
    engine = Engine(config)

    if args.tune:
        tuner = Tuner(engine=engine,
                      model=model,
                      metric=config['METRIC'],
                      n_trials=config['N_TRIALS'],
                      sampler_type=config['SAMPLER_TYPE'])
        
        study = tuner.tune(save=config['SAVE_BEST_PARAMS'],
                           plot_tuning_results=config['PLOT_RESULTS'])
    
        #importance = tuner.get_parameter_importance(study)
    else:
        metrics = engine.train(model=model,
                               cv=args.cross_validate)
        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', type=str)
    parser.add_argument('--ensemble', default=False, type=bool)
    parser.add_argument('--cross_validate', default=False, type=bool)
    parser.add_argument('--tune', default=False, type=bool)
    parser.add_argument('--calibrate', default=False, type=bool)
    #parser.add_argument('--config', type=str, help='Config file path')
    args = parser.parse_args()
    result = main(args)
    print(result)
    