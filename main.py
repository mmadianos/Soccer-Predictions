import argparse
from src import Engine, Tuner, build_model, get_config, get_data
import pandas as pd
import logging


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def main(args):
    """
    Main entry point for training or tuning the model.
    """
    config = get_config(args=args)
    model = build_model(config)
    engine = Engine(config)
    X, y = get_data(config)

    if args.tune:
        tuner = Tuner(engine=engine,
                      model=model,
                      metric=config['METRIC'],
                      n_trials=config['N_TRIALS'],
                      sampler_type=config['SAMPLER_TYPE'])

        study = tuner.tune(X, y, save=config['SAVE_BEST_PARAMS'],
                           plot_tuning_results=config['PLOT_RESULTS'])
        return study

    if args.cross_validate:
        metrics = engine.cross_validate(X, y, model=model)
    else:
        pipeline = engine.train(X, y, model=model)
        metrics = engine.test(pipeline)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--cross_validate', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    arg = parser.parse_args()

    logger = setup_logger()
    logger.info("Starting the main script with %s", vars(arg))
    result = main(arg)
    if isinstance(result, dict):
        print(pd.DataFrame([result]))
    else:
        print(result)
