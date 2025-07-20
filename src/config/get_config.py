from .config import params, tuning_params, ensemble_params, cv_params, holdout_params
import logging

logger = logging.getLogger(__name__)


def get_config(args, custom=None) -> dict:
    config_file = params.copy()
    config_file['CALIBRATION'] = args.calibrate

    if args.ensemble:
        config_file.update(ensemble_params)

    if args.cross_validate:
        config_file.update(cv_params)
    else:
        config_file.update(holdout_params)

    if args.tune:
        config_file.update(tuning_params)
        if args.calibrate:
            logger.warning(
                "Calibrated models cannot be used for tuning. Setting 'CALIBRATION' to False."
            )
            config_file['CALIBRATION'] = False
            if config_file['SAVE_BEST_PARAMS']:
                config_file['SAVE_BEST_PARAMS'] = False
                logger.warning(
                    "Saving best parameters of ensemble model after tuning is not allowed. Setting 'SAVE_BEST_PARAMS' to False."
                )

    if custom:
        for key, value in custom.items():
            if key in config_file:
                config_file[key] = value
            else:
                logger.warning(
                    "Custom config parameter '%s' is not valid. Skipping.",
                    key)

    return config_file
