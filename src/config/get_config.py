#from config.config import params, tuning_params, ensemble_params, cv_params, holdout_params
from .config import params, tuning_params, ensemble_params, cv_params, holdout_params


def get_config():
    config_file = params.copy()
    
    if cv_params.get('CV', True):
        config_file.update(cv_params)
    else:
        config_file.update(holdout_params)

    if ensemble_params.get('MODEL', None):
        config_file.update(ensemble_params)

    if tuning_params.get('TUNE', False):
        config_file.update(tuning_params)
        if config_file['CALIBRATION']:
            config_file['CALIBRATION'] = False
            print("Warning: Calibrated models cannot be used for tuning. Setting 'CALIBRATION' to False.")
        if config_file['SAVE_BEST_PARAMS']:
            config_file['SAVE_BEST_PARAMS'] = False
            print("Warning: Saving best parameters of ensemble model after tuning is not allowed. Setting 'SAVE_BEST_PARAMS' to False.")

    return config_file

#CANT HAVE CALIBRATION + ENSEMBLE + TUNING!
#CALIBRATION+ ENSEMBLE IS OK
#ENSEMBLE + TUNING IS OK
#CALIBRATION OK
#CALIBRATION + TUNING NO
#DONT SAVE IF ENSEMBLE IN TUNE!