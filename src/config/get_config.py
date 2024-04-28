from config import params, tuning_params, ensemble_params, cv_params, holdout_params


def get_config():
    config_file = params

    if ensemble_params.get('MODEL', None):
        config_file.update(ensemble_params)
    
    if cv_params.get('CV', True):
        config_file.update(cv_params)
    else:
        config_file.update(holdout_params)



    
    


