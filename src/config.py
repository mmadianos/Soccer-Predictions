params = {
    'TRAINING_FILE': '../vault/Championship.csv',
    'PARAMS_PATH': '../vault/tuned_params/decisiontreeclassifier.pkl',
    'MODEL': 'DecisionClassifier',
    'TEST_SIZE': 0.2,
    'PARAMETERS': None,
    'PLOT_CONFUSION': True,
    'CV_STRATEGY': 'StratifiedKFold',
    'SCALER_TYPE': 'Standard',
    'ENCODER_TYPE': 'Label',
    'IMPUTER_TYPE': None,
    'CV': True,
    'TUNE': False
}

scikit_params = {
    'n_jobs': -1,
    'random_state': 42
}

tuning_params = {
    'TUNE': True,
    'PLOT_RESULTS': False,
    'SAVE_BEST_PARAMS': False
}


eda_params = {
    'TRAINING_FILE': '../vault/Championship.csv',
    'TRAINING_FILE_NOTES': '../vault/notes.txt',
}