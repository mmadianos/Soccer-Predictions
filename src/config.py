params = {
    'TRAINING_FILE': '../data/Championship.csv',
    'MODEL_OUTPUT': '../models/scikit',
    'MODEL': 'DecisionClassifier',
    'TEST_SIZE': 0.2,
    'PARAMETERS': None,
    'PLOT_CONFUSION': True,
    'CV_STRATEGY': 'StratifiedKFold',
    'SCALER_TYPE': 'Standard',
    'ENCODER_TYPE': 'Label',
    'IMPUTER_TYPE': None,
    'TUNE': True
}

scikit_params = {
    'n_jobs': -1,
    'random_state': 42
}