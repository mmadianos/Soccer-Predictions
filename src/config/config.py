# ================= general ================= #
params = {
    'TRAINING_FILE': '../vault/Championship.csv',
    'PARAMS_PATH': '../vault/tuned_params/',
    'MODEL': 'DecisionClassifier',
    #'PARAMETERS': None,
    'SCALER_TYPE': 'Standard',
    'ENCODER_TYPE': 'Label',
    'IMPUTER_TYPE': None,
    'n_jobs': -1,
    'random_state': 42,
    'CALIBRATION': False,
    'PLOT_CONFUSION': True, #delete
}
# ================= holdout ================= #
holdout_params = {
    'PLOT_CONFUSION': True,
    'TEST_SIZE': 0.2,
}
# ================= cross validation ================= #
cv_params = {
    'CV': True,
    'CV_STRATEGY': 'StratifiedKFold',
    'PLOT_CONFUSION': False,
}
# ================= tuning ================= #
tuning_params = {
    'TUNE': True,
    'PLOT_RESULTS': False,
    'SAVE_BEST_PARAMS': True
}

# ================= model ensemble ================= #
ensemble_params = {
    #'MODEL': ('DecisionClassifier', 'KNNClassifier'),
    'CALIBRATION': False
}