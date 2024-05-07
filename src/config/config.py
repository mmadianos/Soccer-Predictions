# ================= general ================= #
params = {
    'TRAINING_FILE': 'vault/Championship.csv',
    'PARAMS_PATH': '../vault/tuned_params/',
    'MODEL': 'DecisionClassifier',
    'SCALER_TYPE': 'Standard',
    'ENCODER_TYPE': 'Label',
    'IMPUTER_TYPE': None,
    'n_jobs': -1,
    'random_state': 42,
    'CALIBRATION': False,
    'CV': True,
    'ENSEMBLE': False,
    'TUNE': False
}
# ================= holdout ================= #
holdout_params = {
    'PLOT_CONFUSION': True,
    'TEST_SIZE': 0.2,
}
# ================= cross validation ================= #
cv_params = {
    'CV_STRATEGY': 'StratifiedKFold',
    'CV_SPLITS': 5,
    'PLOT_CONFUSION': False,
}
# ================= tuning ================= #
tuning_params = {
    'N_TRIALS': 100,
    'SAMPLER_TYPE': 'TPESampler',
    'PLOT_RESULTS': True, #
    'SAVE_BEST_PARAMS': True,
    'CALIBRATION': False
}
# ================= model ensemble ================= #
ensemble_params = {
    'MODEL': ('DecisionClassifier', 'KNNClassifier'),
    'SAVE_BEST_PARAMS': False
}
