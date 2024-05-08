# ================= general ================= #
params = {
    'TRAINING_FILE': 'Championship.csv',
    'PARAMS_PATH': '../vault/tuned_params/',
    'MODEL': 'DecisionClassifier',
    'SCALER_TYPE': 'Standard',
    'ENCODER_TYPE': 'Label',
    'IMPUTER_TYPE': None,
    'n_jobs': -1,
    'random_state': 42
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
    'SCORING': ['precision_macro', 'accuracy', 'recall_macro', 'f1_weighted', 'roc_auc_ovr'],
    'PLOT_CONFUSION': False,
}
# ================= tuning ================= #
tuning_params = {
    'N_TRIALS': 10,
    'METRIC': 'test_roc_auc_ovr',
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
