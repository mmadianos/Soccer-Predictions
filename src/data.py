import os
import pandas as pd
from typing import Tuple
from .feature_engineering.get_features import FeaturesEngine


def get_data(config) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess the dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target variable.
    """
    absolute_data_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_data_dir = os.path.join(
        absolute_data_dir, f'../vault/{config["TRAINING_FILE"]}')

    keep_col = ['HomeTeam', 'AwayTeam', 'B365H',
                'B365D', 'B365A', 'FTR', 'FTHG', 'FTAG']
    df = pd.read_csv(absolute_data_dir, usecols=keep_col)
    feature_engine = FeaturesEngine()
    df = feature_engine.generate_features(df)
    df = df.drop(columns=['HomeTeam', 'AwayTeam'])
    X, y = df.drop(columns=['FTR', 'FTHG', 'FTAG']), df.FTR
    return X, y
