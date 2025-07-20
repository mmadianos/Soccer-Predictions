"""
A project for soccer match predictions using machine learning.
It includes modules for:
- Data preprocessing
- Model training
- Evaluation
- Hyperparameter tuning.
"""

from .tuner import Tuner
from .engine import Engine
from .config.get_config import get_config
from .models.get_model import build_model
from .data import get_data

__all__ = [
    "Engine",
    "Tuner",
    "build_model",
    "get_config",
    "get_data"
]

__version__ = '0.0.1'
