
import polars as pl
from functools import lru_cache
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.model.disk_ops import load_recent_lr_model
import numpy as np

@lru_cache(maxsize=None)
def get_lr_model()->tuple[LogisticRegression, StandardScaler]:
    model, scaler = load_recent_lr_model()
    return model, scaler

def simulate_game(a_features: pl.DataFrame, b_features: pl.DataFrame)->tuple[int, int]:
    """
    :param a_features: Features df for team a
    :param b_features: Features df for team b
    :return: a_prob, b_prob
    """
    model, scaler = get_lr_model()

    a_features = a_features.drop(["team", "season"]).to_numpy()
    b_features = b_features.drop(["team", "season"]).to_numpy()

    X = np.concatenate((a_features, b_features), axis=1)
    X = scaler.transform(X)

    a_prob = model.predict_proba(X)[0, 1]
    b_prob = 1 - a_prob
    return a_prob, b_prob
