import polars as pl
from functools import lru_cache
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.model.train import MarchMadnessNN
from src.model.disk_ops import load_rf_model, load_nn_model, load_scaler, load_temperatures
from scipy.special import expit
import torch
import numpy as np


@lru_cache(maxsize=None)
def get_models() -> tuple[RandomForestClassifier, MarchMadnessNN, StandardScaler, float, float]:
    rf = load_rf_model()
    nn = load_nn_model()
    scaler = load_scaler()
    T_rf, T_nn = load_temperatures()
    return rf, nn, scaler, T_rf, T_nn


def simulate_game(a_features: pl.DataFrame, b_features: pl.DataFrame) -> tuple[float, float]:
    """
    :param a_features: Features df for team a
    :param b_features: Features df for team b
    :return: a_prob, b_prob
    """
    rf, nn, scaler, T_rf, T_nn = get_models()

    a_features = a_features.drop(["team", "season"]).to_numpy()
    b_features = b_features.drop(["team", "season"]).to_numpy()

    X = np.concatenate((a_features, b_features), axis=1)
    X = scaler.transform(X)

    # RF with temperature scaling
    rf_prob = rf.predict_proba(X)[0, 1]
    rf_prob = np.clip(rf_prob, 0.01, 0.99)
    rf_logit = np.log(rf_prob / (1 - rf_prob))
    rf_calibrated = expit(rf_logit / T_rf)

    # NN with temperature scaling
    X_t = torch.tensor(X, dtype=torch.float32)
    nn.eval()
    with torch.no_grad():
        nn_prob = nn(X_t).squeeze().item()
    nn_prob = np.clip(nn_prob, 0.01, 0.99)
    nn_logit = np.log(nn_prob / (1 - nn_prob))
    nn_calibrated = expit(nn_logit / T_nn)

    a_prob = (rf_calibrated + nn_calibrated) / 2
    b_prob = 1 - a_prob
    return a_prob, b_prob