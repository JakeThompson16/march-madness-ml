import polars as pl
from functools import lru_cache
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from src.model.disk_ops import load_rf_model, load_xgb_model, load_scaler, load_xgb_scaler, load_temperatures
from scipy.special import expit
import numpy as np


@lru_cache(maxsize=None)
def get_models() -> tuple[RandomForestClassifier, CalibratedClassifierCV, StandardScaler, StandardScaler, float, float]:
    rf = load_rf_model()
    xgb = load_xgb_model()
    rf_scaler = load_scaler()
    xgb_scaler = load_xgb_scaler()
    T_rf, T_xgb = load_temperatures()
    return rf, xgb, rf_scaler, xgb_scaler, T_rf, T_xgb


def _get_prob(X: np.ndarray, rf, xgb, rf_scaler, xgb_scaler, T_rf, T_xgb) -> float:
    X_rf = rf_scaler.transform(X)
    rf_prob = np.clip(rf.predict_proba(X_rf)[0, 1], 0.01, 0.99)
    rf_calibrated = expit(np.log(rf_prob / (1 - rf_prob)) / T_rf)

    X_xgb = xgb_scaler.transform(X)
    xgb_prob = np.clip(xgb.predict_proba(X_xgb)[0, 1], 0.01, 0.99)
    xgb_calibrated = expit(np.log(xgb_prob / (1 - xgb_prob)) / T_xgb)

    return (rf_calibrated + xgb_calibrated) / 2


def simulate_game(a_features: pl.DataFrame, b_features: pl.DataFrame) -> tuple[float, float]:
    """
    :param a_features: Features df for team a
    :param b_features: Features df for team b
    :return: a_prob, b_prob
    """
    rf, xgb, rf_scaler, xgb_scaler, T_rf, T_xgb = get_models()

    a = a_features.drop(["team", "season"]).to_numpy()
    b = b_features.drop(["team", "season"]).to_numpy()

    X_ab = np.concatenate((a, b), axis=1)
    p_ab = _get_prob(X_ab, rf, xgb, rf_scaler, xgb_scaler, T_rf, T_xgb)

    a_seed = a_features["seed"][0]
    b_seed = b_features["seed"][0]

    if a_seed == b_seed:
        X_ba = np.concatenate((b, a), axis=1)
        p_ba = _get_prob(X_ba, rf, xgb, rf_scaler, xgb_scaler, T_rf, T_xgb)
        a_prob = (p_ab + (1 - p_ba)) / 2
    else:
        a_prob = p_ab

    b_prob = 1 - a_prob
    return a_prob, b_prob