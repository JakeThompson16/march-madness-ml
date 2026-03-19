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


def simulate_game(a_features: pl.DataFrame, b_features: pl.DataFrame) -> tuple[float, float]:
    """
    :param a_features: Features df for team a
    :param b_features: Features df for team b
    :return: a_prob, b_prob
    """
    rf, xgb, rf_scaler, xgb_scaler, T_rf, T_xgb = get_models()

    a = a_features.drop(["team", "season"]).to_numpy()
    b = b_features.drop(["team", "season"]).to_numpy()
    X = np.concatenate((a, b), axis=1)

    # RF with temperature scaling
    X_rf = rf_scaler.transform(X)
    rf_prob = rf.predict_proba(X_rf)[0, 1]
    rf_prob = np.clip(rf_prob, 0.01, 0.99)
    rf_logit = np.log(rf_prob / (1 - rf_prob))
    rf_calibrated = expit(rf_logit / T_rf)

    # XGB with temperature scaling
    X_xgb = xgb_scaler.transform(X)
    xgb_prob = xgb.predict_proba(X_xgb)[0, 1]
    xgb_prob = np.clip(xgb_prob, 0.01, 0.99)
    xgb_logit = np.log(xgb_prob / (1 - xgb_prob))
    xgb_calibrated = expit(xgb_logit / T_xgb)

    a_prob = (rf_calibrated + xgb_calibrated) / 2
    b_prob = 1 - a_prob
    return a_prob, b_prob