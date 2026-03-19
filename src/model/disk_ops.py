from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import torch

RF_MODEL_PATH = "data/rf_model.joblib"
NN_MODEL_PATH = "data/nn_model.joblib"
SCALER_PATH = "data/scaler.joblib"
TEMPERATURES_PATH = "data/temperatures.joblib"
XGB_MODEL_PATH = "data/xgb_model.joblib"
XGB_SCALER_PATH = "data/xgb_scaler.joblib"

def save_xgb_model(model):
    joblib.dump({"model": model}, XGB_MODEL_PATH)

def load_xgb_model():
    bundle = joblib.load(XGB_MODEL_PATH)
    return bundle["model"]

def save_xgb_scaler(scaler: StandardScaler):
    joblib.dump(scaler, XGB_SCALER_PATH)

def load_xgb_scaler() -> StandardScaler:
    return joblib.load(XGB_SCALER_PATH)

def save_temperatures(T_rf: float, T_nn: float):
    joblib.dump({"T_rf": T_rf, "T_nn": T_nn}, TEMPERATURES_PATH)

def load_temperatures() -> tuple[float, float]:
    bundle = joblib.load(TEMPERATURES_PATH)
    return bundle["T_rf"], bundle["T_nn"]

def save_scaler(scaler: StandardScaler):
    joblib.dump(scaler, SCALER_PATH)

def load_scaler() -> StandardScaler:
    return joblib.load(SCALER_PATH)


def save_rf_model(model: RandomForestClassifier):
    joblib.dump({"model": model}, RF_MODEL_PATH)

def load_rf_model() -> RandomForestClassifier:
    bundle = joblib.load(RF_MODEL_PATH)
    return bundle["model"]


def save_nn_model(model: torch.nn.Module):
    joblib.dump({"model": model}, NN_MODEL_PATH)

def load_nn_model() -> torch.nn.Module:
    bundle = joblib.load(NN_MODEL_PATH)
    return bundle["model"]