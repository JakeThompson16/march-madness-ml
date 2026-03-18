
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import torch

LR_MODEL_PATH = "data/lr_model.joblib"
NN_MODEL_PATH = "data/nn_model.joblib"

def save_lr_model(model: LogisticRegression, scaler: StandardScaler):
    joblib.dump({"model": model, "scaler": scaler}, LR_MODEL_PATH)

def load_recent_lr_model()->tuple[LogisticRegression, StandardScaler]:
    bundle = joblib.load(LR_MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    return model, scaler
