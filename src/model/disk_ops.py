
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib


def save_model(model: LogisticRegression, scaler: StandardScaler):
    joblib.dump({"model": model, "scaler": scaler}, "data/model.joblib")

def load_recent_model()->tuple[LogisticRegression, StandardScaler]:
    bundle = joblib.load("data/model.joblib")
    model = bundle["model"]
    scaler = bundle["scaler"]
    return model, scaler
