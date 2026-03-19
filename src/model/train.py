from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import polars as pl
from src.features.generate_features import build_matchup_df
from src.model.evaluate import metrics, visualize
from src.model.disk_ops import save_rf_model, save_scaler, save_xgb_model, save_xgb_scaler


def _prepare_data(seasons: list[int]):
    df = build_matchup_df(seasons)
    non_features = ["Season", "team_a_name", "team_b_name", "team_a_won"]
    test_season = sorted(seasons)[-2]
    train_df = df.filter(pl.col("Season") < test_season)
    test_df = df.filter(pl.col("Season") == test_season)
    X_train = train_df.drop(non_features).to_numpy()
    y_train = train_df["team_a_won"].to_numpy()
    X_test = test_df.drop(non_features).to_numpy()
    y_test = test_df["team_a_won"].to_numpy()
    return X_train, y_train, X_test, y_test


def train_rf_model(seasons: int | list[int], show_metrics=True) -> tuple[RandomForestClassifier, StandardScaler]:
    """
    Train a Random Forest model based on march madness games from seasons.
    Also fits and saves the shared scaler.
    """
    if isinstance(seasons, int):
        seasons = [seasons]

    X_train, y_train, X_test, y_test = _prepare_data(seasons)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=42,
        min_samples_leaf=3
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    if show_metrics:
        metrics(y_test, y_prob, y_pred)
        visualize(y_test, y_prob)

    return model, scaler


def train_and_save_rf_model(seasons: int | list[int], show_metrics=True) -> tuple[RandomForestClassifier, StandardScaler]:
    model, scaler = train_rf_model(seasons, show_metrics)
    save_rf_model(model)
    save_scaler(scaler)
    return model, scaler


def train_xgb_model(seasons: int | list[int], show_metrics=True) -> tuple[CalibratedClassifierCV, StandardScaler]:
    """
    Train a calibrated XGBoost model based on march madness games from seasons.
    """
    if isinstance(seasons, int):
        seasons = [seasons]

    X_train, y_train, X_test, y_test = _prepare_data(seasons)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    base_xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model = CalibratedClassifierCV(base_xgb, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    if show_metrics:
        metrics(y_test, y_prob, y_pred)
        visualize(y_test, y_prob)

    return model, scaler


def train_and_save_xgb_model(seasons: int | list[int], show_metrics=True) -> tuple[CalibratedClassifierCV, StandardScaler]:
    model, scaler = train_xgb_model(seasons, show_metrics)
    save_xgb_model(model)
    save_xgb_scaler(scaler)
    return model, scaler