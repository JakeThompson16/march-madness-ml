

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import polars as pl
from src.features.generate_features import build_matchup_df
from src.model.evaluate import lr_metrics, lr_visualize
from src.model.disk_ops import save_lr_model


def train_lr_model(seasons: int | list[int], show_metrics=True) -> tuple[LogisticRegression, StandardScaler]:
    """
    Train a Logistic Regression model based on march madness games from seasons
    """
    if isinstance(seasons, int):
        seasons = [seasons]

    model = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        class_weight="balanced",
        max_iter=1000
    )

    df = build_matchup_df(seasons)
    non_features = ["Season", "team_a_name", "team_b_name", "team_a_won"]

    test_season = sorted(seasons)[-2]
    train_df = df.filter(pl.col("Season") < test_season)
    test_df = df.filter(pl.col("Season") == test_season)

    X_train = train_df.drop(non_features).to_numpy()
    y_train = train_df["team_a_won"].to_numpy()
    X_test = test_df.drop(non_features).to_numpy()
    y_test = test_df["team_a_won"].to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    if show_metrics:
        lr_metrics(y_test, y_prob, y_pred)
        lr_visualize(y_test, y_prob)

    return model, scaler

def train_and_save_lr_model(seasons: int | list[int], show_metrics=True)->tuple[LogisticRegression, StandardScaler]:
    model, scaler = train_lr_model(seasons, show_metrics)
    save_lr_model(model, scaler)
    return model, scaler
