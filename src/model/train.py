from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.features.generate_features import build_matchup_df
from src.model.evaluate import metrics, visualize
from src.model.disk_ops import save_rf_model, save_nn_model, save_scaler, load_scaler


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


def train_rf_model(seasons: int | list[int], show_metrics=True) -> RandomForestClassifier:
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

    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    if show_metrics:
        metrics(y_test, y_prob, y_pred)
        visualize(y_test, y_prob)

    return model, scaler


def train_and_save_rf_model(seasons: int | list[int], show_metrics=True) -> RandomForestClassifier:
    model, scaler = train_rf_model(seasons, show_metrics)
    save_rf_model(model)
    save_scaler(scaler)
    return model, scaler


class MarchMadnessNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_nn_model(seasons: int | list[int], show_metrics=True) -> MarchMadnessNN:
    """
    Train a Neural Network model based on march madness games from seasons.
    Requires RF model to be trained and saved first (shares scaler).
    """
    if isinstance(seasons, int):
        seasons = [seasons]

    X_train, y_train, X_test, y_test = _prepare_data(seasons)

    scaler = load_scaler()
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MarchMadnessNN(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    for epoch in range(100):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    if show_metrics:
        model.eval()
        with torch.no_grad():
            y_prob = model(X_test_t).squeeze().numpy()
            y_pred = (y_prob >= 0.5).astype(int)
        metrics(y_test, y_prob, y_pred)
        visualize(y_test, y_prob)

    return model


def train_and_save_nn_model(seasons: int | list[int], show_metrics=True) -> MarchMadnessNN:
    model = train_nn_model(seasons, show_metrics)
    save_nn_model(model)
    return model