
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np


def metrics(y_test: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"Log Loss:  {log_loss(y_test, y_prob):.4f}")
    print(f"Brier:     {brier_score_loss(y_test, y_prob):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")


def visualize(y_test: np.ndarray, y_prob: np.ndarray) -> None:
    wins = y_prob[y_test == 1]
    losses = y_prob[y_test == 0]

    plt.figure(figsize=(8, 5))
    plt.hist(wins, bins=20, alpha=0.6, color="green", label="Team A Won")
    plt.hist(losses, bins=20, alpha=0.6, color="red", label="Team A Lost")
    plt.axvline(0.5, color="black", linestyle="--", label="50% threshold")
    plt.xlabel("Predicted Win Probability")
    plt.ylabel("Count")
    plt.title("Predicted Probability Distribution by Outcome")
    plt.legend()
    plt.tight_layout()
    plt.show()