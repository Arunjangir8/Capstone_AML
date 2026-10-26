"""Small helpers shared by all notebooks (paths, settings, metrics, plots).

Kept in one place so every notebook stays short and easy to read.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------- folders ----------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "raw"
PROCESSED_DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ROOT / "outputs" / "models"
RESULTS = ROOT / "outputs" / "results"
for folder in [PROCESSED_DATA, FIGURES, MODELS, RESULTS]:
    folder.mkdir(parents=True, exist_ok=True)

# ---------------- settings (same for every notebook) ----------------
RUL_CAP = 125   # maximum RUL used as target
WINDOW = 10     # rolling window size (cycles)
SEED = 42

COLORS = {"Baseline": "#8a8984", "Linear Regression": "#1baf7a", "Decision Tree": "#eb6834", "Random Forest": "#2a78d6"}
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})


COLUMNS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]


def load_raw(name):
    """Read one NASA text file, e.g. load_raw("train_FD001.txt")."""
    return pd.read_csv(RAW_DATA / name, sep=r"\s+", header=None, names=COLUMNS)


def load_features():
    """Load the feature tables made in notebook 02."""
    train = pd.read_csv(PROCESSED_DATA / "train_features.csv")
    test = pd.read_csv(PROCESSED_DATA / "test_features.csv")
    features = (PROCESSED_DATA / "feature_list.txt").read_text().split()
    return train, test, features


def evaluate(y_true, y_pred):
    """RMSE, MAE and R² rounded for display."""
    return {"RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
            "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
            "R2": round(float(r2_score(y_true, y_pred)), 3)}


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png")
    plt.show()


def plot_validation(name, y_true, y_pred, fig_name):
    """Left: predicted vs true RUL. Right: error vs true RUL."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(y_true, y_pred, s=3, alpha=0.3, color=COLORS[name])
    ax[0].plot([0, RUL_CAP], [0, RUL_CAP], "k--", lw=1, label="perfect prediction")
    ax[0].set(title=f"{name}: predicted vs true RUL (validation)", xlabel="True RUL", ylabel="Predicted RUL")
    ax[0].legend()
    ax[1].scatter(y_true, np.asarray(y_pred) - np.asarray(y_true), s=3, alpha=0.3, color=COLORS[name])
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set(title=f"{name}: error vs true RUL", xlabel="True RUL", ylabel="Predicted − True")
    savefig(fig_name)


def plot_test(name, y_true, y_pred, fig_name):
    """Left: predicted vs true RUL. Right: the 100 test engines sorted by true RUL."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(y_true, y_pred, color=COLORS[name])
    ax[0].plot([0, RUL_CAP], [0, RUL_CAP], "k--", lw=1)
    ax[0].set(title=f"{name}: official test set (100 engines)", xlabel="True RUL", ylabel="Predicted RUL")
    order = np.argsort(y_true)
    ax[1].plot(range(len(y_true)), y_true[order], color="black", lw=2, label="true RUL")
    ax[1].scatter(range(len(y_true)), y_pred[order], s=12, color=COLORS[name], label="predicted RUL")
    ax[1].set(title="Each test engine (sorted by true RUL)", xlabel="Test engine", ylabel="RUL")
    ax[1].legend()
    savefig(fig_name)


def stage_table(rul_raw, y_true, y_pred):
    """Error split into early life (true RUL > 100) and late life (true RUL <= 100)."""
    stage = np.where(np.asarray(rul_raw) > 100, "early (RUL > 100)", "late (RUL ≤ 100)")
    rows = []
    for st in ["early (RUL > 100)", "late (RUL ≤ 100)"]:
        mask = stage == st
        rows.append({"Stage": st, "Rows": int(mask.sum()),
                     **evaluate(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])})
    return pd.DataFrame(rows).set_index("Stage").drop(columns="R2")


def save_results(name, val, val_pred, test, test_pred, raw_only_rmse):
    """Save predictions + metrics so notebook 06 can compare all models without retraining."""
    slug = name.lower().replace(" ", "_")
    val[["unit", "cycle", "RUL_raw", "RUL"]].assign(predicted=val_pred).to_csv(
        RESULTS / f"{slug}_val_predictions.csv", index=False)
    test[["unit", "RUL_raw", "RUL"]].assign(predicted=test_pred).to_csv(
        RESULTS / f"{slug}_test_predictions.csv", index=False)
    pd.DataFrame([{"Model": name, "raw_sensors_only_val_RMSE": round(float(raw_only_rmse), 2)}]).to_csv(
        RESULTS / f"{slug}_raw_vs_engineered.csv", index=False)
    print(f"Saved predictions for {name} in outputs/results/")
