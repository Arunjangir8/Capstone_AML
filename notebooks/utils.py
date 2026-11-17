"""Small helpers shared by all notebooks (paths, settings, metrics, plots).

Kept in one place so every notebook stays short and easy to read.
Phase 1 (AML) & Phase 2 (Deep Learning PyTorch Sequence Models).
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
RUL_CAP = 125    # maximum RUL used as target
WINDOW = 10      # rolling window size (cycles) for tabular ML
SEQ_LEN = 30     # time sequence length for DL sequence models
SEED = 42

COLORS = {
    "Baseline": "#8a8984",
    "Linear Regression": "#1baf7a",
    "Decision Tree": "#eb6834",
    "Random Forest": "#2a78d6",
    "LSTM": "#9b5de5",
    "1D-CNN": "#e63946",
    "CNN-LSTM": "#00bbf9",
}
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})

COLUMNS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]
USEFUL_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]


def load_raw(name):
    """Read one NASA C-MAPSS raw text file.

    Parameters:
        name (str): Filename inside data/raw (e.g. 'train_FD001.txt').

    Returns:
        pd.DataFrame: Tabular sensor and engine operational data.
    """
    return pd.read_csv(RAW_DATA / name, sep=r"\s+", header=None, names=COLUMNS)


def compute_nasa_score(y_true, y_pred):
    """Compute official NASA asymmetric scoring metric for RUL prediction.

    Penalizes late predictions more severely than early predictions:
    d = y_pred - y_true
    s = exp(-d/13) - 1 for d < 0 (early prediction)
    s = exp(d/10) - 1 for d >= 0 (late prediction)
    """
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return round(float(np.sum(scores)), 2)


def evaluate(y_true, y_pred):
    """Compute regression evaluation metrics (RMSE, MAE, R²).

    Parameters:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Model predictions.

    Returns:
        dict: Rounded evaluation metrics dictionary.
    """
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
    }


def savefig(name):
    """Format figure layout, save PNG to outputs/figures, and close figure."""
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close()


def plot_validation(name, y_true, y_pred, fig_name):
    """Generate validation diagnostic plots.

    Left subplot: Predicted vs True RUL scatter plot.
    Right subplot: Residual error vs True RUL scatter plot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(y_true, y_pred, s=3, alpha=0.3, color=COLORS.get(name, "#2a78d6"))
    ax[0].plot([0, RUL_CAP], [0, RUL_CAP], "k--", lw=1, label="perfect prediction")
    ax[0].set(title=f"{name}: predicted vs true RUL (validation)", xlabel="True RUL", ylabel="Predicted RUL")
    ax[0].legend()
    ax[1].scatter(y_true, np.asarray(y_pred) - np.asarray(y_true), s=3, alpha=0.3, color=COLORS.get(name, "#2a78d6"))
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set(title=f"{name}: error vs true RUL", xlabel="True RUL", ylabel="Predicted − True")
    savefig(fig_name)


def plot_test(name, y_true, y_pred, fig_name):
    """Generate official test set performance diagnostic plots.

    Left subplot: Predicted vs True RUL scatter plot.
    Right subplot: Test engines sorted by true RUL with model predictions overlaid.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(y_true, y_pred, color=COLORS.get(name, "#2a78d6"))
    ax[0].plot([0, RUL_CAP], [0, RUL_CAP], "k--", lw=1)
    ax[0].set(title=f"{name}: official test set (100 engines)", xlabel="True RUL", ylabel="Predicted RUL")
    order = np.argsort(y_true)
    ax[1].plot(range(len(y_true)), y_true[order], color="black", lw=2, label="true RUL")
    ax[1].scatter(range(len(y_true)), y_pred[order], s=12, color=COLORS.get(name, "#2a78d6"), label="predicted RUL")
    ax[1].set(title="Each test engine (sorted by true RUL)", xlabel="Test engine", ylabel="RUL")
    ax[1].legend()
    savefig(fig_name)


def stage_table(rul_raw, y_true, y_pred):
    """Calculate error metrics stratified by engine life stage.

    Splits predictions into:
    - Early life stage: true RUL > 100 cycles
    - Late life stage: true RUL <= 100 cycles
    """
    stage = np.where(np.asarray(rul_raw) > 100, "early (RUL > 100)", "late (RUL ≤ 100)")
    rows = []
    for st in ["early (RUL > 100)", "late (RUL ≤ 100)"]:
        mask = stage == st
        rows.append({
            "Stage": st,
            "Rows": int(mask.sum()),
            **evaluate(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])
        })
    return pd.DataFrame(rows).set_index("Stage").drop(columns="R2")


def gen_sequence_data(df, seq_length=30, feature_cols=None):
    """Generate sliding window 3D array (samples, seq_length, num_features) per engine.

    Ensures no sequence crosses engine boundaries.
    """
    if feature_cols is None:
        feature_cols = USEFUL_SENSORS
    seq_list, target_list, raw_rul_list, unit_list = [], [], [], []
    for unit, group in df.groupby("unit"):
        data = group[feature_cols].values
        targets = group["RUL"].values if "RUL" in group.columns else None
        raw_ruls = group["RUL_raw"].values if "RUL_raw" in group.columns else None
        num_elements = data.shape[0]
        for start, stop in zip(range(0, num_elements - seq_length + 1), range(seq_length, num_elements + 1)):
            seq_list.append(data[start:stop, :])
            if targets is not None:
                target_list.append(targets[stop - 1])
            if raw_ruls is not None:
                raw_rul_list.append(raw_ruls[stop - 1])
            unit_list.append(unit)
    X = np.array(seq_list, dtype=np.float32)
    y = np.array(target_list, dtype=np.float32) if target_list else None
    y_raw = np.array(raw_rul_list, dtype=np.float32) if raw_rul_list else None
    units = np.array(unit_list)
    return X, y, y_raw, units


def gen_test_sequences(df, seq_length=30, feature_cols=None):
    """Extract last seq_length window for each test engine up to its final cycle."""
    if feature_cols is None:
        feature_cols = USEFUL_SENSORS
    seq_list, unit_list = [], []
    for unit, group in df.groupby("unit"):
        data = group[feature_cols].values
        if data.shape[0] >= seq_length:
            seq_list.append(data[-seq_length:, :])
        else:
            pad = np.repeat(data[0:1, :], seq_length - data.shape[0], axis=0)
            seq_list.append(np.vstack([pad, data]))
        unit_list.append(unit)
    return np.array(seq_list, dtype=np.float32), np.array(unit_list)
