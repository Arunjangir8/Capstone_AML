"""Builds predictive_maintenance_aml_phase1.ipynb from cell definitions.

Run:  python build_notebook.py
Then: jupyter nbconvert --to notebook --execute --inplace predictive_maintenance_aml_phase1.ipynb
"""
import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# =============================================================================
# 1. Introduction
# =============================================================================
md(r"""
# Predictive Maintenance of Turbofan Engines — Phase 1 (Classical Applied Machine Learning)

**Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation, subset **FD001**
(one operating condition, one fault mode — HPC degradation).

**Scope of Phase 1:** classical ML only (Random Forest, XGBoost, Elastic Net + baselines).
No deep learning is implemented in this notebook.

**Pipeline**

```
Raw sensor data → EDA → RUL label construction → Preprocessing → Engine-level split
→ Degradation feature engineering → Leakage checks → Baselines / RF / XGBoost / Elastic Net
→ Validation comparison → Interpretability → Early vs late analysis → Official test evaluation
```

### How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib jupyter
# macOS only — XGBoost needs the OpenMP runtime:
brew install libomp
jupyter notebook predictive_maintenance_aml_phase1.ipynb
```

Set the dataset folder with the environment variable `CMAPSS_DATA_DIR`, or edit `DATA_DIR`
in the configuration cell. All outputs (figures, models, metrics, predictions) are written to `results/`.
""")

# =============================================================================
# 2. Problem statement
# =============================================================================
md(r"""
## 2. Problem Statement and Objectives

Each engine in the fleet starts healthy (with unknown manufacturing variation and initial wear),
develops a fault, and degrades until failure. At every operating cycle we observe 3 operating
settings and 21 sensors. **Goal:** given an engine's sensor history up to the current cycle,
estimate its **Remaining Useful Life (RUL)** — the number of cycles left before failure.

RUL is a **continuous, non-negative number of cycles**, so this is a **regression** problem
(not classification): we care how far off the estimate is, not just whether it falls in a class.

**Objectives**
1. Validate and understand the raw data (constant sensors, lifetimes, degradation trends).
2. Construct a correct, leakage-free RUL target (with configurable capping).
3. Engineer interpretable degradation features (rolling means/stds, rolling slopes, health indicators).
4. Train and tune Random Forest, XGBoost and Elastic Net with an **engine-level** validation split.
5. Compare against simple baselines, analyse errors by degradation stage, and evaluate once on the official NASA test set.
6. Answer the research question: *Can carefully engineered sensor degradation features help
   classical ML models accurately estimate the RUL of turbofan engines?*
""")

# =============================================================================
# 3. Setup / config / loading
# =============================================================================
md(r"""
## 3. Dataset Loading and Validation

### 3.1 Imports and configuration
Everything that affects the experiment (paths, RUL cap, windows, seeds, search budget) lives in one `CONFIG` dictionary.
""")

code(r"""
import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception as exc:  # ImportError, or XGBoostError when libomp is missing on macOS
    XGB_AVAILABLE = False
    print("XGBoost could not be imported:", exc)
    print("Install with:  pip install xgboost   (macOS additionally: brew install libomp)")

warnings.filterwarnings("ignore", category=FutureWarning)

print("pandas", pd.__version__, "| numpy", np.__version__, "| scikit-learn", sklearn.__version__,
      "| xgboost", xgb.__version__ if XGB_AVAILABLE else "N/A", "| matplotlib", matplotlib.__version__)
""")

code(r"""
CONFIG = {
    # --- paths -------------------------------------------------------------
    "data_dir": os.environ.get("CMAPSS_DATA_DIR", "../CMAPSSData"),
    "subset": "FD001",
    "results_dir": "results",
    # --- target ------------------------------------------------------------
    "rul_cap": 125,                 # set to None to disable capping
    # --- split -------------------------------------------------------------
    "val_fraction": 0.20,           # fraction of TRAINING ENGINES held out for validation
    "seed": 42,
    # --- sensor screening (computed on training engines only) -------------
    "min_unique_values": 3,         # <3 distinct values -> constant / near-constant
    "min_rel_std": 1e-6,            # std / |mean| below this -> near-constant (s8/s13 have rel. std ~3e-5 yet trend clearly, so keep this tiny)
    # --- feature engineering -----------------------------------------------
    "roll_windows": [5, 10, 20],    # rolling mean/std windows (cycles)
    "slope_windows": [10, 20],      # rolling OLS slope windows (cycles)
    "baseline_cycles": 10,          # first N cycles define each engine's healthy baseline
    "diff_lag": 10,                 # change vs. N cycles earlier
    # --- tuning --------------------------------------------------------------
    "cv_folds": 3,                  # GroupKFold over training engines
    "n_iter_search": 12,            # random-search budget per model
    "n_jobs": -1,
    # --- early/late analysis -----------------------------------------------
    "stage_threshold": 100,         # true RUL > threshold -> "early", else "late"
}

DATA_DIR = Path(CONFIG["data_dir"])
SUBSET = CONFIG["subset"]
SEED = CONFIG["seed"]
np.random.seed(SEED)

RESULTS = Path(CONFIG["results_dir"])
FIG_DIR, MODEL_DIR, METRIC_DIR, PRED_DIR = (RESULTS / d for d in ["figures", "models", "metrics", "predictions"])
for d in [FIG_DIR, MODEL_DIR, METRIC_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

with open(RESULTS / "config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

# --- consistent plotting style --------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.6, "grid.alpha": 0.35})
# Fixed colour per model (colour follows the entity everywhere in the notebook)
MODEL_COLORS = {"Baseline (mean)": "#8a8984", "Baseline (cycle-only)": "#52514e",
                "Random Forest": "#2a78d6", "XGBoost": "#eb6834", "Elastic Net": "#1baf7a"}
STAGE_COLORS = {"early": "#2a78d6", "late": "#e34948"}


def save_fig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path
""")

md(r"""
### 3.2 Load the raw files
The files are whitespace-separated with no header (and a trailing space on each line), so they are read with
`sep=r"\s+"` and explicit column names. A clear error is raised if any file is missing.
""")

code(r"""
INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
RAW_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def resolve_files(data_dir: Path, subset: str) -> dict:
    files = {"train": data_dir / f"train_{subset}.txt",
             "test": data_dir / f"test_{subset}.txt",
             "rul": data_dir / f"RUL_{subset}.txt"}
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing C-MAPSS file(s):\n  " + "\n  ".join(missing) +
            f"\nSet CONFIG['data_dir'] or the CMAPSS_DATA_DIR environment variable to the folder "
            f"containing train_{subset}.txt, test_{subset}.txt and RUL_{subset}.txt.")
    return files


def load_cmapss(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(RAW_COLS):
        raise ValueError(f"{path.name}: expected {len(RAW_COLS)} columns, found {df.shape[1]}")
    df.columns = RAW_COLS
    df[INDEX_COLS] = df[INDEX_COLS].astype(int)
    return df.sort_values(INDEX_COLS).reset_index(drop=True)


FILES = resolve_files(DATA_DIR, SUBSET)
train_raw = load_cmapss(FILES["train"])
test_raw = load_cmapss(FILES["test"])
rul_official = pd.read_csv(FILES["rul"], sep=r"\s+", header=None, names=["true_rul"])
rul_official["unit"] = np.arange(1, len(rul_official) + 1)

print(f"train: {train_raw.shape}  test: {test_raw.shape}  RUL file: {rul_official.shape}")
train_raw.head()
""")

md(r"""
### 3.3 Data validation
Dimensions, dtypes, missing values, duplicates, engine counts, cycle continuity and alignment between the test file and the RUL file.
""")

code(r"""
def validate_frame(df, name):
    n_units = df["unit"].nunique()
    # cycles must run 1..max without gaps inside every engine
    contiguous = df.groupby("unit")["cycle"].apply(lambda c: (c.values == np.arange(1, len(c) + 1)).all()).all()
    return {"dataset": name, "rows": len(df), "columns": df.shape[1], "engines": n_units,
            "missing_values": int(df.isna().sum().sum()), "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_unit_cycle": int(df.duplicated(INDEX_COLS).sum()),
            "cycles_contiguous_from_1": bool(contiguous),
            "min_cycles_per_engine": int(df.groupby("unit")["cycle"].max().min()),
            "max_cycles_per_engine": int(df.groupby("unit")["cycle"].max().max())}


validation_summary = pd.DataFrame([validate_frame(train_raw, "train"), validate_frame(test_raw, "test")])
display(validation_summary)

assert validation_summary["missing_values"].eq(0).all(), "Unexpected missing values"
assert validation_summary["duplicate_unit_cycle"].eq(0).all(), "Duplicate (unit, cycle) rows"
assert validation_summary["cycles_contiguous_from_1"].all(), "Cycle numbering has gaps"
assert test_raw["unit"].nunique() == len(rul_official), "Test engines and RUL file are not aligned"
assert set(test_raw["unit"]) == set(rul_official["unit"])
print("dtypes:", dict(train_raw.dtypes.value_counts()))
""")

md(r"""
### 3.4 Constant and near-constant columns (full training file, for EDA only)
This table is *descriptive*. The sensor list actually used for modelling is re-derived later from the
**training-split engines only** (Section 6) so that validation engines never influence feature selection.
""")

code(r"""
def column_variability(df, cols):
    stats = pd.DataFrame({
        "mean": df[cols].mean(), "std": df[cols].std(), "min": df[cols].min(), "max": df[cols].max(),
        "n_unique": df[cols].nunique()})
    stats["rel_std"] = stats["std"] / stats["mean"].abs().replace(0, np.nan)
    stats["near_constant"] = (stats["n_unique"] < CONFIG["min_unique_values"]) | (stats["rel_std"].fillna(0) < CONFIG["min_rel_std"])
    return stats


variability = column_variability(train_raw, SETTING_COLS + SENSOR_COLS)
display(variability.round(5))
print("Constant / near-constant:", variability.index[variability["near_constant"]].tolist())
""")

md(r"""
### 3.5 Data dictionary
Sensor descriptions follow Saxena et al. (2008), *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*.
""")

code(r"""
SENSOR_INFO = {
    "s1": ("T2", "Total temperature at fan inlet", "°R"),
    "s2": ("T24", "Total temperature at LPC outlet", "°R"),
    "s3": ("T30", "Total temperature at HPC outlet", "°R"),
    "s4": ("T50", "Total temperature at LPT outlet", "°R"),
    "s5": ("P2", "Pressure at fan inlet", "psia"),
    "s6": ("P15", "Total pressure in bypass-duct", "psia"),
    "s7": ("P30", "Total pressure at HPC outlet", "psia"),
    "s8": ("Nf", "Physical fan speed", "rpm"),
    "s9": ("Nc", "Physical core speed", "rpm"),
    "s10": ("epr", "Engine pressure ratio (P50/P2)", "–"),
    "s11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "s12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "s13": ("NRf", "Corrected fan speed", "rpm"),
    "s14": ("NRc", "Corrected core speed", "rpm"),
    "s15": ("BPR", "Bypass ratio", "–"),
    "s16": ("farB", "Burner fuel-air ratio", "–"),
    "s17": ("htBleed", "Bleed enthalpy", "–"),
    "s18": ("Nf_dmd", "Demanded fan speed", "rpm"),
    "s19": ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    "s20": ("W31", "HPT coolant bleed", "lbm/s"),
    "s21": ("W32", "LPT coolant bleed", "lbm/s"),
}

rows = [("unit", "Engine identifier", "id", "–"), ("cycle", "Operating cycle (time index)", "time", "cycles"),
        ("setting_1", "Operational setting 1 (altitude-related)", "setting", "–"),
        ("setting_2", "Operational setting 2 (Mach-related)", "setting", "–"),
        ("setting_3", "Operational setting 3 (throttle resolver angle)", "setting", "–")]
rows += [(s, f"{SENSOR_INFO[s][0]} — {SENSOR_INFO[s][1]}", "sensor", SENSOR_INFO[s][2]) for s in SENSOR_COLS]
data_dictionary = pd.DataFrame(rows, columns=["column", "description", "role", "unit_of_measure"]).set_index("column")
data_dictionary = data_dictionary.join(variability[["mean", "std", "n_unique", "near_constant"]])
data_dictionary.to_csv(METRIC_DIR / "data_dictionary.csv")
display(data_dictionary.round(4))
""")

# =============================================================================
# 4. EDA
# =============================================================================
md(r"""
## 4. Exploratory Data Analysis

For EDA we temporarily attach the **uncapped** RUL (`max cycle − current cycle`) to the training data.
The formal target construction (with checks and capping) is in Section 5.
""")

code(r"""
eda = train_raw.copy()
eda["rul"] = eda.groupby("unit")["cycle"].transform("max") - eda["cycle"]
lifetimes = train_raw.groupby("unit")["cycle"].max().rename("lifetime")
test_lengths = test_raw.groupby("unit")["cycle"].max().rename("observed_cycles")
print(lifetimes.describe().round(1).to_string())
""")

md("### 4.1 Cycles per engine and lifetime distribution")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
order = lifetimes.sort_values()
axes[0].bar(range(len(order)), order.values, color="#2a78d6", width=0.8)
axes[0].set(title="Training: cycles to failure per engine (sorted)", xlabel="Engine (sorted by lifetime)", ylabel="Cycles")
axes[0].axhline(order.mean(), color="#52514e", ls="--", lw=1)
axes[0].text(2, order.mean() + 5, f"mean = {order.mean():.0f}", color="#52514e")

bins = np.arange(0, 380, 20)
axes[1].hist(lifetimes, bins=bins, color="#2a78d6", alpha=0.85, label="Train: full life (to failure)", edgecolor="white")
axes[1].hist(test_lengths, bins=bins, color="#eb6834", alpha=0.6, label="Test: observed (truncated) history", edgecolor="white")
axes[1].set(title="Engine lifetime vs. test history length", xlabel="Cycles", ylabel="Engines")
axes[1].legend()
save_fig(fig, "eda_lifetimes"); plt.show()
print(f"Train lifetimes: min={lifetimes.min()}, median={lifetimes.median():.0f}, max={lifetimes.max()}")
print(f"Test histories : min={test_lengths.min()}, median={test_lengths.median():.0f}, max={test_lengths.max()}")
""")

md(r"""
**Interpretation.** Training engines run to failure after 128–362 cycles (heavily right-skewed: most fail
around 180–230 cycles, a few survive much longer). Test engines are cut off *before* failure, so their
observed histories are much shorter (as few as 31 cycles). A model must therefore work with short
histories — this matters for rolling-window features in the first cycles.
""")

md("### 4.2 RUL distribution (uncapped vs. capped)")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(eda["rul"], bins=60, color="#2a78d6", edgecolor="white")
axes[0].set(title="Training RUL (uncapped)", xlabel="RUL (cycles)", ylabel="Rows")
cap = CONFIG["rul_cap"]
axes[1].hist(eda["rul"].clip(upper=cap) if cap else eda["rul"], bins=60, color="#2a78d6", edgecolor="white")
axes[1].set(title=f"Training RUL capped at {cap}", xlabel="RUL (cycles)", ylabel="Rows")
axes[1].axvline(cap, color="#e34948", ls="--", lw=1)
save_fig(fig, "eda_rul_distribution"); plt.show()
print(f"Share of training rows with RUL > {cap}: {(eda['rul'] > cap).mean():.1%}")
""")

md(r"""
**Interpretation.** Uncapped RUL is roughly uniform from 0 up to ~200 and then tails off, because every engine
contributes one row per remaining cycle. Capping turns the long-RUL tail into a single "healthy" plateau
(the spike at the cap) — see Section 5 for why this is justified.
""")

md("### 4.3 Sensor variability and constant-sensor analysis")

code(r"""
cv_table = variability.loc[SENSOR_COLS, ["std", "rel_std", "n_unique"]].copy()
fig, ax = plt.subplots(figsize=(13, 4))
vals = cv_table["rel_std"].fillna(0).clip(lower=1e-7)
colors = ["#8a8984" if variability.loc[s, "near_constant"] else "#2a78d6" for s in SENSOR_COLS]
ax.bar(SENSOR_COLS, vals, color=colors)
ax.set_yscale("log")
ax.set(title="Relative variability (std / |mean|) per sensor — grey = constant / near-constant",
       ylabel="std / |mean|  (log scale)")
for i, s in enumerate(SENSOR_COLS):
    ax.text(i, vals[s] * 1.3, int(cv_table.loc[s, "n_unique"]), ha="center", fontsize=8, color="#52514e")
ax.text(0.99, 0.95, "numbers = distinct values", transform=ax.transAxes, ha="right", fontsize=9, color="#52514e")
save_fig(fig, "eda_sensor_variability"); plt.show()
""")

md(r"""
**Interpretation.** Sensors **s1, s5, s10, s16, s18, s19** take a single value throughout FD001 (zero information),
and **s6** takes only two values, so it is essentially a quantised constant. `setting_3` is constant (100) as well.
The remaining 14 sensors vary and are candidates for modelling.
""")

md("### 4.4 Operating settings — is operating-condition normalisation needed for FD001?")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(train_raw["setting_1"], train_raw["setting_2"], s=3, alpha=0.3, color="#2a78d6")
axes[0].set(title="Operating settings 1 vs 2 (all training rows)", xlabel="setting_1", ylabel="setting_2")
informative_sensors_eda = [s for s in SENSOR_COLS if not variability.loc[s, "near_constant"]]
corr_settings = train_raw[["setting_1", "setting_2"] + informative_sensors_eda].corr().loc[["setting_1", "setting_2"], informative_sensors_eda]
sns.heatmap(corr_settings, ax=axes[1], cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7}, cbar=False)
axes[1].set_title("Correlation of settings with sensors")
save_fig(fig, "eda_operating_settings"); plt.show()
print("setting_1 range:", train_raw["setting_1"].min(), "to", train_raw["setting_1"].max())
print("setting_2 range:", train_raw["setting_2"].min(), "to", train_raw["setting_2"].max())
print("setting_3 unique:", train_raw["setting_3"].unique())
print("max |corr(setting, sensor)| =", corr_settings.abs().values.max().round(3))
""")

md(r"""
**Interpretation.** In FD001 the settings form a single tight cluster (tiny variations around 0 and a constant
`setting_3 = 100`) and are essentially uncorrelated with every sensor. This confirms FD001 has **one operating
condition**, so **per-condition (regime) normalisation is unnecessary** — a single global standardisation
is enough. The settings are dropped from the feature set because they carry no degradation signal
(this would *not* be true for FD002/FD004, which have six operating conditions).
""")

md("### 4.5 Sensor trajectories for selected engines")

code(r"""
example_units = lifetimes.sort_values().index[[0, 33, 66, 99]].tolist()   # short, medium, long-lived engines
unit_palette = dict(zip(example_units, ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]))
fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
for ax, s in zip(axes.flat, informative_sensors_eda):
    for u in example_units:
        d = eda[eda["unit"] == u]
        ax.plot(d["rul"], d[s], color=unit_palette[u], lw=1, alpha=0.85, label=f"engine {u}")
    ax.set_title(f"{s} ({SENSOR_INFO[s][0]})", fontsize=10)
for ax in axes.flat[len(informative_sensors_eda):]:
    ax.axis("off")
for ax in axes[-1]:
    ax.set_xlabel("RUL (cycles; failure at 0)")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.08), fontsize=11)
fig.suptitle("Raw sensor trajectories vs. RUL for four engines (short → long lifetime)", y=1.0)
fig.tight_layout()
save_fig(fig, "eda_sensor_trajectories"); plt.show()
""")

md(r"""
**Interpretation.** Aligning the engines on RUL (failure at RUL = 0, on the left) shows a common
degradation pattern: most sensors are flat and noisy early in life and then drift monotonically during the last
~100 cycles. Approaching failure, temperatures (s2, s3, s4), static pressure s11, fan speeds s8/s13, bypass ratio
s15 and bleed enthalpy s17 *rise*, while HPC outlet pressure s7, fuel-flow ratio s12 and coolant bleeds s20/s21
*fall*. s8 and s13 vary over a tiny absolute range (≈ 0.3 rpm around 2388) but still trend clearly. s9 (core speed) and s14 (corrected core
speed) drift too, but their level differs strongly between engines (engine-to-engine offset), which makes their
raw values less reliable across the fleet. The high per-cycle noise motivates rolling-window smoothing.
""")

md("### 4.6 Correlation heatmap of informative sensors and RUL")

code(r"""
corr = eda[informative_sensors_eda + ["rul"]].corr()
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
            annot_kws={"size": 7}, square=True, cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title("Pearson correlation — informative sensors and uncapped RUL")
save_fig(fig, "eda_correlation_heatmap"); plt.show()
rul_corr = corr["rul"].drop("rul").sort_values(key=np.abs, ascending=False)
display(rul_corr.round(3).to_frame("corr_with_RUL"))
""")

md(r"""
**Interpretation.** Many sensors are strongly inter-correlated (they all respond to the same HPC degradation),
e.g. s11/s4/s12/s7 and s9/s14. Strongest linear relationships with RUL: s11, s4, s12, s7, s21, s20, s15
(|r| ≈ 0.63–0.70). s9, s14 and s6 show weaker correlation. Because of the redundancy, a regularised linear
model (Elastic Net) and tree ensembles (which tolerate correlated inputs) are reasonable choices.
""")

md("### 4.7 Relationship between sensor values and RUL")

code(r"""
top6 = rul_corr.index[:6].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
sample = eda.sample(6000, random_state=SEED)
for ax, s in zip(axes.flat, top6):
    ax.scatter(sample["rul"], sample[s], s=3, alpha=0.25, color="#2a78d6")
    # median trend per RUL bin
    trend = eda.groupby(pd.cut(eda["rul"], bins=np.arange(0, 370, 10)), observed=True)[s].median()
    ax.plot([iv.mid for iv in trend.index], trend.values, color="#e34948", lw=2, label="median per 10-cycle bin")
    ax.set(title=f"{s} ({SENSOR_INFO[s][0]}), r = {rul_corr[s]:.2f}", xlabel="RUL (uncapped)", ylabel=s)
    ax.axvline(CONFIG["rul_cap"], color="#52514e", ls=":", lw=1)
axes.flat[0].legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "eda_sensor_vs_rul"); plt.show()
""")

md(r"""
**Interpretation.** The relationship is clearly **non-linear**: the median curve is almost flat for large RUL
(roughly > 125 cycles) and bends sharply as the engine approaches failure. Sensor readings carry little
information about *how far* a healthy engine is from failure — which is exactly the motivation for piece-wise
(capped) RUL targets and for non-linear models.
""")

md("### 4.8 Early vs. late degradation — sensor behaviour")

code(r"""
def life_stage(rul):
    return np.where(rul > CONFIG["stage_threshold"], "early", "late")

eda["stage"] = life_stage(eda["rul"])
z = eda[informative_sensors_eda].sub(eda[informative_sensors_eda].mean()).div(eda[informative_sensors_eda].std())
z["stage"] = eda["stage"]
stage_long = z.melt(id_vars="stage", var_name="sensor", value_name="z")
fig, ax = plt.subplots(figsize=(15, 4.5))
sns.boxplot(data=stage_long, x="sensor", y="z", hue="stage", palette=STAGE_COLORS, fliersize=0, ax=ax, width=0.7)
ax.set(title=f"Standardised sensor values: early life (RUL > {CONFIG['stage_threshold']}) vs late life (RUL ≤ {CONFIG['stage_threshold']})",
       ylabel="z-score", ylim=(-4, 4))
save_fig(fig, "eda_early_vs_late"); plt.show()
shift = z.groupby("stage")[informative_sensors_eda].mean().T
shift["late_minus_early"] = shift["late"] - shift["early"]
display(shift.sort_values("late_minus_early", key=np.abs, ascending=False).round(2))
""")

md(r"""
**Interpretation.** For the informative sensors the late-life distribution is shifted by roughly one standard
deviation relative to early life, but the boxes still overlap — a single reading is ambiguous, while a *history*
of readings (trend) is far more discriminative. s6 shows practically no shift, and s9/s14 shift less relative
to their spread, matching the correlation analysis.
""")

# =============================================================================
# 5. RUL construction
# =============================================================================
md(r"""
## 5. RUL Target Engineering

For every training engine $u$ with final observed cycle $T_u$ (the failure cycle):

$$\text{RUL}_{u,t} = T_u - t$$

so an engine that fails at cycle 200 has RUL = 50 at cycle 150 and RUL = 0 at cycle 200.

**Capping (piece-wise linear target).** Early in life the sensors show no degradation (Section 4.7), so it is
impossible to tell an engine with 300 cycles left from one with 150 left. Following common practice for C-MAPSS,
the target is capped: $\text{RUL}^{cap}_{u,t} = \min(\text{RUL}_{u,t},\ C)$ with $C$ = `CONFIG["rul_cap"]` (125).
This says "the engine is healthy" until degradation begins. The **same capped target is used for every model**, for
validation, and for the official test labels. The cap can be disabled by setting `rul_cap=None`.

The official test RUL file is **never** used to build training targets.
""")

code(r"""
def add_rul(df: pd.DataFrame, cap=None) -> pd.DataFrame:
    # RUL computed independently inside each engine
    out = df.sort_values(INDEX_COLS).copy()
    out["rul_raw"] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    out["rul"] = out["rul_raw"].clip(upper=cap) if cap is not None else out["rul_raw"]
    return out


train_df = add_rul(train_raw, CONFIG["rul_cap"])

# --- checks -------------------------------------------------------------------
last_rows = train_df.groupby("unit").tail(1)
assert (last_rows["rul_raw"] == 0).all(), "RUL must be 0 at each engine's final cycle"
assert (train_df.groupby("unit")["rul_raw"].diff().dropna() == -1).all(), "RUL must decrease by 1 per cycle"
assert (train_df["rul_raw"] >= 0).all()
if CONFIG["rul_cap"] is not None:
    assert train_df["rul"].max() == CONFIG["rul_cap"]
example = train_df[train_df["unit"] == 1][["unit", "cycle", "rul_raw", "rul"]]
display(pd.concat([example.head(3), example.iloc[[100, 150]], example.tail(3)]))
print("All RUL checks passed. Rows:", len(train_df), "| engines:", train_df['unit'].nunique())
""")

# =============================================================================
# 6-7. Preprocessing and split
# =============================================================================
md(r"""
## 6. Preprocessing and 7. Engine-Level Train/Validation Split

**Why split by engine?** Consecutive rows of the same engine are almost identical. If rows were split randomly,
the validation set would contain near-copies of training rows (cycle 120 in train, cycle 121 in validation) and
the score would be optimistically biased. Splitting by **engine ID** mimics the real use case — predicting for
engines the model has never seen.

Order of operations (to prevent information leakage):
1. Sort by engine and cycle.
2. Split **engines** into training (80%) and validation (20%) with a fixed seed.
3. Screen constant / near-constant sensors using **training engines only**.
4. Standardisation is placed **inside each model pipeline**, so it is fitted on training rows only
   (and, during cross-validation, only on the training folds).
""")

code(r"""
splitter = GroupShuffleSplit(n_splits=1, test_size=CONFIG["val_fraction"], random_state=SEED)
tr_idx, va_idx = next(splitter.split(train_df, groups=train_df["unit"]))
train_units = np.sort(train_df.iloc[tr_idx]["unit"].unique())
val_units = np.sort(train_df.iloc[va_idx]["unit"].unique())

# --- engine-level split checks ---------------------------------------------
assert set(train_units).isdisjoint(val_units), "An engine appears in both training and validation!"
assert len(train_units) + len(val_units) == train_df["unit"].nunique()
print(f"Training engines: {len(train_units)} | Validation engines: {len(val_units)}")
print("Validation engine IDs:", val_units.tolist())

pd.DataFrame({"unit": np.concatenate([train_units, val_units]),
              "split": ["train"] * len(train_units) + ["validation"] * len(val_units)}).to_csv(
    METRIC_DIR / "engine_split.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.hist([lifetimes.loc[train_units], lifetimes.loc[val_units]], bins=np.arange(120, 380, 20),
        color=["#2a78d6", "#eb6834"], label=["train engines", "validation engines"], edgecolor="white")
ax.set(title="Lifetime distribution of training vs validation engines", xlabel="Lifetime (cycles)", ylabel="Engines")
ax.legend(); save_fig(fig, "split_lifetimes"); plt.show()
""")

code(r"""
def select_sensors(df: pd.DataFrame) -> list:
    # Keep sensors that actually vary in the given (training) data.
    stats = column_variability(df, SENSOR_COLS)
    return stats.index[~stats["near_constant"]].tolist()


train_part = train_df[train_df["unit"].isin(train_units)]
SELECTED_SENSORS = select_sensors(train_part)          # decided on training engines only
DROPPED_SENSORS = [s for s in SENSOR_COLS if s not in SELECTED_SENSORS]
print(f"Selected {len(SELECTED_SENSORS)} sensors:", SELECTED_SENSORS)
print("Dropped (constant / near-constant on training engines):", DROPPED_SENSORS)
print("Operating settings dropped: single operating condition in FD001 (Section 4.4).")
""")

# =============================================================================
# 8. Feature engineering
# =============================================================================
md(r"""
## 8. Feature Engineering

All features are computed by **one function** applied engine-by-engine (`groupby("unit")`), using only the
**current and past** cycles. The same function is later applied to the official test histories.

| Group | Features (per selected sensor *s*) | Idea |
|---|---|---|
| A. Basic | `cycle`, raw value `s` | current state; age of the engine |
| B. Rolling statistics | `s_mean_w`, `s_std_w` for w ∈ {5, 10, 20} | smooth out sensor noise; capture volatility |
| C. Rolling slope | `s_slope_w` for w ∈ {10, 20} | local *rate of change* (trend) |
| D. Health indicators | `s_dev_base`, `s_diff_10` | drift from the engine's own healthy baseline; change over the last 10 cycles |

**Rolling slope.** Within the last *w* cycles we fit an ordinary least-squares line $x = a + b\,t$ and use the
slope $b = \frac{\overline{tx} - \bar t\,\bar x}{\overline{t^2} - \bar t^2}$, computed with rolling means (exact
OLS, vectorised). Its sign is **sensor-specific**: for temperatures (s2, s3, s4) or Ps30 (s11) a *positive*
slope means the engine is getting hotter / pressure building — degradation; for s7 (P30), s12 (phi), s20/s21
(coolant bleed) a *negative* slope indicates degradation. A slope near zero means the sensor is stable (healthy
or noise only). The models learn the direction automatically.

**Healthy baseline.** Each engine has its own manufacturing offset (visible for s9/s14 in Section 4.5). The
baseline is the mean of the engine's first `baseline_cycles` (10) readings; `s_dev_base = s_mean_10 − baseline`
measures drift relative to *that engine's* healthy state. For cycles < 10 the baseline uses only the cycles
observed so far (expanding mean), so no future data are used.

**First cycles of an engine.** Windows are *partial* (`min_periods=1`) at the start of each engine: the rolling
mean over 5 cycles at cycle 2 is the mean of cycles 1–2. A rolling std or slope needs ≥ 2 points; at cycle 1 they
are set to 0 ("no observed variation / trend yet"). This lets us predict even for test engines with short histories.

**Not used:** normalised lifecycle progress (`cycle / total_life`) — total life is unknown at prediction time,
so this would be target leakage. Operating settings — uninformative in FD001.
""")

code(r"""
def _rolling(grouped, col, window, func):
    r = grouped[col].rolling(window, min_periods=1)
    return getattr(r, func)().reset_index(level=0, drop=True)


def build_features(df: pd.DataFrame, sensors: list, cfg: dict = CONFIG) -> pd.DataFrame:
    # Build degradation features engine-by-engine using past + current cycles only.
    df = df.sort_values(INDEX_COLS).reset_index(drop=True)
    g = df.groupby("unit", sort=False)
    feats = {"unit": df["unit"], "cycle": df["cycle"]}
    t = df["cycle"].astype(float)
    tmp = pd.DataFrame({"unit": df["unit"], "t": t, "t2": t * t})
    gt = tmp.groupby("unit", sort=False)

    for s in sensors:
        x = df[s].astype(float)
        feats[s] = x                                                   # A. raw value
        for w in cfg["roll_windows"]:                                  # B. rolling mean / std
            feats[f"{s}_mean_{w}"] = _rolling(g, s, w, "mean")
            feats[f"{s}_std_{w}"] = _rolling(g, s, w, "std").fillna(0.0)  # 1 point -> std 0
        tmp["x"], tmp["tx"] = x, t * x
        gt = tmp.groupby("unit", sort=False)
        for w in cfg["slope_windows"]:                                 # C. rolling OLS slope
            m_t, m_x = _rolling(gt, "t", w, "mean"), _rolling(gt, "x", w, "mean")
            m_tx, m_t2 = _rolling(gt, "tx", w, "mean"), _rolling(gt, "t2", w, "mean")
            var_t = m_t2 - m_t ** 2
            slope = (m_tx - m_t * m_x) / var_t.where(var_t > 1e-12)
            feats[f"{s}_slope_{w}"] = slope.fillna(0.0)                # 1 point -> slope 0
        # D. health indicators
        n = cfg["baseline_cycles"]
        cum_sum = g[s].cumsum()
        k = g.cumcount() + 1                                           # number of cycles seen so far
        first_n_sum = cum_sum.where(k <= n).groupby(df["unit"]).ffill()  # sum of first min(k, n) readings
        baseline = first_n_sum / np.minimum(k, n)
        smooth = feats[f"{s}_mean_{10 if 10 in cfg['roll_windows'] else cfg['roll_windows'][0]}"]
        feats[f"{s}_dev_base"] = smooth - baseline
        lag = cfg["diff_lag"]
        earlier = g[s].shift(lag).fillna(g[s].transform("first"))      # earliest reading if < lag cycles
        feats[f"{s}_diff_{lag}"] = x - earlier

    out = pd.DataFrame(feats)
    for extra in ["rul", "rul_raw"]:
        if extra in df.columns:
            out[extra] = df[extra].values
    return out


FEATURE_GROUPS = {
    "A_basic": lambda c: c == "cycle" or c in SELECTED_SENSORS,
    "B_rolling_mean": lambda c: "_mean_" in c,
    "B_rolling_std": lambda c: "_std_" in c,
    "C_slope": lambda c: "_slope_" in c,
    "D_health": lambda c: c.endswith("_dev_base") or "_diff_" in c,
}

t0 = time.time()
feat_all = build_features(train_df, SELECTED_SENSORS)
FEATURE_COLS = [c for c in feat_all.columns if c not in ("unit", "rul", "rul_raw")]
print(f"Built {len(FEATURE_COLS)} features for {len(feat_all)} rows in {time.time() - t0:.1f}s")
for gname, rule in FEATURE_GROUPS.items():
    print(f"  {gname:15s}: {sum(rule(c) for c in FEATURE_COLS)} features")
assert sum(any(rule(c) for rule in FEATURE_GROUPS.values()) for c in FEATURE_COLS) == len(FEATURE_COLS)
""")

md("### 8.1 Feature dictionary")

code(r"""
def describe_feature(c):
    if c == "cycle":
        return "A_basic", "Current operating cycle (engine age so far)", "Older engines tend to be closer to failure"
    base = c.split("_")[0]
    name = f"{base} ({SENSOR_INFO[base][0]})"
    if c in SELECTED_SENSORS:
        return "A_basic", f"Current raw reading of {name}", "Instantaneous health state (noisy)"
    kind, w = c.split("_")[1], c.split("_")[-1]
    if kind == "mean":
        return "B_rolling_mean", f"Mean of {name} over last {w} cycles", "Noise-reduced sensor level"
    if kind == "std":
        return "B_rolling_std", f"Std. dev. of {name} over last {w} cycles", "Volatility; may rise as degradation accelerates"
    if kind == "slope":
        return "C_slope", f"OLS slope of {name} over last {w} cycles (units/cycle)", "Rate of degradation (trend direction & speed)"
    if kind == "dev":
        return "D_health", f"{name} 10-cycle mean minus engine's healthy baseline (first {CONFIG['baseline_cycles']} cycles)", "Cumulative drift from this engine's own healthy state"
    if kind == "diff":
        return "D_health", f"{name} now minus {CONFIG['diff_lag']} cycles earlier", "Short-term change in sensor level"
    raise ValueError(c)


feature_dictionary = pd.DataFrame([(c, *describe_feature(c)) for c in FEATURE_COLS],
                                  columns=["feature", "group", "definition", "why_it_may_help"]).set_index("feature")
feature_dictionary.to_csv(METRIC_DIR / "feature_dictionary.csv")
display(feature_dictionary.head(16))
print("Full dictionary saved to", METRIC_DIR / "feature_dictionary.csv")
""")

code(r"""
# Visualise engineered features for one engine and one sensor
u, s = example_units[2], "s11"
d = feat_all[feat_all["unit"] == u]
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(d["cycle"], d[s], color="#8a8984", lw=1, label="raw")
for w, col in zip(CONFIG["roll_windows"], ["#2a78d6", "#eb6834", "#1baf7a"]):
    axes[0].plot(d["cycle"], d[f"{s}_mean_{w}"], color=col, label=f"rolling mean {w}")
axes[0].set(title=f"Engine {u}: {s} raw vs rolling means", xlabel="cycle"); axes[0].legend(fontsize=8)
for w, col in zip(CONFIG["slope_windows"], ["#2a78d6", "#eb6834"]):
    axes[1].plot(d["cycle"], d[f"{s}_slope_{w}"], color=col, label=f"slope {w}")
axes[1].axhline(0, color="#52514e", lw=0.8)
axes[1].set(title=f"Rolling OLS slope of {s}", xlabel="cycle"); axes[1].legend(fontsize=8)
axes[2].plot(d["cycle"], d[f"{s}_dev_base"], color="#4a3aa7", label="deviation from baseline")
axes[2].plot(d["cycle"], d[f"{s}_diff_{CONFIG['diff_lag']}"], color="#8a8984", lw=1, label=f"diff vs {CONFIG['diff_lag']} cycles ago")
axes[2].axhline(0, color="#52514e", lw=0.8)
axes[2].set(title="Health indicators", xlabel="cycle"); axes[2].legend(fontsize=8)
fig.tight_layout(); save_fig(fig, "features_example"); plt.show()
""")

md(r"""
**Interpretation.** The rolling means remove most of the cycle-to-cycle noise and reveal the upward drift in
Ps30 (s11) late in life. The 20-cycle slope is close to zero for most of the life and turns clearly positive as
the engine degrades; the 10-cycle slope reacts faster but is noisier. The deviation-from-baseline indicator rises
smoothly, while the 10-cycle difference is noisy because it subtracts two single raw readings.
""")

# =============================================================================
# 9. Feature validation & leakage checks
# =============================================================================
md(r"""
## 9. Feature Validation and Leakage Checks

1. **Engine boundaries** — features computed on the full table equal features computed on each engine separately.
2. **No future information** — truncating an engine's history at cycle *k* and recomputing features gives
   *identical* values for cycles ≤ *k*. If any feature looked ahead, this would fail.
3. **First-cycle handling** — no NaN/inf values; at cycle 1 std and slope are 0.
4. **Target leakage** — `rul`/`rul_raw` are not among the features; no feature is (almost) perfectly correlated with RUL.
5. **Dimensions** — identical column order for train/validation/test.
""")

code(r"""
# 1. engine-boundary check
per_engine = pd.concat([build_features(train_df[train_df["unit"] == u], SELECTED_SENSORS)
                        for u in train_df["unit"].unique()[:15]], ignore_index=True)
joint = feat_all[feat_all["unit"].isin(train_df["unit"].unique()[:15])].reset_index(drop=True)
pd.testing.assert_frame_equal(per_engine[FEATURE_COLS], joint[FEATURE_COLS], check_exact=False, rtol=1e-9)
print("✔ 1. Rolling windows never cross engine boundaries")

# 2. no-future-information check on several engines and truncation points
rng = np.random.default_rng(SEED)
for u in rng.choice(train_df["unit"].unique(), 10, replace=False):
    full = feat_all[feat_all["unit"] == u].reset_index(drop=True)
    k = int(rng.integers(5, len(full) - 1))
    trunc = build_features(train_df[(train_df["unit"] == u) & (train_df["cycle"] <= k)], SELECTED_SENSORS)
    pd.testing.assert_frame_equal(trunc[FEATURE_COLS], full.loc[: k - 1, FEATURE_COLS], check_exact=False, rtol=1e-9)
print("✔ 2. Features use only past and current cycles (truncation-invariant)")

# 3. first cycles
assert np.isfinite(feat_all[FEATURE_COLS].values).all(), "NaN/inf in features"
first = feat_all[feat_all["cycle"] == 1]
assert (first.filter(like="_std_") == 0).all().all() and (first.filter(like="_slope_") == 0).all().all()
assert (first.filter(like="_dev_base") .abs() < 1e-9).all().all()
print("✔ 3. No NaN/inf; first-cycle std/slope/deviation = 0")

# 4. target leakage
assert not {"rul", "rul_raw", "unit"} & set(FEATURE_COLS)
max_abs_corr = feat_all[FEATURE_COLS].corrwith(feat_all["rul"]).abs().max()
assert max_abs_corr < 0.99, "A feature is suspiciously correlated with the target"
print(f"✔ 4. Target/ID columns excluded; max |corr(feature, capped RUL)| = {max_abs_corr:.3f}")
""")

code(r"""
X_COLS = FEATURE_COLS
train_mask = feat_all["unit"].isin(train_units)
val_mask = feat_all["unit"].isin(val_units)
X_train, y_train, g_train = feat_all.loc[train_mask, X_COLS], feat_all.loc[train_mask, "rul"], feat_all.loc[train_mask, "unit"]
X_val, y_val = feat_all.loc[val_mask, X_COLS], feat_all.loc[val_mask, "rul"]
val_meta = feat_all.loc[val_mask, ["unit", "cycle", "rul", "rul_raw"]].reset_index(drop=True)

assert X_train.shape[1] == X_val.shape[1] == len(X_COLS)
assert set(g_train).isdisjoint(val_meta["unit"])
print(f"✔ 5. X_train {X_train.shape}  X_val {X_val.shape}  (features: {len(X_COLS)})")

# Redundancy check: how many feature pairs are almost duplicates?
c = X_train.corr().abs().values
n_dup = int(((c > 0.98) & ~np.eye(len(c), dtype=bool)).sum() / 2)
print(f"Feature pairs with |r| > 0.98: {n_dup} (mostly rolling means of the same sensor; "
      "handled by tree ensembles and Elastic Net's L1/L2 penalty)")
""")

# =============================================================================
# Evaluation helpers
# =============================================================================
md(r"""
### Evaluation utilities
* **RMSE** — square-root of mean squared error; penalises large errors more (same units as RUL: cycles).
* **MAE** — mean absolute error; average miss in cycles.
* **R²** — share of target variance explained (1 = perfect, 0 = as good as predicting the mean).
* **NASA score** (additional metric, PHM08) — asymmetric exponential penalty with $d = \hat y - y$:
  $s = \sum (e^{-d/13} - 1)$ if $d<0$ (early prediction) and $\sum (e^{d/10} - 1)$ if $d \ge 0$ (late
  prediction). Over-estimating RUL (predicting failure too late) is penalised more because it is more dangerous.
  Lower is better; it is a *sum*, so it is only comparable on the same set of engines.
""")

code(r"""
def nasa_score(y_true, y_pred):
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def regression_metrics(y_true, y_pred, nasa=False):
    out = {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
           "MAE": float(mean_absolute_error(y_true, y_pred)),
           "R2": float(r2_score(y_true, y_pred))}
    if nasa:
        out["NASA_score"] = nasa_score(y_true, y_pred)
    return out


def clip_pred(p):
    # RUL can't be negative and (with capping) can't exceed the cap
    return np.clip(p, 0, CONFIG["rul_cap"] if CONFIG["rul_cap"] else None)


def last_cycle_rows(meta):
    return meta.groupby("unit")["cycle"].transform("max") == meta["cycle"]


# Test-like validation view: one random truncation point per validation engine (mimics the official test set)
rng_tr = np.random.default_rng(SEED)
val_trunc_idx = []
for u, d in val_meta.groupby("unit"):
    k = rng_tr.integers(int(0.25 * len(d)), len(d))      # cut somewhere between 25% and 100% of life
    val_trunc_idx.append(d.index[k])
val_trunc_idx = np.array(val_trunc_idx)
print(f"Test-like validation points: {len(val_trunc_idx)} (one per validation engine)")

VAL_RESULTS, VAL_PREDS, FITTED = {}, {}, {}


def record(name, model, pred_val, fit_seconds, best_params=None):
    pred_val = clip_pred(pred_val)
    VAL_PREDS[name] = pred_val
    FITTED[name] = model
    m_all = regression_metrics(y_val, pred_val)
    m_tl = regression_metrics(val_meta.loc[val_trunc_idx, "rul"], pred_val[val_trunc_idx], nasa=True)
    VAL_RESULTS[name] = {**{f"val_{k}": v for k, v in m_all.items()},
                         **{f"val_testlike_{k}": v for k, v in m_tl.items()},
                         "fit_seconds": fit_seconds, "best_params": json.dumps(best_params or {}, default=str)}
    print(f"{name:22s} | all-cycle val RMSE {m_all['RMSE']:6.2f}  MAE {m_all['MAE']:6.2f}  R² {m_all['R2']:.3f}"
          f" | test-like RMSE {m_tl['RMSE']:6.2f}")
""")

# =============================================================================
# 10. Baseline
# =============================================================================
md(r"""
## 10. Baseline Models

Two deliberately simple references:
* **Baseline (mean):** always predicts the mean training RUL — any useful model must beat this.
* **Baseline (cycle-only):** linear regression on the current cycle only (age-based maintenance rule of thumb).
""")

code(r"""
t0 = time.time()
base_mean = DummyRegressor(strategy="mean").fit(X_train, y_train)
record("Baseline (mean)", base_mean, base_mean.predict(X_val), time.time() - t0)

t0 = time.time()
base_cycle = LinearRegression().fit(X_train[["cycle"]], y_train)
record("Baseline (cycle-only)", base_cycle, base_cycle.predict(X_val[["cycle"]]), time.time() - t0,
       {"coef_cycle": round(float(base_cycle.coef_[0]), 4), "intercept": round(float(base_cycle.intercept_), 2)})
""")

md(r"""
## Hyper-parameter tuning protocol (same for all models)

* Search space sampled with `RandomizedSearchCV` (`n_iter` = 12, fixed seed).
* Cross-validation: **`GroupKFold` (3 folds) over the 80 training engines** — engines never straddle folds.
* Scoring: RMSE. The validation engines are **not** touched during tuning; they are used once to compare the tuned models.
* The scaler sits inside the pipeline, so it is re-fitted on each training fold only.
""")

code(r"""
CV = GroupKFold(n_splits=CONFIG["cv_folds"])


def tune(name, pipeline, param_space):
    search = RandomizedSearchCV(pipeline, param_space, n_iter=CONFIG["n_iter_search"], cv=CV,
                                scoring="neg_root_mean_squared_error", random_state=SEED,
                                n_jobs=CONFIG["n_jobs"], refit=True, return_train_score=False)
    t0 = time.time()
    search.fit(X_train, y_train, groups=g_train)
    elapsed = time.time() - t0
    best = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
    print(f"{name}: best CV RMSE {-search.best_score_:.2f} | search time {elapsed:.0f}s | best params {best}")
    cvres = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    cvres["mean_test_RMSE"] = -cvres.pop("mean_test_score")
    cvres.sort_values("rank_test_score").to_csv(METRIC_DIR / f"cv_results_{name.replace(' ', '_').lower()}.csv", index=False)
    record(name, search.best_estimator_, search.best_estimator_.predict(X_val).astype(float), elapsed, best)
    VAL_RESULTS[name]["cv_RMSE"] = -search.best_score_
    return search
""")

# =============================================================================
# 11. RF
# =============================================================================
md(r"""
## 11. Random Forest Regressor

**In simple words:** a Random Forest grows many decision trees, each on a bootstrap sample of the rows and a
random subset of features at each split, and averages their predictions. Each tree asks a sequence of yes/no
questions ("is the 20-cycle mean of s11 above 47.6?"); averaging many de-correlated trees reduces variance.
It captures non-linear effects and interactions and needs no feature scaling.
""")

code(r"""
rf_pipe = Pipeline([("model", RandomForestRegressor(random_state=SEED, n_jobs=1))])
rf_space = {
    "model__n_estimators": [150, 250],
    "model__max_depth": [10, 15, 20, None],
    "model__min_samples_leaf": [2, 5, 10, 20],
    "model__max_features": [0.2, 0.33, 0.5],
}
rf_search = tune("Random Forest", rf_pipe, rf_space)
""")

# =============================================================================
# 12. XGB
# =============================================================================
md(r"""
## 12. XGBoost Regressor

**In simple words:** gradient boosting builds trees *sequentially*; each new small tree is trained to correct
the residual errors of the trees before it. Learning rate, tree depth, subsampling and L1/L2 penalties control
over-fitting. It is often strong on tabular data, but that is not guaranteed — we compare honestly.
""")

code(r"""
if XGB_AVAILABLE:
    xgb_pipe = Pipeline([("model", XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                                random_state=SEED, n_jobs=1, verbosity=0))])
    xgb_space = {
        "model__n_estimators": [200, 400, 600],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__max_depth": [3, 4, 6],
        "model__min_child_weight": [1, 5, 10],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.3, 0.5, 0.8],
        "model__reg_lambda": [1.0, 5.0, 10.0],
    }
    xgb_search = tune("XGBoost", xgb_pipe, xgb_space)
else:
    print("XGBoost unavailable — install with `pip install xgboost` (macOS: `brew install libomp`).")
""")

# =============================================================================
# 13. Elastic Net
# =============================================================================
md(r"""
## 13. Elastic Net Regressor

**In simple words:** a linear model $\hat y = b_0 + \sum_j w_j x_j$ with two penalties on the weights:
**L1** (pushes unhelpful weights to exactly zero → automatic feature selection) and **L2** (shrinks correlated
weights together → stability). `alpha` sets the overall penalty strength and `l1_ratio` the L1/L2 mix.
Because the penalty depends on the size of the weights, features **must be standardised** first — the
`StandardScaler` is the first step of the pipeline (fitted on training rows only).
""")

code(r"""
en_pipe = Pipeline([("scaler", StandardScaler()),
                    ("model", ElasticNet(max_iter=20000, tol=1e-3, random_state=SEED))])
en_space = {
    "model__alpha": np.logspace(-3, 0.5, 20),
    "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
}
en_search = tune("Elastic Net", en_pipe, en_space)
""")

# =============================================================================
# 14. Validation comparison
# =============================================================================
md(r"""
## 14. Validation Performance Comparison

Metrics on **all cycles of the 20 unseen validation engines** (capped target), plus a "test-like" view that uses
one random truncation point per validation engine (like the official test set; only 20 points, so noisy).
**The final model is selected by all-cycle validation RMSE.**
""")

code(r"""
val_table = pd.DataFrame(VAL_RESULTS).T
num_cols = [c for c in val_table.columns if c != "best_params"]
val_table[num_cols] = val_table[num_cols].astype(float)
val_table.index.name = "Model"
val_table.to_csv(METRIC_DIR / "validation_results.csv")

learned = [m for m in ["Random Forest", "XGBoost", "Elastic Net"] if m in VAL_RESULTS]
BEST_MODEL = val_table.loc[learned, "val_RMSE"].idxmin()

show = val_table[["val_RMSE", "val_MAE", "val_R2", "val_testlike_RMSE", "val_testlike_MAE", "val_testlike_NASA_score", "cv_RMSE", "fit_seconds"]]
display(show.round(3))
print("Selected model (lowest all-cycle validation RMSE):", BEST_MODEL)
""")

code(r"""
models_order = list(VAL_RESULTS)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for ax, metric, label in zip(axes, ["val_RMSE", "val_MAE", "val_R2"], ["RMSE (cycles)", "MAE (cycles)", "R²"]):
    vals = val_table.loc[models_order, metric]
    bars = ax.bar(range(len(vals)), vals, color=[MODEL_COLORS[m] for m in models_order], width=0.65)
    ax.set_xticks(range(len(vals)), [m.replace(" (", "\n(") for m in models_order], fontsize=8)
    ax.set_title(f"Validation {label}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout(); save_fig(fig, "validation_comparison"); plt.show()
""")

code(r"""
fig, axes = plt.subplots(3, len(learned), figsize=(5.2 * len(learned), 13))
for j, m in enumerate(learned):
    p, col = VAL_PREDS[m], MODEL_COLORS[m]
    res = p - y_val.values
    ax = axes[0, j]
    ax.scatter(y_val, p, s=4, alpha=0.25, color=col)
    ax.plot([0, CONFIG["rul_cap"]], [0, CONFIG["rul_cap"]], color="#52514e", ls="--", lw=1)
    ax.set(title=f"{m}: actual vs predicted", xlabel="True RUL (capped)", ylabel="Predicted RUL")
    ax = axes[1, j]
    ax.hist(res, bins=50, color=col, edgecolor="white")
    ax.axvline(0, color="#52514e", lw=1)
    ax.set(title=f"Residuals (mean {res.mean():+.1f}, sd {res.std():.1f})", xlabel="Predicted − true (cycles)")
    ax = axes[2, j]
    ax.scatter(y_val, res, s=4, alpha=0.25, color=col)
    binned = pd.Series(res).groupby(pd.cut(y_val.values, np.arange(0, CONFIG["rul_cap"] + 10, 10)), observed=True).mean()
    ax.plot([iv.mid for iv in binned.index], binned.values, color="#0b0b0b", lw=2, label="mean residual")
    ax.axhline(0, color="#52514e", lw=1, ls="--")
    ax.set(title="Residual vs true RUL", xlabel="True RUL (capped)", ylabel="Predicted − true"); ax.legend(fontsize=8)
fig.tight_layout(); save_fig(fig, "validation_diagnostics"); plt.show()
""")

code(r"""
# Prediction trajectories for a few validation engines
show_units = val_units[:4]
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for ax, u in zip(axes, show_units):
    sel = (val_meta["unit"] == u).values
    ax.plot(val_meta.loc[sel, "cycle"], val_meta.loc[sel, "rul"], color="#0b0b0b", lw=2, label="true (capped)")
    for m in learned:
        ax.plot(val_meta.loc[sel, "cycle"], VAL_PREDS[m][sel], color=MODEL_COLORS[m], lw=1.2, label=m)
    ax.set(title=f"Validation engine {u}", xlabel="cycle")
axes[0].set_ylabel("RUL"); axes[0].legend(fontsize=8)
fig.tight_layout(); save_fig(fig, "validation_trajectories"); plt.show()
""")

md("### 14.1 Feature-group ablation (which feature groups help?)")

md(r"""
To answer *which feature groups matter*, each learned model is re-trained on cumulative feature sets with its
**tuned hyper-parameters fixed** and scored on the same validation engines. (Hyper-parameters were tuned on the
full feature set, so this slightly favours the "All" row; the purpose is the *trend*, not a new model selection.)
""")

code(r"""
from sklearn.base import clone

group_of = {c: describe_feature(c)[0] for c in X_COLS}
ablation_sets = {
    "A: raw sensors + cycle": ["A_basic"],
    "A + rolling mean": ["A_basic", "B_rolling_mean"],
    "A + rolling mean + std": ["A_basic", "B_rolling_mean", "B_rolling_std"],
    "A + B + slopes": ["A_basic", "B_rolling_mean", "B_rolling_std", "C_slope"],
    "All (A + B + C + D)": ["A_basic", "B_rolling_mean", "B_rolling_std", "C_slope", "D_health"],
}
ablation_rows = []
for set_name, groups in ablation_sets.items():
    cols = [c for c in X_COLS if group_of[c] in groups]
    for m in learned:
        est = clone(FITTED[m])
        est.fit(X_train[cols], y_train)
        p = clip_pred(est.predict(X_val[cols]))
        ablation_rows.append({"feature_set": set_name, "n_features": len(cols), "model": m, **regression_metrics(y_val, p)})
ablation = pd.DataFrame(ablation_rows)
ablation.to_csv(METRIC_DIR / "feature_group_ablation.csv", index=False)
display(ablation.pivot(index=["feature_set", "n_features"], columns="model", values="RMSE").loc[list(ablation_sets)].round(2))

fig, ax = plt.subplots(figsize=(11, 4))
for m in learned:
    d = ablation[ablation["model"] == m].set_index("feature_set").loc[list(ablation_sets)]
    ax.plot(range(len(d)), d["RMSE"], marker="o", ms=7, color=MODEL_COLORS[m], label=m)
ax.set_xticks(range(len(ablation_sets)), [s.replace(" + ", "\n+ ") for s in ablation_sets], fontsize=8)
ax.set(title="Validation RMSE as feature groups are added", ylabel="RMSE (cycles)")
ax.legend(); fig.tight_layout(); save_fig(fig, "feature_group_ablation"); plt.show()
""")

# =============================================================================
# 15. Interpretability
# =============================================================================
md(r"""
## 15. Feature Importance and Interpretability

* **Tree models:** impurity-based importance (fast, built-in) **and** permutation importance on the validation
  engines (how much RMSE worsens when a feature is shuffled — less biased toward high-cardinality features).
* **Elastic Net:** standardised coefficients (effect of +1 standard deviation of a feature on predicted RUL) and
  the number of features set exactly to zero by the L1 penalty.
""")

code(r"""
from sklearn.inspection import permutation_importance

tree_models = [m for m in ["Random Forest", "XGBoost"] if m in FITTED]
imp_tables = {}
fig, axes = plt.subplots(1, len(tree_models), figsize=(7.5 * len(tree_models), 7))
axes = np.atleast_1d(axes)
for ax, m in zip(axes, tree_models):
    imp = pd.Series(FITTED[m].named_steps["model"].feature_importances_, index=X_COLS).sort_values(ascending=False)
    imp_tables[m] = imp
    top = imp.head(20)[::-1]
    ax.barh(top.index, top.values, color=MODEL_COLORS[m])
    ax.set(title=f"{m}: top-20 impurity importances", xlabel="importance")
fig.tight_layout(); save_fig(fig, "feature_importance_trees"); plt.show()
pd.DataFrame(imp_tables).to_csv(METRIC_DIR / "feature_importance_trees.csv")
""")

code(r"""
# Permutation importance on validation engines (subsample of rows for speed), aggregated by group and by sensor
perm_rows = val_meta.sample(min(2000, len(val_meta)), random_state=SEED).index
perm_summary = {}
for m in learned:
    r = permutation_importance(FITTED[m], X_val.iloc[perm_rows], y_val.iloc[perm_rows], n_repeats=3,
                               random_state=SEED, scoring="neg_root_mean_squared_error", n_jobs=CONFIG["n_jobs"])
    perm_summary[m] = pd.Series(r.importances_mean, index=X_COLS)
perm_df = pd.DataFrame(perm_summary)
perm_df.to_csv(METRIC_DIR / "permutation_importance.csv")

by_group = perm_df.groupby(lambda c: group_of[c]).sum()
by_sensor = perm_df.groupby(lambda c: "cycle" if c == "cycle" else c.split("_")[0]).sum()

fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
by_group.plot.bar(ax=axes[0], color=[MODEL_COLORS[m] for m in by_group.columns], width=0.75, rot=0)
axes[0].set(title="Permutation importance summed by feature group", ylabel="Δ RMSE when shuffled (cycles)")
order_s = by_sensor.mean(axis=1).sort_values(ascending=False).index
by_sensor.loc[order_s].plot.bar(ax=axes[1], color=[MODEL_COLORS[m] for m in by_sensor.columns], width=0.75, rot=0)
axes[1].set(title="Permutation importance summed by sensor", ylabel="Δ RMSE when shuffled (cycles)")
fig.tight_layout(); save_fig(fig, "permutation_importance_groups_sensors"); plt.show()
display(by_group.round(2)); display(by_sensor.loc[order_s].round(2).T)
""")

md(r"""
**Caveat.** Within a group of highly correlated features (e.g. rolling means of the same sensor), shuffling one
feature costs little because its "siblings" carry the same information, so per-feature permutation importances
understate the importance of the group. The group/sensor totals above are therefore more meaningful than
individual feature ranks.
""")

code(r"""
en_model = FITTED["Elastic Net"].named_steps["model"]
coefs = pd.Series(en_model.coef_, index=X_COLS)
nonzero = coefs[coefs != 0]
print(f"Elastic Net kept {len(nonzero)} of {len(coefs)} features (L1 zeroed {int((coefs == 0).sum())});"
      f" intercept = {en_model.intercept_:.1f}")
top_en = nonzero.reindex(nonzero.abs().sort_values(ascending=False).index).head(20)
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.barh(top_en.index[::-1], top_en.values[::-1], color=["#2a78d6" if v > 0 else "#e34948" for v in top_en.values[::-1]])
ax.axvline(0, color="#52514e", lw=0.8)
ax.set(title="Elastic Net: top-20 standardised coefficients\n(blue = higher value → longer RUL, red = shorter RUL)",
       xlabel="Δ predicted RUL per +1 std of feature (cycles)")
fig.tight_layout(); save_fig(fig, "elastic_net_coefficients"); plt.show()
coefs.to_frame("standardised_coef").to_csv(METRIC_DIR / "elastic_net_coefficients.csv")
""")

# =============================================================================
# 16. Early vs late
# =============================================================================
md(r"""
## 16. Early- versus Late-Degradation Analysis

Stages are defined on the **true uncapped RUL** of the validation rows:
* **Early degradation:** true RUL > 100 cycles. Note that with the cap at 125, rows with true RUL > 125 all have the
  same target (125), so "early" mixes the flat healthy plateau and the start of degradation.
* **Late degradation:** true RUL ≤ 100 cycles.

Errors are measured against the capped target (the quantity the models were trained to predict). A finer
three-bin view (> 125, 51–125, ≤ 50) is also reported.
""")

code(r"""
def stage_table(meta, preds, bins, labels):
    stage = pd.cut(meta["rul_raw"], bins=bins, labels=labels)
    rows = []
    for m, p in preds.items():
        for st in labels:
            sel = (stage == st).values
            if sel.sum() == 0:
                continue
            mm = regression_metrics(meta.loc[sel, "rul"], p[sel])
            rows.append({"model": m, "stage": st, "n_obs": int(sel.sum()), "n_engines": int(meta.loc[sel, "unit"].nunique()),
                         "RMSE": mm["RMSE"], "MAE": mm["MAE"], "mean_error": float(np.mean(p[sel] - meta.loc[sel, "rul"])),
                         "RMSE_vs_uncapped": float(np.sqrt(np.mean((p[sel] - meta.loc[sel, "rul_raw"]) ** 2)))})
    return pd.DataFrame(rows)


thr = CONFIG["stage_threshold"]
stage2 = stage_table(val_meta, VAL_PREDS, [-1, thr, np.inf], [f"late (RUL ≤ {thr})", f"early (RUL > {thr})"])
stage3 = stage_table(val_meta, VAL_PREDS, [-1, 50, CONFIG["rul_cap"], np.inf],
                     ["≤ 50", f"51–{CONFIG['rul_cap']}", f"> {CONFIG['rul_cap']} (plateau)"])
stage2.to_csv(METRIC_DIR / "stage_errors_2bins.csv", index=False)
stage3.to_csv(METRIC_DIR / "stage_errors_3bins.csv", index=False)
display(stage2.pivot(index="model", columns="stage", values=["n_obs", "RMSE", "MAE", "mean_error"]).round(2))
display(stage3.pivot(index="model", columns="stage", values=["n_obs", "RMSE", "MAE", "mean_error", "RMSE_vs_uncapped"]).round(2))
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
for ax, tab, title in [(axes[0], stage2, "Two stages"), (axes[1], stage3, "Three stages")]:
    stages = list(dict.fromkeys(tab["stage"]))
    width = 0.8 / len(learned)
    for i, m in enumerate(learned):
        d = tab[tab["model"] == m].set_index("stage").loc[stages]
        ax.bar(np.arange(len(stages)) + i * width - 0.4 + width / 2, d["RMSE"], width=width * 0.92,
               color=MODEL_COLORS[m], label=m)
    ax.set_xticks(range(len(stages)), stages)
    ax.set(title=f"Validation RMSE by degradation stage — {title}", ylabel="RMSE (cycles)")
axes[0].legend()
fig.tight_layout(); save_fig(fig, "stage_errors"); plt.show()
""")

# =============================================================================
# 17. Official test
# =============================================================================
md(r"""
## 17. Official NASA Test-Set Evaluation

Protocol:
1. The model type and hyper-parameters were chosen on validation engines (Section 14). Before testing, every
   learned model is **re-fitted on all 100 training engines** with those fixed hyper-parameters (the scaler inside
   the pipeline is re-fitted on training data only). Nothing is fitted on test data.
2. Test features are built with the **same** `build_features` function, separately inside each test engine's
   truncated history (windows shorter than the history automatically use partial windows).
3. One prediction per test engine, taken at its **last observed cycle**.
4. True RUL = value in `RUL_FD001.txt` (row *i* ↔ engine *i*), capped at 125 for the primary evaluation —
   the same target definition as training. Uncapped metrics are reported as a supplementary view.

The **primary result is the selected model**. The other models' test scores are shown afterwards for transparency
only; they were **not** used for any decision.
""")

code(r"""
X_full, y_full = feat_all[X_COLS], feat_all["rul"]
FINAL = {}
for m in ["Baseline (mean)", "Baseline (cycle-only)"] + learned:
    est = clone(FITTED[m])
    cols = ["cycle"] if m == "Baseline (cycle-only)" else X_COLS
    t0 = time.time()
    FINAL[m] = (est.fit(X_full[cols], y_full), cols)
    joblib.dump({"model": FINAL[m][0], "feature_columns": cols, "selected_sensors": SELECTED_SENSORS, "config": CONFIG},
                MODEL_DIR / f"{m.replace(' ', '_').replace('(', '').replace(')', '').lower()}.joblib")
    print(f"refit {m:22s} on all {feat_all['unit'].nunique()} training engines in {time.time() - t0:.1f}s")

# --- test features (each engine's own history only) -------------------------
test_feat = build_features(test_raw, SELECTED_SENSORS)
assert list(test_feat[X_COLS].columns) == X_COLS
test_last = test_feat[last_cycle_rows(test_feat)].sort_values("unit").reset_index(drop=True)
assert len(test_last) == test_raw["unit"].nunique() == len(rul_official)
assert (test_last["cycle"].values == test_raw.groupby("unit")["cycle"].max().loc[test_last["unit"]].values).all()

truth = rul_official.set_index("unit").loc[test_last["unit"], "true_rul"].values
truth_capped = np.minimum(truth, CONFIG["rul_cap"]) if CONFIG["rul_cap"] else truth
print(f"Test feature matrix: {test_last[X_COLS].shape} — one row per engine at its last observed cycle")
""")

code(r"""
TEST_PRED = {m: clip_pred(est.predict(test_last[cols]).astype(float)) for m, (est, cols) in FINAL.items()}

best_pred = TEST_PRED[BEST_MODEL]
primary = regression_metrics(truth_capped, best_pred, nasa=True)
uncapped = regression_metrics(truth, best_pred, nasa=True)
print(f"PRIMARY RESULT — selected model: {BEST_MODEL}")
print(f"  capped target   : RMSE {primary['RMSE']:.2f} | MAE {primary['MAE']:.2f} | R² {primary['R2']:.3f} | NASA score {primary['NASA_score']:.1f}")
print(f"  uncapped target : RMSE {uncapped['RMSE']:.2f} | MAE {uncapped['MAE']:.2f} | R² {uncapped['R2']:.3f} | NASA score {uncapped['NASA_score']:.1f}")

per_engine = pd.DataFrame({"unit": test_last["unit"], "observed_cycles": test_last["cycle"],
                           "true_RUL": truth, "true_RUL_capped": truth_capped,
                           "predicted_RUL": best_pred.round(1)})
per_engine["error"] = (per_engine["predicted_RUL"] - per_engine["true_RUL_capped"]).round(1)
per_engine["abs_error"] = per_engine["error"].abs()
for m, p in TEST_PRED.items():
    per_engine[f"pred_{m}"] = p.round(1)
per_engine.to_csv(PRED_DIR / "test_predictions_per_engine.csv", index=False)
display(per_engine[["unit", "observed_cycles", "true_RUL", "true_RUL_capped", "predicted_RUL", "error"]].head(25))
print("Full 100-engine table saved to", PRED_DIR / "test_predictions_per_engine.csv")
""")

code(r"""
test_rows = []
for m, p in TEST_PRED.items():
    test_rows.append({"Model": m, **regression_metrics(truth_capped, p, nasa=True),
                      **{f"uncapped_{k}": v for k, v in regression_metrics(truth, p).items()},
                      "selected_on_validation": m == BEST_MODEL})
test_table = pd.DataFrame(test_rows).set_index("Model")
test_table.to_csv(METRIC_DIR / "test_results.csv")
display(test_table.round(3))
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
col = MODEL_COLORS[BEST_MODEL]
axes[0].scatter(truth_capped, best_pred, s=22, color=col, edgecolor="white", linewidth=0.6)
axes[0].plot([0, CONFIG["rul_cap"]], [0, CONFIG["rul_cap"]], color="#52514e", ls="--", lw=1)
axes[0].set(title=f"Official test: {BEST_MODEL}\nRMSE {primary['RMSE']:.2f}, MAE {primary['MAE']:.2f}",
            xlabel="True RUL (capped)", ylabel="Predicted RUL")
o = per_engine.sort_values("true_RUL_capped")
axes[1].plot(range(len(o)), o["true_RUL_capped"], color="#0b0b0b", lw=2, label="true (capped)")
axes[1].scatter(range(len(o)), o["predicted_RUL"], s=14, color=col, label="predicted", zorder=3)
axes[1].set(title="Per-engine predictions (sorted by true RUL)", xlabel="Test engine (sorted)", ylabel="RUL"); axes[1].legend()
axes[2].scatter(per_engine["true_RUL_capped"], per_engine["error"], s=22, color=col, edgecolor="white", linewidth=0.6)
axes[2].axhline(0, color="#52514e", ls="--", lw=1)
axes[2].set(title="Test error vs true RUL (positive = over-estimate, riskier)", xlabel="True RUL (capped)", ylabel="Predicted − true")
fig.tight_layout(); save_fig(fig, "test_selected_model"); plt.show()

thr_mask = per_engine["true_RUL"] <= CONFIG["stage_threshold"]
print(f"Test engines with true RUL ≤ {thr}: {thr_mask.sum()} | MAE {per_engine.loc[thr_mask, 'abs_error'].mean():.2f}")
print(f"Test engines with true RUL > {thr}: {(~thr_mask).sum()} | MAE {per_engine.loc[~thr_mask, 'abs_error'].mean():.2f}")
print("Largest 5 absolute errors:")
display(per_engine.nlargest(5, "abs_error")[["unit", "observed_cycles", "true_RUL", "predicted_RUL", "error"]])
""")

# =============================================================================
# 18. Final results
# =============================================================================
md(r"""
## 18. Final Results and Discussion

### 18.1 Summary tables (generated from this run)
""")

code(r"""
summary = pd.DataFrame({
    "Val RMSE": val_table["val_RMSE"], "Val MAE": val_table["val_MAE"], "Val R²": val_table["val_R2"],
    "Test RMSE": test_table["RMSE"], "Test MAE": test_table["MAE"], "Test R²": test_table["R2"],
    "Test NASA score": test_table["NASA_score"]}).loc[models_order]
summary.to_csv(METRIC_DIR / "final_summary.csv")
print("Validation = all cycles of 20 held-out training engines; Test = last cycle of 100 official test engines (capped at", CONFIG["rul_cap"], ")")
print("Model selected on validation:", BEST_MODEL)
display(summary.round(2))
""")

md(r"""
### 18.2 Discussion
*The numbers quoted below are those produced by this notebook with `seed = 42`; re-running reproduces them.*

**Which classical model performs best?** XGBoost had the lowest all-cycle validation RMSE (9.41 vs 10.21 for
Random Forest and 15.45 for Elastic Net) and was therefore selected *before* touching the test set. On the 100
official test engines it obtained **RMSE 10.50, MAE 7.81, R² 0.93 and NASA score 163.9** (capped target). The
ranking on the test set (XGBoost < Random Forest 11.30 < Elastic Net 16.15 < cycle-only 32.25 < mean 41.94) is
the same as on validation, which suggests the engine-level validation protocol was a reliable guide. The gap between
XGBoost and Random Forest is small (≈ 0.8 RMSE on both validation and test) and comes from a single split and seed,
so it should be described as "slightly better", not decisively better.

**Did the engineered features help? (research question)** Yes — clearly, and this is measured, not assumed. With
raw sensors + cycle only, the tree models reach ≈ 15.6 validation RMSE; with the full engineered set they reach
10.2 (RF) and 9.4 (XGBoost), a **~35–40 % reduction in RMSE**. All learned models beat the cycle-only baseline
(26.0) and the mean baseline (41.7) by a wide margin.

**Which feature groups appear most useful?** From the ablation and the permutation importances:
* **D – health indicators** (especially *deviation from the engine's own healthy baseline*) gave the largest single
  improvement (e.g. XGBoost 13.43 → 9.41) and dominate the permutation importance of all three models.
* **B – rolling means** give a modest, consistent gain by removing measurement noise (≈ 1 cycle RMSE).
* **C – rolling slopes** helped the tree models (RF 14.13 → 12.75) but have near-zero permutation importance once
  health indicators are present — the trend information is largely redundant with the baseline deviation.
* **B – rolling standard deviations** brought no measurable benefit; sensor volatility does not appear to change
  much with degradation in FD001.

**Which sensors appear most informative?** s11 (Ps30, HPC static pressure), s15 (bypass ratio), s4 (LPT outlet
temperature) and s9 (core speed), followed by s12, s14, s20, s21. Interestingly, **s9 and s14 have weak raw
correlation with RUL** (|r| ≈ 0.3–0.4) because each engine has a different offset, but their *baseline-corrected*
versions (`s9_dev_base`, `s14_dev_base`) are among the most important features — a concrete example of feature
engineering turning a weak raw signal into a useful one. XGBoost also leans heavily on `cycle` (engine age), which is
legitimate (known at prediction time) but means part of its accuracy comes from fleet-average ageing behaviour.

**How do errors differ between early and late degradation?** Against the capped target, the "early" stage
(true RUL > 100) actually has *lower* error (XGBoost RMSE 7.06) than the late stage (11.31). This is an artefact of
capping: most early rows sit on the flat 125 plateau, which is easy to predict. The three-bin view shows the real
picture: errors are smallest close to failure (≤ 50 cycles: RMSE 6.10, where degradation is obvious), largest in the
**transition zone 51–125 cycles (RMSE 13.49)**, where degradation is just starting and engines differ in how fast it
develops, and on the plateau the models are accurate only relative to the cap — measured against the *uncapped*
truth their RMSE there is ≈ 64 cycles. In other words, the sensors simply do not contain information about how long a
healthy engine will stay healthy. On the test set the same pattern appears: MAE 6.67 for engines with true
RUL ≤ 100 (67 engines) vs 10.15 for RUL > 100 (33 engines). The early-life numbers rest on 20 validation engines and
33 test engines, so the exact values are uncertain.

**Bias / safety.** In the late stage the models' mean error is slightly positive (XGBoost ≈ +3.7 cycles for
RUL ≤ 100), i.e. they tend to *over-estimate* remaining life, which is the more dangerous direction and is what the
NASA score penalises. The largest test errors (≈ 23–34 cycles) occur mostly for engines with short observed histories
or true RUL near the transition zone.

**How interpretable are the models?** Elastic Net is the most transparent: with L1 ratio 1.0 (pure Lasso) it kept only
**18 of 155 features**, and the signs make physical sense — higher Ps30 drift (`s11_dev_base`, −9.6 cycles per
std) and bypass-ratio drift (`s15_dev_base`, −6.5) reduce predicted RUL, while the coolant-bleed deviations
(`s20/s21_dev_base`) have positive coefficients — these sensors *fall* as the engine degrades, so a falling value also
reduces predicted RUL. This matches the trends seen in the EDA. But it is ~50 % less accurate because the
RUL–sensor relationship is non-linear (visible as the curved residual pattern). The tree ensembles are more accurate
but only *post-hoc* interpretable (importances, permutation tests); importances of correlated features must be read
at group/sensor level.
""")

md(r"""
## 19. Limitations and Future Scope

**Limitations of rolling means and slopes**
* A window of *w* cycles introduces **lag**: the smoothed signal reacts to a change only after several cycles.
* At the start of a history, windows are partial, so features are noisier; test engines with very short histories
  (31–50 cycles) get less reliable trend estimates.
* Slopes over short windows amplify noise (10-cycle slopes are visibly noisier than 20-cycle slopes).
* Fixed windows treat all engines the same, although they degrade at different speeds.
* The healthy baseline assumes the first 10 cycles are healthy, which holds for C-MAPSS by design but not necessarily in real fleets.

**Limitations of the experiment**
* Only **FD001** (one operating condition, one fault mode) — results do not transfer automatically to FD002–FD004.
* Simulated data (C-MAPSS), not real engines.
* A single 80/20 engine split and a single seed; with only 20 validation engines the differences between RF and XGBoost
  (≈ 1 cycle RMSE) are within plausible split-to-split variation. Repeated group-wise CV would give confidence intervals.
* The RUL cap (125) is a modelling choice that shapes all reported metrics; metrics against uncapped truth are higher.
* Hyper-parameter search was deliberately small (12 random configurations per model) to stay laptop-friendly.
* The feature-group ablation re-used hyper-parameters tuned on the full feature set.
* Point predictions only — no uncertainty intervals, which maintenance decisions would need.

**Future scope (Phase 2 and beyond)**
* Deep learning sequence models (LSTM/CNN/Transformer) on raw windows, compared against this classical baseline
  under the *same* engine-level protocol and RUL cap.
* Operating-condition normalisation and fault-mode handling for FD002–FD004.
* Repeated group k-fold evaluation and confidence intervals; quantile or probabilistic RUL prediction.
* Asymmetric (safety-aware) loss functions that penalise over-estimation, aligned with the NASA score.
* Exponentially-weighted features or health-index construction to reduce smoothing lag.
""")

md(r"""
## 20. Conclusion

Using only classical machine learning, an engine-level validation protocol and leakage-checked features, the
selected **XGBoost** model predicts the RUL of the 100 official FD001 test engines with **RMSE 10.50 cycles and MAE
7.81 cycles (R² 0.93)**, versus 41.94 for a constant baseline and 32.25 for an age-only baseline. Random Forest is a
close second (RMSE 11.30) and Elastic Net, while far more interpretable (18 active features), is limited by the
non-linear degradation pattern (RMSE 16.15).

The answer to the research question is **yes**: engineered degradation features — above all the *deviation of each
sensor from the engine's own healthy baseline*, supported by noise-reducing rolling means and trend slopes — reduced
validation RMSE by roughly 35–40 % compared with raw sensor readings. The remaining error is concentrated in the
transition from healthy to degrading operation, where the sensors carry little information about how much life is
left; this is a property of the data, not only of the models, and is the natural starting point for Phase 2.
""")

code(r"""
# Final artifact inventory
for sub in [FIG_DIR, MODEL_DIR, METRIC_DIR, PRED_DIR]:
    print(f"{sub}/: {sorted(p.name for p in sub.iterdir())}")
""")

# Appendix: academic documentation (single source: ACADEMIC_REPORT.md)
report = open("ACADEMIC_REPORT.md", encoding="utf-8").read()
md(report.replace("# Academic Documentation", "# Appendix — Academic Documentation", 1))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "predictive_maintenance_aml_phase1.ipynb")
print(f"wrote notebook with {len(cells)} cells")
