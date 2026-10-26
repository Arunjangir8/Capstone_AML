"""Builds the (simple) predictive_maintenance_aml_phase1.ipynb.

Run:  python build_notebook.py
Then: jupyter nbconvert --to notebook --execute --inplace predictive_maintenance_aml_phase1.ipynb
"""
import nbformat as nbf

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip("\n")))

# ---------------------------------------------------------------- 1-2
md(r"""
# Predictive Maintenance — Phase 1 (Classical Machine Learning)

**Goal:** predict the **Remaining Useful Life (RUL)** of turbofan engines — how many cycles an engine can still run
before it fails — using the NASA C-MAPSS dataset (subset **FD001**).

**Steps:** Load data → EDA → Build RUL target → Preprocess → Split by engine → Feature engineering →
Train models (Baseline, Random Forest, XGBoost, Elastic Net) → Evaluate → Test on official NASA test set.

**Setup**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib jupyter
brew install libomp      # macOS only, needed by XGBoost
```

## 1–2. Problem Statement
Each engine starts healthy, slowly degrades and finally fails. At every cycle we get 3 operating settings and
21 sensor readings. RUL is a **number** (cycles left), so this is a **regression** problem.

**Research question:** *Can engineered sensor degradation features help classical ML models estimate RUL accurately?*
""")

# ---------------------------------------------------------------- 3
md("## 3. Load and Check the Data")

code(r"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor   # if this fails: pip install xgboost  (macOS: brew install libomp)

# ---- settings (change here) ----
DATA_DIR = os.environ.get("CMAPSS_DATA_DIR", "../CMAPSSData")   # folder with the NASA .txt files
RUL_CAP = 125          # maximum RUL used as target
WINDOW = 10            # rolling window size (cycles)
SEED = 42
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

sns.set_theme(style="whitegrid")
COLORS = {"Baseline": "#8a8984", "Random Forest": "#2a78d6", "XGBoost": "#eb6834", "Elastic Net": "#1baf7a"}
""")

code(r"""
columns = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"s{i}" for i in range(1, 22)]

def read_file(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - set DATA_DIR to your CMAPSSData folder")
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns)

train = read_file("train_FD001.txt")
test = read_file("test_FD001.txt")
test_rul = pd.read_csv(os.path.join(DATA_DIR, "RUL_FD001.txt"), header=None, names=["true_rul"])
test_rul["unit"] = range(1, len(test_rul) + 1)

print("Train shape:", train.shape, "| Test shape:", test.shape, "| RUL file:", test_rul.shape)
print("Engines -> train:", train["unit"].nunique(), "| test:", test["unit"].nunique())
print("Missing values:", train.isna().sum().sum(), "| Duplicate rows:", train.duplicated().sum())
train.head()
""")

md("### Constant sensors\nSensors that (almost) never change carry no information, so we look at their spread.")

code(r"""
sensor_cols = [f"s{i}" for i in range(1, 22)]
summary = pd.DataFrame({"std": train[sensor_cols].std(), "unique_values": train[sensor_cols].nunique()})
summary["constant"] = summary["unique_values"] <= 2
print(summary.round(4))
print("\nOperating setting_3 unique values:", train["setting_3"].unique())
""")

md(r"""
**Data dictionary (short):** `unit` = engine id, `cycle` = time step, `setting_1-3` = operating settings,
`s1-s21` = sensors (temperatures, pressures, fan/core speeds, fuel ratio, bleed flows). For example s2/s3/s4 are
temperatures, s7/s11 are pressures at the compressor outlet, s9/s14 are core speeds and s20/s21 are coolant bleeds.

**Result:** s1, s5, s10, s16, s18 and s19 never change, and s6 has only 2 values. These 7 sensors will be removed.
""")

# ---------------------------------------------------------------- 4 EDA
md("## 4. Exploratory Data Analysis")

code(r"""
life = train.groupby("unit")["cycle"].max()

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(life, bins=20, color="#2a78d6", edgecolor="white")
ax[0].set(title="Engine lifetime (cycles until failure)", xlabel="Cycles", ylabel="Number of engines")
train_rul_uncapped = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
ax[1].hist(train_rul_uncapped, bins=40, color="#2a78d6", edgecolor="white")
ax[1].set(title="RUL distribution (training rows)", xlabel="RUL", ylabel="Rows")
plt.tight_layout(); plt.savefig(f"{RESULTS}/eda_lifetime_rul.png"); plt.show()
print(life.describe().round(1))
""")

md("**Observation:** engines live between 128 and 362 cycles (median about 199). Every cycle is one row, so small RUL values appear for every engine while very large RUL values come only from long-lived engines.")

code(r"""
# Sensor trends for 4 engines (x = RUL, failure at 0)
plot_sensors = ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s21"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, s in zip(axes.flat, plot_sensors):
    for unit in [1, 20, 50, 80]:
        d = train[train["unit"] == unit]
        ax.plot(d["cycle"].max() - d["cycle"], d[s], lw=1, label=f"engine {unit}")
    ax.set(title=s, xlabel="RUL")
axes[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{RESULTS}/eda_sensor_trends.png"); plt.show()
""")

md("**Observation:** sensors are noisy, but close to failure (RUL near 0) they clearly drift. s2, s3, s4, s11 and s15 go **up**, while s7, s12 and s21 go **down**. Far from failure the lines are almost flat.")

code(r"""
useful_sensors = summary.index[~summary["constant"]].tolist()
corr = train[useful_sensors].assign(RUL=train_rul_uncapped).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".1f", annot_kws={"size": 7})
plt.title("Correlation between sensors and RUL")
plt.savefig(f"{RESULTS}/eda_correlation.png"); plt.show()
print(corr["RUL"].drop("RUL").sort_values().round(2))
""")

md("**Observation:** s11, s4, s12, s7, s15, s21 and s20 are the most correlated with RUL (|r| ≈ 0.6–0.7). Many sensors are correlated with each other because they all react to the same fault. s9 and s14 are weaker.")

code(r"""
# Early life vs late life: does a sensor look different close to failure?
stage = np.where(train_rul_uncapped > 100, "early (RUL > 100)", "late (RUL <= 100)")
z = (train[useful_sensors] - train[useful_sensors].mean()) / train[useful_sensors].std()
z["stage"] = stage
plt.figure(figsize=(14, 4))
sns.boxplot(data=z.melt(id_vars="stage"), x="variable", y="value", hue="stage", showfliers=False)
plt.title("Standardised sensor values: early vs late life"); plt.xlabel("sensor"); plt.ylabel("z-score")
plt.savefig(f"{RESULTS}/eda_early_vs_late.png"); plt.show()
""")

md("**Observation:** late-life values are shifted by about one standard deviation for most sensors, but the boxes overlap. A single reading is not enough, so the **trend over several cycles** should help. This motivates rolling features.")

md("**Operating settings:** FD001 has only one operating condition (`setting_3` is always 100, and settings 1–2 vary only slightly), so we do **not** need operating-condition normalisation. The settings are dropped.")

# ---------------------------------------------------------------- 5 RUL
md(r"""
## 5. RUL Target
For every engine: **RUL = last cycle of that engine − current cycle**. If an engine fails at cycle 200, at cycle 150 its RUL is 50.

We **cap RUL at 125**. Early in life the sensors look healthy, so the model cannot tell 300 cycles left from 150 cycles left.
Capping means "healthy engine = 125". The same capped target is used for all models and for the test set.
""")

code(r"""
def add_rul(df):
    df = df.sort_values(["unit", "cycle"]).copy()
    df["RUL_raw"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["RUL"] = df["RUL_raw"].clip(upper=RUL_CAP)
    return df

train = add_rul(train)

# checks
assert (train.groupby("unit")["RUL_raw"].min() == 0).all(), "RUL must be 0 at the last cycle"
print(train[train["unit"] == 1][["unit", "cycle", "RUL_raw", "RUL"]].iloc[[0, 1, 100, 150, -1]])
""")

# ---------------------------------------------------------------- 6-7
md(r"""
## 6–7. Preprocessing and Engine-Level Split
* Drop the constant sensors and the operating settings.
* Split **by engine**: 80 engines for training and 20 for validation. Rows of the same engine are almost identical,
  so splitting rows randomly would let the model "see" the validation engines during training.
* Scaling (only needed for Elastic Net) is done inside a pipeline, so it is learned from **training data only**.
""")

code(r"""
SENSORS = useful_sensors          # 14 sensors kept
print("Sensors used:", SENSORS)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(splitter.split(train, groups=train["unit"]))
train_units = train.iloc[train_idx]["unit"].unique()
val_units = train.iloc[val_idx]["unit"].unique()

assert len(set(train_units) & set(val_units)) == 0, "an engine is in both sets!"
print(f"Training engines: {len(train_units)}, validation engines: {len(val_units)}")
""")

# ---------------------------------------------------------------- 8 FE
md(r"""
## 8. Feature Engineering
For every sensor, inside each engine, using only the **current and past** cycles (window = 10 cycles):

| Feature | Meaning | Why it helps |
|---|---|---|
| `s_mean` | average of the last 10 readings | removes noise |
| `s_std` | spread of the last 10 readings | shows if the sensor becomes unstable |
| `s_slope` | slope of a straight line through the last 10 readings | shows the **trend**: is the sensor rising or falling, and how fast? |
| `s_dev` | rolling mean minus the engine's own healthy start value (mean of its first 10 cycles) | how far the engine has drifted from **its own** healthy state |

We also keep `cycle` (engine age) and the raw sensor values.
For the first cycles the window is shorter (e.g. at cycle 3 we average cycles 1–3). std and slope are 0 at cycle 1.

*Slope sign:* for temperature sensors a **positive** slope means the engine is heating up (degrading). For s7, s12,
s20 and s21 a **negative** slope means degradation.
""")

code(r"""
def make_features(df):
    all_engines = []
    for unit, g in df.groupby("unit"):              # each engine separately -> no mixing of engines
        g = g.sort_values("cycle").copy()
        t = g["cycle"]
        for s in SENSORS:
            roll = g[s].rolling(WINDOW, min_periods=1)
            g[f"{s}_mean"] = roll.mean()
            g[f"{s}_std"] = roll.std().fillna(0)
            slope = g[s].rolling(WINDOW, min_periods=2).cov(t) / t.rolling(WINDOW, min_periods=2).var()
            g[f"{s}_slope"] = slope.fillna(0)
            baseline = g[s].expanding().mean().where(t <= 10).ffill()   # mean of first 10 cycles (past only)
            g[f"{s}_dev"] = g[f"{s}_mean"] - baseline
        all_engines.append(g)
    return pd.concat(all_engines, ignore_index=True)

train_feat = make_features(train)
FEATURES = ["cycle"] + SENSORS + [f"{s}_{k}" for s in SENSORS for k in ["mean", "std", "slope", "dev"]]
print("Number of features:", len(FEATURES))
train_feat[["unit", "cycle", "s11", "s11_mean", "s11_std", "s11_slope", "s11_dev", "RUL"]].head(12)
""")

code(r"""
# Plot the engineered features for one engine
d = train_feat[train_feat["unit"] == 1]
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].plot(d["cycle"], d["s11"], color="#8a8984", lw=1, label="raw")
ax[0].plot(d["cycle"], d["s11_mean"], color="#2a78d6", lw=2, label="rolling mean")
ax[0].set(title="s11: raw vs rolling mean", xlabel="cycle"); ax[0].legend()
ax[1].plot(d["cycle"], d["s11_slope"], color="#eb6834"); ax[1].axhline(0, color="black", lw=0.8)
ax[1].set(title="s11: rolling slope", xlabel="cycle")
ax[2].plot(d["cycle"], d["s11_dev"], color="#1baf7a"); ax[2].axhline(0, color="black", lw=0.8)
ax[2].set(title="s11: deviation from healthy start", xlabel="cycle")
plt.tight_layout(); plt.savefig(f"{RESULTS}/features_example.png"); plt.show()
""")

md("**Observation:** the rolling mean removes most of the noise. The slope and the deviation stay near 0 while the engine is healthy and rise near the end of its life.")

# ---------------------------------------------------------------- 9 leakage
md("## 9. Leakage Checks")

code(r"""
# 1) Features must not use the future: recompute features for engine 1 using only its first 100 cycles
short = make_features(train[(train["unit"] == 1) & (train["cycle"] <= 100)])
full = train_feat[train_feat["unit"] == 1].head(100)
assert np.allclose(short[FEATURES].values, full[FEATURES].values), "features use future data!"

# 2) No missing values, and the target is not a feature
assert train_feat[FEATURES].isna().sum().sum() == 0
assert "RUL" not in FEATURES and "RUL_raw" not in FEATURES

# 3) Train and validation engines are different
X_train = train_feat[train_feat["unit"].isin(train_units)]
X_val = train_feat[train_feat["unit"].isin(val_units)]
assert set(X_train["unit"]).isdisjoint(X_val["unit"])

y_train, y_val = X_train["RUL"], X_val["RUL"]
groups = X_train["unit"]
print("All leakage checks passed. Train rows:", len(X_train), "| Validation rows:", len(X_val))
""")

# ---------------------------------------------------------------- metrics helper + baseline
md(r"""
## 10. Baseline Model
**Metrics:** RMSE = typical error size (big errors punished more); MAE = average error in cycles;
R² = how much of the variation is explained (1 = perfect).
The baseline always predicts the average training RUL. Every real model must beat it.
""")

code(r"""
def evaluate(name, y_true, y_pred):
    return {"Model": name,
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
            "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
            "R2": round(float(r2_score(y_true, y_pred)), 3)}

results = []
val_preds = {}

val_preds["Baseline"] = np.full(len(y_val), y_train.mean())
results.append(evaluate("Baseline", y_val, val_preds["Baseline"]))
print(results[-1])
""")

md(r"""
**Tuning method (the same for all models):** a small grid search with 3-fold `GroupKFold` on the **training engines
only** (an engine never sits in two folds). The validation engines are used only to compare the final models.
""")

code(r"""
cv = GroupKFold(n_splits=3)

def tune_and_validate(name, pipeline, grid):
    search = GridSearchCV(pipeline, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(X_train[FEATURES], y_train, groups=groups)
    model = search.best_estimator_
    pred = np.clip(model.predict(X_val[FEATURES]), 0, RUL_CAP)
    val_preds[name] = pred.astype(float)
    results.append(evaluate(name, y_val, pred))
    print(name, "best params:", search.best_params_)
    print(name, "validation:", results[-1])
    joblib.dump(model, f"{RESULTS}/{name.replace(' ', '_').lower()}.joblib")
    return model
""")

# ---------------------------------------------------------------- 11-13
md(r"""
## 11. Random Forest
Many decision trees, each trained on a random sample of the data. The final prediction is the **average** of all trees.
It handles non-linear patterns and needs no scaling.
""")
code(r"""
rf = tune_and_validate("Random Forest",
    Pipeline([("model", RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=-1))]),
    {"model__max_depth": [10, 20], "model__min_samples_leaf": [5, 20]})
""")

md(r"""
## 12. XGBoost
Trees are built **one after another**, and each new tree corrects the errors of the previous ones (gradient boosting).
""")
code(r"""
xgb = tune_and_validate("XGBoost",
    Pipeline([("model", XGBRegressor(n_estimators=300, subsample=0.8, colsample_bytree=0.5, random_state=SEED, n_jobs=-1))]),
    {"model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1]})
""")

md(r"""
## 13. Elastic Net
A **linear** model (a weighted sum of the features) with a penalty that shrinks weights. L1 sets useless weights to 0
and L2 keeps correlated features stable. Features must be **scaled** first, so a `StandardScaler` is the first step of the pipeline.
""")
code(r"""
enet = tune_and_validate("Elastic Net",
    Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(max_iter=10000, random_state=SEED))]),
    {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.9]})
""")

# ---------------------------------------------------------------- 14
md("## 14. Validation Comparison")
code(r"""
val_table = pd.DataFrame(results).set_index("Model").round(2)
val_table.to_csv(f"{RESULTS}/validation_results.csv")
best_model_name = val_table.drop("Baseline")["RMSE"].idxmin()
print("Best model on validation:", best_model_name)
val_table
""")

code(r"""
models = ["Random Forest", "XGBoost", "Elastic Net"]
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for i, m in enumerate(models):
    err = val_preds[m] - y_val.values
    ax[0, i].scatter(y_val, val_preds[m], s=3, alpha=0.3, color=COLORS[m])
    ax[0, i].plot([0, RUL_CAP], [0, RUL_CAP], "k--", lw=1)
    ax[0, i].set(title=f"{m}: actual vs predicted", xlabel="True RUL", ylabel="Predicted RUL")
    ax[1, i].scatter(y_val, err, s=3, alpha=0.3, color=COLORS[m])
    ax[1, i].axhline(0, color="black", lw=1)
    ax[1, i].set(title=f"{m}: error vs true RUL", xlabel="True RUL", ylabel="Predicted - True")
plt.tight_layout(); plt.savefig(f"{RESULTS}/validation_predictions.png"); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for m in models:
    ax[0].hist(val_preds[m] - y_val.values, bins=50, alpha=0.5, color=COLORS[m], label=m)
ax[0].set(title="Error distribution", xlabel="Predicted - True"); ax[0].legend()
ax[1].bar(val_table.index, val_table["RMSE"], color=[COLORS[m] for m in val_table.index])
ax[1].set(title="Validation RMSE (lower is better)", ylabel="RMSE (cycles)")
plt.tight_layout(); plt.savefig(f"{RESULTS}/validation_comparison.png"); plt.show()
""")

# ---------------------------------------------------------------- 15
md("## 15. Feature Importance")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
for a, (name, model) in zip(ax[:2], [("Random Forest", rf), ("XGBoost", xgb)]):
    imp = pd.Series(model.named_steps["model"].feature_importances_, index=FEATURES).nlargest(15)
    a.barh(imp.index[::-1], imp.values[::-1], color=COLORS[name])
    a.set_title(f"{name}: top 15 features")

coef = pd.Series(enet.named_steps["model"].coef_, index=FEATURES)
top = coef[coef.abs().nlargest(15).index]
ax[2].barh(top.index[::-1], top.values[::-1], color=["#2a78d6" if v > 0 else "#e34948" for v in top.values[::-1]])
ax[2].set_title("Elastic Net: top 15 coefficients\n(blue = longer RUL, red = shorter RUL)")
plt.tight_layout(); plt.savefig(f"{RESULTS}/feature_importance.png"); plt.show()
print(f"Elastic Net uses {(coef != 0).sum()} of {len(coef)} features (the rest were set to 0)")
""")

code(r"""
# Which feature TYPE matters most? Sum the tree importances by type
def feature_type(f):
    return f.split("_")[1] if "_" in f else ("cycle" if f == "cycle" else "raw")

for name, model in [("Random Forest", rf), ("XGBoost", xgb)]:
    imp = pd.Series(model.named_steps["model"].feature_importances_, index=FEATURES)
    print(name, (imp.groupby(feature_type).sum().sort_values(ascending=False) * 100).round(1).astype(str).add(" %").to_dict())
""")

md("### Does feature engineering help? (raw sensors only vs all features)")
code(r"""
from sklearn.base import clone
RAW = ["cycle"] + SENSORS
compare = []
for name, model in [("Random Forest", rf), ("XGBoost", xgb), ("Elastic Net", enet)]:
    m_raw = clone(model).fit(X_train[RAW], y_train)
    rmse_raw = np.sqrt(mean_squared_error(y_val, np.clip(m_raw.predict(X_val[RAW]), 0, RUL_CAP)))
    compare.append({"Model": name, "RMSE raw sensors only": rmse_raw,
                    "RMSE with engineered features": val_table.loc[name, "RMSE"]})
compare = pd.DataFrame(compare).set_index("Model").round(2)
compare.to_csv(f"{RESULTS}/raw_vs_engineered.csv")
compare
""")

# ---------------------------------------------------------------- 16
md(r"""
## 16. Early vs Late Degradation
* **Early:** true RUL > 100 cycles
* **Late:** true RUL ≤ 100 cycles

(The true uncapped RUL decides the stage. Errors are measured against the capped target the models learned.)
""")
code(r"""
stage = np.where(X_val["RUL_raw"] > 100, "early", "late")
rows = []
for m in ["Baseline"] + models:
    for st in ["early", "late"]:
        mask = stage == st
        rows.append({"Model": m, "Stage": st, "Rows": mask.sum(),
                     "RMSE": np.sqrt(mean_squared_error(y_val[mask], val_preds[m][mask])),
                     "MAE": mean_absolute_error(y_val[mask], val_preds[m][mask])})
stage_table = pd.DataFrame(rows).round(2)
stage_table.to_csv(f"{RESULTS}/early_vs_late.csv", index=False)
stage_table.pivot(index="Model", columns="Stage", values=["Rows", "RMSE", "MAE"])
""")

# ---------------------------------------------------------------- 17
md(r"""
## 17. Official NASA Test Set
* The best model (chosen on validation) is **re-trained on all 100 training engines** with the same settings.
* Test features are built for each test engine from **its own history only**, with the same function.
* We predict once per engine, at its **last available cycle**, and compare with `RUL_FD001.txt` (capped at 125).
* Nothing is fitted or tuned on the test set.
""")
code(r"""
best = {"Random Forest": rf, "XGBoost": xgb, "Elastic Net": enet}[best_model_name]
final_model = clone(best).fit(train_feat[FEATURES], train_feat["RUL"])
joblib.dump(final_model, f"{RESULTS}/final_model_{best_model_name.replace(' ', '_').lower()}.joblib")

test_feat = make_features(test)
test_last = test_feat.groupby("unit").tail(1).sort_values("unit")          # last cycle of each engine
test_last = test_last.merge(test_rul, on="unit")
assert len(test_last) == 100

true_rul = test_last["true_rul"].clip(upper=RUL_CAP)
pred_rul = np.clip(final_model.predict(test_last[FEATURES]).astype(float), 0, RUL_CAP)

test_result = evaluate(best_model_name, true_rul, pred_rul)
print("OFFICIAL TEST RESULT:", test_result)

per_engine = pd.DataFrame({"engine": test_last["unit"].values, "true_RUL": true_rul.values,
                           "predicted_RUL": pred_rul.round(1)})
per_engine["error"] = (per_engine["predicted_RUL"] - per_engine["true_RUL"]).round(1)
per_engine.to_csv(f"{RESULTS}/test_predictions.csv", index=False)
per_engine.head(15)
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].scatter(true_rul, pred_rul, color=COLORS[best_model_name])
ax[0].plot([0, RUL_CAP], [0, RUL_CAP], "k--", lw=1)
ax[0].set(title=f"Test set: {best_model_name}", xlabel="True RUL", ylabel="Predicted RUL")
order = per_engine.sort_values("true_RUL")
ax[1].plot(range(100), order["true_RUL"], color="black", lw=2, label="true")
ax[1].scatter(range(100), order["predicted_RUL"], s=12, color=COLORS[best_model_name], label="predicted")
ax[1].set(title="Per-engine predictions (sorted by true RUL)", xlabel="Test engine", ylabel="RUL"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{RESULTS}/test_results.png"); plt.show()
""")

code(r"""
# For reference only (NOT used to choose the model): every model on the test set
test_rows = []
for name, model in [("Random Forest", rf), ("XGBoost", xgb), ("Elastic Net", enet)]:
    m = clone(model).fit(train_feat[FEATURES], train_feat["RUL"])
    test_rows.append(evaluate(name, true_rul, np.clip(m.predict(test_last[FEATURES]), 0, RUL_CAP)))
test_rows.append(evaluate("Baseline", true_rul, np.full(100, train_feat["RUL"].mean())))
test_table = pd.DataFrame(test_rows).set_index("Model").round(2)
test_table.to_csv(f"{RESULTS}/test_results_all_models.csv")
test_table
""")

# ---------------------------------------------------------------- 18-20
md(r"""
## 18. Results and Discussion

*(Numbers below come from this notebook's run with seed 42.)*

| Model | Validation RMSE | Validation MAE | Validation R² | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (mean) | 41.72 | 36.93 | 0.00 | 41.94 | 34.83 | −0.10 |
| Random Forest | 10.34 | 6.77 | 0.94 | 11.44 | 8.27 | 0.92 |
| **XGBoost (selected)** | **9.60** | **6.05** | **0.95** | **10.14** | **7.40** | **0.94** |
| Elastic Net | 15.04 | 12.01 | 0.87 | 16.13 | 13.14 | 0.84 |

* **Best model:** XGBoost had the lowest validation error, so it was chosen *before* looking at the test set. On the 100
  official test engines it is off by **7.4 cycles on average (MAE)**, with RMSE 10.14. Random Forest is close behind (about
  1 cycle worse), so XGBoost is only *slightly* better. Elastic Net is clearly weaker because the sensor–RUL relationship
  is not a straight line.
* **Do engineered features help? Yes.** Using only raw sensors and the cycle number, validation RMSE was 15.5–18.0.
  With the engineered features it dropped to 9.6–15.0, an improvement of about **38 % for XGBoost and 35 % for Random Forest**.
* **Most useful features:** the **deviation from the engine's own healthy start** (`_dev`) carries about 79–86 % of
  the tree models' importance, followed by rolling means. Slopes and rolling std add little on top of them in FD001.
* **Most useful sensors:** s11 (HPC static pressure), s15 (bypass ratio), s20/s21 (coolant bleeds) and s4 (LPT temperature)
  appear at the top of the importance plots of all three models.
* **Early vs late:** errors are larger in the late stage (XGBoost RMSE 11.45 for RUL ≤ 100 vs 7.33 for RUL > 100). The
  early stage looks "easy" only because of the cap: most early rows have target 125, which the model learns to predict for
  healthy-looking engines. In reality, sensors cannot tell how long a healthy engine will stay healthy. The hardest part
  is the transition from healthy to degrading.
* **Interpretability:** Elastic Net is the easiest to explain (it keeps 36 of 71 features, each with a + or − weight).
  The tree models are more accurate but can only be explained through feature importance.
""")
md(r"""
## 19. Limitations and Future Scope

* Only FD001 is used (one operating condition, one fault mode), and the data is simulated.
* One train/validation split with 20 validation engines, so small differences (RF vs XGBoost) are not certain.
* Rolling windows **lag** behind sudden changes, and short test histories give less reliable trends.
* The RUL cap of 125 is a modelling choice that affects all the scores.
* Small hyper-parameter grids were used to keep the notebook fast.

**Future scope:** apply the method to FD002–FD004 (several operating conditions), use repeated cross-validation, add
prediction uncertainty, and in **Phase 2** compare against deep learning models (LSTM/CNN) using the same split and target.
""")
md(r"""
## 20. Conclusion

Using classical machine learning only, the selected **XGBoost** model predicts the RUL of the 100 official test
engines with **RMSE 10.14 and MAE 7.40 cycles (R² 0.94)**, against RMSE 41.94 for the simple baseline. The answer to the
research question is **yes**: simple engineered features, especially each sensor's drift from the engine's own
healthy state, reduced the error by more than a third compared with raw sensor values.
""")

report = open("ACADEMIC_REPORT.md", encoding="utf-8").read()
md(report.replace("# Academic Documentation", "# Appendix — Academic Documentation", 1))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "predictive_maintenance_aml_phase1.ipynb")
print(f"wrote notebook with {len(cells)} cells")
