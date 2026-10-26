"""Simple web app: predict the Remaining Useful Life (RUL) of an engine with the best model.

Run from the project folder:
    streamlit run app/app.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
from utils import load_raw, MODELS, RESULTS, FIGURES, PROCESSED_DATA, RUL_CAP  # noqa: E402

st.set_page_config(page_title="Engine RUL Predictor", page_icon="✈️", layout="wide")


@st.cache_resource
def load_everything():
    features = (PROCESSED_DATA / "feature_list.txt").read_text().split()
    test = pd.read_csv(PROCESSED_DATA / "test_features.csv")
    val_table = pd.read_csv(RESULTS / "final_validation_results.csv", index_col=0)
    best = val_table.drop("Baseline")["RMSE"].idxmin()          # best model = lowest validation RMSE
    model = joblib.load(MODELS / f"{best.lower().replace(' ', '_')}.joblib")

    # Example sensor values taken from the training engines
    train = load_raw("train_FD001.txt")
    train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    sensors = [f for f in features if "_" not in f and f != "cycle"]
    healthy_start = train[train["cycle"] <= 10][sensors].mean()   # typical healthy value of each sensor
    presets = {}
    for name, rows in [("🟢 Healthy engine", train[train["RUL"] > 150]),
                       ("🟠 Mid-life engine", train[train["RUL"].between(50, 70)]),
                       ("🔴 Engine near failure", train[train["RUL"] <= 10])]:
        presets[name] = {"cycle": int(rows["cycle"].median()), **rows[sensors].mean().to_dict()}
    return features, test, best, val_table.loc[best, "RMSE"], model, sensors, healthy_start, presets, load_raw("test_FD001.txt")


FEATURES, test, BEST, BEST_RMSE, model, SENSORS, healthy_start, presets, raw_test = load_everything()


def predict(X):
    return float(min(max(model.predict(X[FEATURES])[0], 0), RUL_CAP))


def show_result(predicted):
    if predicted < 30:
        st.error(f"### Predicted RUL: {predicted:.0f} cycles\n🔴 Critical — schedule maintenance now")
    elif predicted < 80:
        st.warning(f"### Predicted RUL: {predicted:.0f} cycles\n🟠 Wearing out — plan maintenance")
    else:
        st.success(f"### Predicted RUL: {predicted:.0f} cycles\n🟢 Healthy")


st.title("✈️ Turbofan Engine — Remaining Useful Life Predictor")
st.caption(f"NASA C-MAPSS FD001 · model used: **{BEST}** (best model, validation RMSE {BEST_RMSE:.2f} cycles) · "
           f"RUL is capped at {RUL_CAP} cycles (= healthy)")

tab_input, tab_engine, tab_compare = st.tabs(["🧮 Enter sensor values", "🔧 NASA test engine", "📊 Compare models"])

# ---------------------------------------------------------------- 1. manual input
with tab_input:
    st.markdown("Enter the **current cycle** and **sensor readings** of an engine, then press **Predict**. "
                "Choose an example to fill the boxes with realistic values.")
    preset = st.selectbox("Start from an example", list(presets))
    values = presets[preset]

    with st.form("sensor_form"):
        cycle = st.number_input("Current cycle (engine age)", 1, 400, values["cycle"], key=f"{preset}_cycle")
        cols = st.columns(4)
        readings = {s: cols[i % 4].number_input(s, value=round(values[s], 4), format="%.4f", key=f"{preset}_{s}")
                    for i, s in enumerate(SENSORS)}
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        # Build the 71 model features from one reading:
        # moving average = current value, std and trend = 0 (no history), deviation = value - typical healthy value
        row = {"cycle": cycle}
        for s, v in readings.items():
            row.update({s: v, f"{s}_mean": v, f"{s}_std": 0.0, f"{s}_slope": 0.0, f"{s}_dev": v - healthy_start[s]})
        show_result(predict(pd.DataFrame([row])))
        st.caption("With a single reading there is no history, so std and trend are set to 0. "
                   "The most important features (drift from healthy values) are still computed exactly.")

# ---------------------------------------------------------------- 2. official test engine
with tab_engine:
    engine = st.slider("Choose one of the 100 NASA test engines", 1, 100, 1)
    row = test[test["unit"] == engine]
    predicted, true_rul = predict(row), float(row["RUL"].iloc[0])

    show_result(predicted)
    c1, c2, c3 = st.columns(3)
    c1.metric("True RUL", f"{true_rul:.0f} cycles")
    c2.metric("Error", f"{predicted - true_rul:+.0f} cycles")
    c3.metric("Cycles recorded", int(row["cycle"].iloc[0]))

    st.markdown("#### Sensor history of this engine")
    chosen = st.multiselect("Sensors", SENSORS, default=["s4", "s11"])
    history = raw_test[raw_test["unit"] == engine].set_index("cycle")[chosen]
    # standardise so sensors with different units fit on one chart
    st.line_chart((history - history.mean()) / history.std())
    st.caption("Values are standardised (z-score). Rising s2/s3/s4/s11/s15 or falling s7/s12/s20/s21 = degradation.")

# ---------------------------------------------------------------- 3. comparison
with tab_compare:
    st.markdown("#### Official test set results (100 engines)")
    table = pd.read_csv(RESULTS / "final_test_results.csv", index_col=0)
    st.dataframe(table, width="stretch")
    st.bar_chart(table["RMSE"], y_label="Test RMSE (lower is better)")
    st.image(str(FIGURES / "06_feature_engineering_effect.png"), caption="Engineered features vs raw sensors")
    st.image(str(FIGURES / "06_best_model_test.png"), caption=f"{BEST} predictions on the 100 test engines")
