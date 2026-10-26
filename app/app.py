"""Simple web app: predict the Remaining Useful Life (RUL) of a NASA test engine.

Run from the project folder (after running notebooks 02-05 once):
    streamlit run app/app.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
from utils import load_features, load_raw, MODELS, RESULTS, FIGURES, RUL_CAP  # noqa: E402

st.set_page_config(page_title="Engine RUL Predictor", page_icon="✈️", layout="wide")


@st.cache_resource
def load_everything():
    _, test, features = load_features()
    models = {name: joblib.load(MODELS / f"{name.lower().replace(' ', '_')}.joblib")
              for name in ["XGBoost", "Random Forest", "Elastic Net"]}
    return test, features, models, load_raw("test_FD001.txt")


test, FEATURES, models, raw_test = load_everything()

st.title("✈️ Turbofan Engine — Remaining Useful Life Predictor")
st.caption("NASA C-MAPSS FD001 · 100 official test engines · classical ML (Phase 1)")

model_name = st.sidebar.selectbox("Model", list(models))
engine = st.sidebar.slider("Test engine", 1, 100, 1)
st.sidebar.markdown(f"RUL is capped at **{RUL_CAP}** cycles (= healthy).")

tab_predict, tab_compare = st.tabs(["🔧 Predict an engine", "📊 Compare models"])

with tab_predict:
    row = test[test["unit"] == engine]
    predicted = float(min(max(models[model_name].predict(row[FEATURES])[0], 0), RUL_CAP))
    true_rul = float(row["RUL"].iloc[0])

    if predicted < 30:
        status = "🔴 Critical — schedule maintenance now"
    elif predicted < 80:
        status = "🟠 Wearing out — plan maintenance"
    else:
        status = "🟢 Healthy"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted RUL", f"{predicted:.0f} cycles")
    c2.metric("True RUL", f"{true_rul:.0f} cycles")
    c3.metric("Error", f"{predicted - true_rul:+.0f} cycles")
    c4.metric("Cycles recorded", int(row["cycle"].iloc[0]))
    st.subheader(status)

    st.markdown("#### Sensor history of this engine")
    sensors = st.multiselect("Sensors", ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s20", "s21"],
                             default=["s4", "s11"])
    history = raw_test[raw_test["unit"] == engine].set_index("cycle")[sensors]
    # standardise so sensors with different units fit on one chart
    st.line_chart((history - history.mean()) / history.std())
    st.caption("Values are standardised (z-score). Rising s2/s3/s4/s11/s15 or falling s7/s12/s20/s21 = degradation.")

with tab_compare:
    st.markdown("#### Official test set results (100 engines)")
    table = pd.read_csv(RESULTS / "final_test_results.csv", index_col=0)
    st.dataframe(table, width="stretch")
    st.bar_chart(table["RMSE"], y_label="Test RMSE (lower is better)")
    st.image(str(FIGURES / "06_feature_engineering_effect.png"), caption="Engineered features vs raw sensors")
    st.image(str(FIGURES / "05_test.png"), caption="XGBoost predictions on the 100 test engines")
