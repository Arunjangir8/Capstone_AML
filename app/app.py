"""Interactive web app: predict Remaining Useful Life (RUL) of aircraft turbofan engines.

Supports:
- Phase 1: Classical Machine Learning (Random Forest with 71 engineered degradation features)
- Phase 2: Deep Learning Sequence Models (PyTorch LSTM & Temporal 1D-CNN)
- Hybrid Innovation: Fused ML + DL Model (State of the Art)

Run from the project root:
    streamlit run app/app.py
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# Add notebooks directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
from utils import (
    load_raw, MODELS, RESULTS, FIGURES, PROCESSED_DATA, RUL_CAP, USEFUL_SENSORS, SEQ_LEN
)

st.set_page_config(page_title="Turbofan RUL Predictor (AML & DL Hybrid)", layout="wide")


# --- Define PyTorch Architectures for App ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


class TemporalCNNRegressor(nn.Module):
    def __init__(self, in_channels=14, dropout=0.2):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        feats = self.conv_net(x).squeeze(-1)
        return self.fc(feats).squeeze(-1)


@st.cache_resource
def load_all_artifacts():
    """Load cached classical, deep learning, and hybrid artifacts."""
    features = (PROCESSED_DATA / "feature_list.txt").read_text().split()
    test_features = pd.read_csv(PROCESSED_DATA / "test_features.csv")
    rf_model = joblib.load(MODELS / "random_forest.joblib")

    scaler = joblib.load(MODELS / "dl_sensor_scaler.joblib")
    
    lstm_model = LSTMRegressor(input_dim=14, hidden_dim=64, num_layers=2)
    lstm_model.load_state_dict(torch.load(MODELS / "lstm_model.pt", map_location=torch.device("cpu")))
    lstm_model.eval()

    cnn_model = TemporalCNNRegressor(in_channels=14)
    cnn_model.load_state_dict(torch.load(MODELS / "cnn_model.pt", map_location=torch.device("cpu")))
    cnn_model.eval()

    X_test_seq = np.load(PROCESSED_DATA / "X_test_seq.npy")

    train_raw = load_raw("train_FD001.txt")
    train_raw["RUL"] = train_raw.groupby("unit")["cycle"].transform("max") - train_raw["cycle"]
    sensors = [f for f in features if "_" not in f and f != "cycle"]
    healthy_start = train_raw[train_raw["cycle"] <= 10][sensors].mean()

    presets = {}
    for name, rows in [
        ("Healthy engine (RUL > 150)", train_raw[train_raw["RUL"] > 150]),
        ("Mid-life engine (RUL 50-70)", train_raw[train_raw["RUL"].between(50, 70)]),
        ("Engine near failure (RUL ≤ 10)", train_raw[train_raw["RUL"] <= 10])
    ]:
        presets[name] = {"cycle": int(rows["cycle"].median()), **rows[sensors].mean().to_dict()}

    raw_test = load_raw("test_FD001.txt")
    comparison_table = pd.read_csv(RESULTS / "final_comparison_phase1_vs_phase2.csv", index_col=0)

    return (
        features, test_features, rf_model, lstm_model, cnn_model, scaler,
        X_test_seq, sensors, healthy_start, presets, raw_test, comparison_table
    )


(
    FEATURES, test_features, rf_model, lstm_model, cnn_model, dl_scaler,
    X_test_seq, SENSORS, healthy_start, presets, raw_test, comparison_table
) = load_all_artifacts()


def show_result_card(predicted, model_name):
    """Display color-coded status alert."""
    if predicted < 30:
        st.error(f"### {model_name} Predicted RUL: {predicted:.1f} cycles\n**CRITICAL ALERT** — Schedule replacement immediately")
    elif predicted < 80:
        st.warning(f"### {model_name} Predicted RUL: {predicted:.1f} cycles\n**WARNING** — Degradation detected, plan maintenance")
    else:
        st.success(f"### {model_name} Predicted RUL: {predicted:.1f} cycles\n**HEALTHY** — Normal operating regime")


st.title("Turbofan Engine — Remaining Useful Life (RUL) Predictor")
st.markdown(
    "**Project 7: Predictive Maintenance via Sensor Features, Sequence Models & Hybrid Innovation**  \n"
    "*Comparison of Classical Machine Learning, PyTorch Deep Sequence Networks, and Fused Hybrid Modeling*"
)

# Sidebar model selection
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose Active Prediction Model",
    [
        "Hybrid Model (ML + DL Fusion — Best)",
        "Random Forest (Phase 1 Best ML)",
        "PyTorch LSTM (Phase 2 Sequence)",
        "PyTorch 1D-CNN (Phase 2 Temporal)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "### Benchmark Snapshot (100 Engines)\n"
    "- 🏆 **Hybrid Model:** **RMSE 10.99** | **MAE 7.98** | **NASA 170.1**\n"
    "- **Random Forest:** RMSE 11.44 | MAE 8.27 | NASA 184.5\n"
    "- **PyTorch LSTM:** RMSE 13.23 | MAE 9.88 | NASA 240.5\n"
    "- **PyTorch 1D-CNN:** RMSE 18.93 | MAE 13.30 | NASA 1404.3\n"
)

tab_engine, tab_input, tab_compare = st.tabs(["NASA Test Engines", "Single Sensor Reading Input", "Comprehensive Benchmark & Ablations"])

# ---------------------------------------------------------------- 1. NASA test engine
with tab_engine:
    st.subheader("Official NASA Test Set (100 Engines)")
    engine_id = st.slider("Select NASA Test Engine ID", 1, 100, 1)
    
    row_feat = test_features[test_features["unit"] == engine_id]
    true_rul = float(row_feat["RUL"].iloc[0])
    engine_seq = torch.from_numpy(X_test_seq[engine_id - 1: engine_id])

    # Compute predictions from all models
    pred_rf = float(min(max(rf_model.predict(row_feat[FEATURES])[0], 0), RUL_CAP))
    with torch.no_grad():
        pred_lstm = float(min(max(lstm_model(engine_seq).item(), 0), RUL_CAP))
        pred_cnn = float(min(max(cnn_model(engine_seq).item(), 0), RUL_CAP))
    
    pred_hybrid = float(min(max(0.69 * pred_rf + 0.31 * pred_lstm, 0), RUL_CAP))

    active_pred = (
        pred_hybrid if "Hybrid" in selected_model_name
        else (pred_rf if "Random Forest" in selected_model_name
              else (pred_lstm if "LSTM" in selected_model_name else pred_cnn))
    )
    show_result_card(active_pred, selected_model_name)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ground Truth RUL", f"{true_rul:.0f} cycles")
    c2.metric("🏆 Hybrid Model (Best)", f"{pred_hybrid:.1f} cycles", f"{pred_hybrid - true_rul:+.1f}")
    c3.metric("Random Forest (ML)", f"{pred_rf:.1f} cycles", f"{pred_rf - true_rul:+.1f}")
    c4.metric("PyTorch LSTM (DL)", f"{pred_lstm:.1f} cycles", f"{pred_lstm - true_rul:+.1f}")

    st.markdown("#### Multivariate Sensor Trajectory (Engine Lifetime)")
    chosen = st.multiselect("Choose Sensor Channels to Plot", SENSORS, default=["s4", "s11", "s15", "s21"])
    history = raw_test[raw_test["unit"] == engine_id].set_index("cycle")[chosen]
    st.line_chart((history - history.mean()) / history.std())
    st.caption("Standardized sensor history (z-score). Upward drift in s4/s11/s15 indicates thermal and pressure degradation.")

# ---------------------------------------------------------------- 2. manual input
with tab_input:
    st.subheader("Simulate Sensor Readings (Cycle-Level Prediction)")
    st.markdown("Select a degradation preset or enter sensor readings:")
    preset_choice = st.selectbox("Load Engine Condition Preset", list(presets.keys()))
    preset_vals = presets[preset_choice]

    with st.form("manual_sensor_form"):
        cycle_val = st.number_input("Current Engine Operating Cycle", 1, 500, preset_vals["cycle"])
        cols = st.columns(4)
        input_readings = {
            s: cols[i % 4].number_input(s, value=round(preset_vals[s], 4), format="%.4f")
            for i, s in enumerate(SENSORS)
        }
        submit_btn = st.form_submit_button("Predict RUL", type="primary")

    if submit_btn:
        row_dict = {"cycle": cycle_val}
        for s, v in input_readings.items():
            row_dict.update({
                s: v, f"{s}_mean": v, f"{s}_std": 0.0, f"{s}_slope": 0.0,
                f"{s}_dev": v - healthy_start[s]
            })
        pred_single_rf = float(min(max(rf_model.predict(pd.DataFrame([row_dict])[FEATURES])[0], 0), RUL_CAP))
        show_result_card(pred_single_rf, "Random Forest")

# ---------------------------------------------------------------- 3. comparison
with tab_compare:
    st.subheader("Comprehensive Benchmark: Classical ML vs Deep Learning vs Hybrid")
    st.markdown(
        "Comparison of all models evaluated on the **100 official NASA test engines**."
    )
    st.dataframe(comparison_table, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if (FIGURES / "11_hybrid_test_scatter.png").exists():
            st.image(str(FIGURES / "11_hybrid_test_scatter.png"), caption="Hybrid Model vs Individual Models across 100 Test Engines")
        elif (FIGURES / "10_phase1_vs_phase2_test_comparison.png").exists():
            st.image(str(FIGURES / "10_phase1_vs_phase2_test_comparison.png"), caption="RMSE, MAE & NASA Score across all models")
            
    with col2:
        if (FIGURES / "11_ablation_model_comparison.png").exists():
            st.image(str(FIGURES / "11_ablation_model_comparison.png"), caption="Ablation Study: Error & Penalty across representations")
        elif (FIGURES / "10_phase1_vs_phase2_early_vs_late.png").exists():
            st.image(str(FIGURES / "10_phase1_vs_phase2_early_vs_late.png"), caption="Early (Healthy) vs Late (Critical) Degradation Error")

    col3, col4 = st.columns(2)
    with col3:
        if (FIGURES / "11_ablation_window_size.png").exists():
            st.image(str(FIGURES / "11_ablation_window_size.png"), caption="Ablation: Sequence Window Sensitivity W in {15, 30, 45}")
    with col4:
        if (FIGURES / "11_ablation_loss_functions.png").exists():
            st.image(str(FIGURES / "11_ablation_loss_functions.png"), caption="Ablation: MSE vs Huber Loss on Test Performance")
