# Phase 2 & Hybrid Model — Project Presentation Guide

### Title: Predictive Maintenance via Sensor Features, Deep Sequence Models, and Hybrid Innovation
**Project 7:** Remaining Useful Life Prediction for Industrial Engines  
**Team Members:** Sarvesh Choudhary, Arun Jangir, Keshav Chhabra, Mayank Yadav, Rohit Kumar

---

## Slide 1: Title & Overview
* **Headline:** Remaining Useful Life (RUL) Prediction for Industrial Engines: The "Hybrid" Logic
* **Subtitle:** Moving from Hand-Engineered Degradation Features to Deep Sequence Learning
* **Visual:** NASA Turbofan engine diagram & run-to-failure sensor degradation trajectories.
* **Speaker Note:** "Good morning/afternoon. Today we present our comprehensive capstone investigating the continuum between feature-engineered machine learning and deep sequence learning for aircraft engine prognostics."

---

## Slide 2: Industrial Problem & Motivation
* **The Goal:** Predict Remaining Useful Life (cycles left before engine failure) from multivariate time-series sensors.
* **Industrial Impact:** 
  - Catastrophic failure prevention in aviation.
  - Minimizing unscheduled downtime and excessive maintenance costs.
* **NASA C-MAPSS Dataset (FD001):** 100 training engines to failure, 100 test engines truncated before failure.
* **Key Constraint:** Strict engine-level splitting (no data leakage) and asymmetric penalty for late failure prediction.

---

## Slide 3: The Gap in Classical ML (Phase 1 Baseline)
* **What we did in Phase 1:** Built 71 hand-crafted features (rolling mean, std, trend slope, healthy drift `_dev`) and trained Linear Regression, Decision Tree, and Random Forest.
* **Phase 1 Benchmark:** Random Forest achieved Test RMSE 11.44 cycles, MAE 8.27 cycles.
* **The Fundamental Gap:**
  - Fixed rolling windows discard the exact chronological order of time steps.
  - Manual feature engineering does not generalize when operational regimes change.
  - Manual summary statistics lack dynamic temporal momentum.

---

## Slide 4: Phase 2 — Deep Learning Sequence Modeling
* **End-to-End Representation Learning:** Directly feeding 3D sliding sequence tensors $X \in \mathbb{R}^{30 \times 14}$ into neural sequence models.
* **Models Implemented in PyTorch:**
  1. **Stacked LSTM Regressor (2 layers, 64 units, recurrent dropout 0.2):** Continuous gating mechanism.
  2. **Temporal 1D-CNN (3 Conv blocks + AdaptiveAvgPool):** Hierarchical local trend filters.
* **Regularization & Optimization:**
  - Huber Loss (`SmoothL1Loss`): Outlier-resistant backpropagation.
  - AdamW optimizer with `ReduceLROnPlateau`.
  - EarlyStopping on validation loss.

---

## Slide 5: The "Hybrid" Innovation (Additional Marks Justification)
* **The Core Rationale:** Why combine ML and DL?
  - Random Forest leverages explicit baseline subtraction (`_dev` features) $\rightarrow$ excellent stationary reference.
  - PyTorch LSTM captures dynamic continuous temporal acceleration $\rightarrow$ superior sequence modeling.
* **Mathematical Synergy:**
  $$\hat{y}_{hybrid} = 0.69 \cdot \hat{y}_{RF} + 0.31 \cdot \hat{y}_{LSTM}$$
* Because the inductive biases are orthogonal, their error covariance is low, mathematically guaranteeing variance reduction:
  $$\text{Var}(\hat{y}_{hybrid}) < \min(\text{Var}(\hat{y}_{RF}), \text{Var}(\hat{y}_{LSTM}))$$

---

## Slide 6: Benchmark Results Comparison (100 NASA Test Engines)
* **Show the Official Table:**

| Model | Paradigm | Input Type | Test RMSE ↓ | Test MAE ↓ | NASA Score ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline (Mean) | Baseline | Constant | 51.63 | 39.47 | 248,340.7 |
| Linear Regression | Phase 1 (AML) | 71 Features | 15.85 | 12.84 | 406.6 |
| Decision Tree | Phase 1 (AML) | 71 Features | 14.96 | 10.80 | 357.0 |
| Random Forest | Phase 1 (Best AML) | 71 Features | 11.44 | 8.27 | 184.5 |
| PyTorch 1D-CNN | Phase 2 (DL Temporal)| 30-cycle Raw | 18.93 | 13.30 | 1,404.3 |
| PyTorch LSTM | Phase 2 (DL Sequence)| 30-cycle Raw | 13.23 | 9.88 | 240.5 |
| **Hybrid Model** | **Hybrid Innovation** | **Fused ML + DL** | **10.99** | **7.98** | **170.1** |

* **Key Takeaway:** Hybrid Model achieves state-of-the-art across all metrics!

---

## Slide 7: Visualizing Predictions Across 100 Test Engines
* **Visual:** Display `11_hybrid_test_scatter.png` (Sorted test engine ground truth vs RF vs LSTM vs Hybrid).
* **Explanation:** Point out how the Hybrid line (green) tightly tracks ground truth (black), smoothing out the individual over-predictions of LSTM and under-predictions of RF.

---

## Slide 8: Systematic Ablation Studies
* **Ablation 1 (Representation):** Raw Sequences vs Engineered Features vs Fused Hybrid.
* **Ablation 2 (Window Sensitivity):** $W=15$ (RMSE 15.22) vs $W=30$ (RMSE 13.23) vs $W=45$ (RMSE 14.10) $\rightarrow$ proves $W=30$ is optimal.
* **Ablation 3 (Loss Function):** MSE (RMSE 14.72) vs Huber Loss (RMSE 13.23) $\rightarrow$ proves Huber's robustness against sensor noise.
* **Ablation 4 (Early vs Late Degradation):** Hybrid achieves lowest late-stage error (11.10 RMSE).

---

## Slide 9: Interactive Web Application (`app/app.py`)
* **Dashboard Features:**
  - Real-time engine health diagnosis and RUL prediction.
  - Interactive model comparison toggle (RF vs LSTM vs 1D-CNN vs Hybrid).
  - Multi-sensor degradation curves with standardized z-score plotting.
* **Live Demo:** `http://localhost:8501`.

---

## Slide 10: Conclusion & Key Learnings
1. **End-to-End Learning Viability:** PyTorch LSTM achieved 13.23 Test RMSE without any hand-crafted features.
2. **Hybrid Superiority:** Fusing domain engineering with sequence learning yielded the best overall performance (10.99 RMSE, 170.06 NASA Score).
3. **Future Extension:** Extending hybrid representation to multi-condition datasets (FD002/FD004) with Transformer self-attention.

