# Degradation Feature Engineering and Classical Machine Learning for Turbofan Engine Remaining Useful Life Prediction (Phase 1)

**IEEE Conference Format Report**  
*Course:* Advanced Machine Learning (AML)  
*Authors:* Sarvesh Choudhary, Arun Jangir, Keshav Chhabra, Mayank Yadav, Rohit Kumar  
*Dataset:* NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001)  
*GitHub:* [https://github.com/sarveshDhayal/Capstone_AML](https://github.com/sarveshDhayal/Capstone_AML)

---

## Abstract
Accurately estimating the Remaining Useful Life (RUL) of aircraft engines is critical for preventing in-flight failures and optimizing maintenance logistics. In Phase 1 of this investigation, we formulate RUL estimation as a supervised multivariate regression task using the benchmark NASA C-MAPSS dataset (subset FD001). We remove 7 non-informative sensors with zero variance, cap the RUL target at 125 cycles to represent nominal operating life, and extract 71 leakage-free degradation features (moving average, moving standard deviation, trend slope, and deviation from the engine's initial healthy baseline) using past telemetry only. We conduct a rigorous engine-level split (80 training engines, 20 validation engines) to prevent data leakage and evaluate three classical algorithms: Linear Regression (OLS), Decision Tree (CART), and Random Forest. On the official 100 NASA test engines, Random Forest achieves state-of-the-art classical performance with a **Test RMSE of 11.44 cycles, MAE of 8.27 cycles, and $R^2$ of 0.918**, substantially outperforming the mean baseline (RMSE 41.94 cycles). Engineered features reduce Random Forest validation RMSE by **35.3%** compared to raw sensor telemetry. Feature importance analysis confirms that deviation from the healthy baseline ($\Delta s$) accounts for approximately 85% of total predictive power, with high-pressure compressor outlet static pressure drift (`s11_dev`) being the dominant prognostic indicator.

**Keywords:** Prognostics and Health Management (PHM), Remaining Useful Life (RUL), Feature Engineering, Random Forest, Decision Tree, C-MAPSS FD001.

---

## I. Introduction
Gas turbine engines undergo continuous thermomechanical wear during operation. Early detection of subsystem degradation and accurate prediction of Remaining Useful Life (RUL)---the number of operational flight cycles before the engine reaches terminal failure---allows airlines to transition from costly reactive servicing to proactive condition-based maintenance.

The NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset provides realistic run-to-failure multivariate sensor trajectories. However, raw sensor measurements are characterized by high-frequency instrumentation noise and non-stationary operating trends. A fundamental challenge in Prognostics and Health Management (PHM) is transforming raw, noisy multi-channel signals into domain-meaningful degradation indicators that capture wear accumulation without future data leakage.

In this paper (Phase 1), we systematically explore classical machine learning techniques paired with structured degradation feature engineering:
1. We formulate a piecewise linear RUL target capped at 125 cycles to address the early-life degradation insensitivity of turbofan sensors.
2. We engineer 71 leakage-free features capturing temporal smoothing, sensor stability, rate of wear (slope), and drift from each engine's unique healthy baseline ($\Delta s$).
3. We perform a strict engine-level train/validation split (80/20) and 3-fold GroupKFold cross-validation, guaranteeing zero cross-engine data leakage.
4. We tune and benchmark Linear Regression, Cost-Complexity Pruned Decision Trees (CART), and Random Forest against the official 100 NASA test engines.

---

## II. Dataset & Exploratory Data Analysis

### A. C-MAPSS FD001 Characteristics
The FD001 dataset models degradation of a high-pressure compressor (HPC) under sea-level static flight conditions:
- **Training Set:** 100 engines (20,631 total cycles), all monitored continuously until functional failure.
- **Test Set:** 100 engines whose time series are truncated at an arbitrary operational cycle prior to failure.
- **Ground Truth ($RUL_{true}$):** Verified cycle counts provided in `RUL_FD001.txt`.
- **Variables (26):** Unit ID, cycle, 3 operational flight settings, and 21 thermodynamic sensors.

### B. Variance Thresholding & Flat Sensor Pruning
Telemetry inspection reveals that 7 sensors exhibit zero variance ($\sigma \approx 0$) across all units and cycles:
$$\{s_1, s_5, s_6, s_{10}, s_{16}, s_{18}, s_{19}\} \rightarrow \text{Pruned}$$
Because FD001 simulates a single operating condition, the 3 operational settings are constant and discarded, leaving 14 informative dynamic sensors: $s_2, s_3, s_4, s_7, s_8, s_9, s_{11}, s_{12}, s_{13}, s_{14}, s_{15}, s_{17}, s_{20}, s_{21}$.

### C. Engine Lifespans
Training engine lifespans range from 128 cycles (minimum) to 362 cycles (maximum), with a mean lifespan of $\approx 206$ cycles. Pearson correlation demonstrates that HPC outlet static pressure ($s_{11}$), LPT outlet temperature ($s_4$), and bypass ratio ($s_{15}$) have the strongest linear correlation with degradation ($|r| \in [0.65, 0.70]$).

---

## III. Methodology & Feature Engineering

### A. Piecewise Linear Target Formulation
In physical engines, wear propagation is negligible during the initial 50--100 cycles; the engine behaves as new. A model trained on uncapped linear targets would be forced to predict differing values (e.g., 300 vs 200 cycles) on indistinguishable sensor profiles. We apply an upper cap at $RUL_{cap} = 125$ cycles:
$$y_{i,t} = \min\left(125, \text{cycle}_{i,\max} - t\right)$$

### B. Engine-Level Data Partitioning
To prevent optimistic bias, we split the 100 training engines into 80 training units (16,561 records) and 20 validation units (4,070 records) using `GroupShuffleSplit` with seed 42. No engine ID appears in both training and validation sets.

### C. Degradation Feature Engineering (71 Features)
For each of the 14 informative sensors, we compute four features using a 10-cycle backward rolling window ($\tau \in [t-9, t]$) on the same engine:
1. **Moving Average ($\mu_s$):** Dampens high-frequency sensor noise:
   $$\mu_s(t) = \frac{1}{10}\sum_{k=0}^{9} s(t-k)$$
2. **Moving Standard Deviation ($\sigma_s$):** Detects sensor instability and turbulent flutter:
   $$\sigma_s(t) = \sqrt{\frac{1}{9}\sum_{k=0}^{9}\left(s(t-k) - \mu_s(t)\right)^2}$$
3. **Trend Slope ($\beta_s$):** Quantifies rate of wear acceleration via Ordinary Least Squares:
   $$\beta_s(t) = \frac{\sum_{k=0}^9 (t-k - \bar{t})(s(t-k) - \mu_s(t))}{\sum_{k=0}^9 (t-k - \bar{t})^2}$$
4. **Baseline Drift ($\Delta s$ or `_dev`):** Difference between current moving average and the engine's initial healthy baseline (mean of cycles $1 \le t \le 10$):
   $$\Delta s(t) = \mu_s(t) - \frac{1}{10}\sum_{\tau=1}^{10} s(\tau)$$

Together with the current operational cycle and 14 raw sensors, this produces $1 + 14 + (14 \times 4) = 71$ engineered features.

---

## IV. Machine Learning Models & Tuning

### A. Linear Regression (OLS)
Acts as the simplest baseline:
$$\hat{y} = w_0 + \sum_{j=1}^{71} w_j x_j$$
Features are standardized using `StandardScaler` fitted on the training split.

### B. Decision Tree Regressor (CART)
Captures non-linear piecewise degradation curves. Hyperparameters are tuned via 3-fold `GroupKFold` cross-validation:
- `max_depth`: Optimal at depth 7.
- `min_samples_leaf`: Optimal at 30 samples.
- Post-pruning via cost-complexity pruning ($\alpha = 0.001$).

### C. Random Forest Regressor
An ensemble of $B=150$ bootstrap-aggregated (bagged) decision trees. Each tree is trained on a random subset of training engines and random feature subspaces ($\sqrt{p}$), dramatically reducing model variance:
$$\hat{y}_{RF} = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$$

---

## V. Experimental Results and Discussion

### A. Comparative Model Benchmark
Performance benchmark across all Phase 1 models on validation and official test sets:

| Model | Val RMSE | Val MAE | Val $R^2$ | Test RMSE | Test MAE | Test $R^2$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Mean Predictor)** | 41.72 | 36.93 | 0.000 | 41.94 | 34.83 | -0.095 |
| **Linear Regression** | 15.29 | 11.94 | 0.866 | 15.85 | 12.84 | 0.844 |
| **Decision Tree** | 14.13 | 9.18 | 0.886 | 14.96 | 10.80 | 0.861 |
| **Random Forest (Best AML)** | **10.34** | **6.77** | **0.939** | **11.44** | **8.27** | **0.918** |

Random Forest achieves the superior score on the 100 official test engines: **RMSE 11.44 cycles, MAE 8.27 cycles, and $R^2$ 0.918**. Linear regression struggles because it cannot fit the piecewise flat-then-decay curvature. Single decision trees suffer from high variance, whereas Random Forest's 150-tree bagging eliminates tree instability.

### B. Quantitative Effect of Feature Engineering
Comparing raw sensor models against engineered features:
- **Linear Regression:** Val RMSE reduced from 17.37 to 15.29 (**12.0% improvement**).
- **Decision Tree:** Val RMSE reduced from 17.73 to 14.13 (**20.3% improvement**).
- **Random Forest:** Val RMSE reduced from 15.98 to **10.34** (**35.3% improvement**).

### C. Feature Importance Analysis
Gini impurity reduction across the 150 trees reveals that:
- **Feature Type:** Baseline drift features (`_dev`) account for **84.8%** of total model importance, followed by moving averages (11.2%).
- **Individual Sensor:** HPC outlet static pressure drift (`s11_dev`) alone contributes **74.1%** of total importance.
- **Secondary Features:** LPT outlet temperature drift (`s4_dev`), bypass ratio drift (`s15_dev`), and core speed drift (`s9_dev`).

### D. Early vs Late Life Degradation Errors
- **Early Stage ($RUL > 100$):** Random Forest RMSE = 8.01 cycles.
- **Late Stage ($RUL \le 100$):** Random Forest RMSE = 12.26 cycles.

---

## VI. Conclusion
In Phase 1, we proved that domain feature engineering combined with ensemble machine learning provides an effective prognostic framework for turbofan engines. By constructing baseline drift indicators ($\Delta s$) and tuning Random Forests with strict engine-level cross-validation, we achieved an official test RMSE of 11.44 cycles, outperforming raw sensor models by 35.3%. These results establish the foundational classical benchmark for comparison with Phase 2 deep sequence models.

---

## References
1. A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure simulation," in *Proc. IEEE PHM*, 2008, pp. 1--9.
2. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5--32, 2001.
3. L. Breiman, J. Friedman, C. J. Stone, and R. A. Olshen, *Classification and Regression Trees*. CRC press, 1984.

