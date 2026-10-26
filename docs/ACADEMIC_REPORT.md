# Academic Documentation & Technical Report

## Project 7: Remaining Useful Life Prediction for Industrial Engines
### Predictive Maintenance via Sensor Features and Machine Learning (Phase 1)

- **Course:** Advanced Machine Learning (AML)
- **Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation (Subset FD001)
- **Evaluation Benchmark:** Official NASA Test Set (100 test engines)

> [!NOTE]
> **Data & Experiment Reference:**  
> All experimental results and metrics in this report are derived directly from the executed notebooks in `notebooks/` (01 – 06) (NASA C-MAPSS FD001, seed `42`, piecewise linear RUL target capped at `125` cycles).

---

## 📑 Table of Contents
1. [Abstract](#1-abstract)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Methodology](#4-methodology)
   - [Data Ingestion & Integrity Checks](#1-data-ingestion--integrity-checks)
   - [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
   - [RUL Target Formulation](#3-rul-target-formulation)
   - [Engine-Level Splitting](#4-engine-level-splitting)
   - [Feature Engineering (71 Features)](#5-feature-engineering-71-features)
   - [Data Leakage Prevention Checks](#6-data-leakage-prevention-checks)
   - [Model Training & Cross-Validation](#7-model-training--cross-validation)
   - [Official NASA Test Evaluation](#8-official-nasa-test-evaluation)
5. [Results & Benchmark Evaluation](#5-results--benchmark-evaluation)
   - [Overall Model Comparison](#overall-model-comparison)
   - [Impact of Feature Engineering](#impact-of-feature-engineering)
   - [Feature Importance & Lifecycle Error Dynamics](#feature-importance--lifecycle-error-dynamics)
6. [Conclusion & Future Scope](#6-conclusion--future-scope)
7. [Core Concepts & Simple Explanations](#7-core-concepts--simple-explanations)
8. [Comprehensive Viva Questions & Answers](#8-comprehensive-viva-questions--answers)

---

## 1. Abstract

This project focuses on estimating the **Remaining Useful Life (RUL)** of turbofan engines using multivariate time-series sensor data and classical machine learning techniques. We utilize the benchmark **NASA C-MAPSS FD001** subset, consisting of **100 training engines** run to run-to-failure conditions and **100 test engines** truncated at an operational cycle prior to failure.

After screening and dropping **7 invariant/constant sensors**, we designed a piecewise-linear RUL target capped at **125 cycles** to reflect early-life non-degraded operating states. From the **14 informative sensors**, we engineered **71 time-series degradation features** per engine computed strictly from historical cycles: 10-cycle rolling means, standard deviations, rolling linear regression slopes, and deviations from each engine's healthy initial baseline.

Three diverse regression architectures (**Random Forest**, **XGBoost**, and **Elastic Net**) were tuned with 3-fold `GroupKFold` cross-validation on 80 training engines and compared against 20 held-out validation engines. The champion model, **XGBoost**, achieved an **RMSE of 10.14 cycles**, an **MAE of 7.40 cycles**, and an **$R^2$ score of 0.94** on the official NASA test set, drastically outperforming a naive average baseline (RMSE 41.94). Furthermore, compared to models trained on raw sensor values alone, our engineered degradation features delivered an error reduction of **35% to 38%**.

---

## 2. Problem Statement

Given the multivariate sensor observation history of an industrial turbofan engine up to the current operational cycle $t$, predict the Remaining Useful Life (RUL)—the exact number of remaining flight cycles before the engine reaches terminal operational failure.

### Operational Constraints & Requirements:
1. **Zero Future Leakage:** At any time step $t$, the model must only leverage information available at or before cycle $t$.
2. **Generalizability to Unseen Units:** The model must reliably generalize to novel, previously unseen engines with varying initial health states and operational lifespans.

---

## 3. Objectives

- [x] **Data Ingestion & Integrity Auditing:** Validate sensor feeds, check for missing values, eliminate redundant variables, and profile engine operational distributions.
- [x] **RUL Target Construction:** Formulate an operational RUL target with an optimal saturation threshold (piecewise linear cap at 125 cycles).
- [x] **Strict Leakage-Free Feature Engineering:** Develop rolling statistics and baseline deviation metrics calculated strictly within each engine's backward history.
- [x] **Cross-Validation & Hyperparameter Tuning:** Optimize classical machine learning pipelines (Random Forest, XGBoost, Elastic Net) using engine-level grouping (`GroupKFold`).
- [x] **Lifecycle Error Stratification:** Analyze performance divergence between early-life healthy stages versus late-life degraded regimes.
- [x] **Benchmark Validation on Official Test Set:** Evaluate the best-performing pipeline on the 100 official NASA C-MAPSS test engines against ground-truth targets.

---

## 4. Methodology

### 1. Data Ingestion & Integrity Checks
- **Dimensions:** Exactly 100 engines in both training (`train_FD001.txt`) and testing (`test_FD001.txt`) sets.
- **Completeness:** 0 missing values (NaNs) and 0 duplicate rows across all sensor recordings.

### 2. Exploratory Data Analysis (EDA)
- **Lifespan Variability:** Training engine lifespans vary widely, spanning from **128 to 362 cycles** (mean lifespan $\approx 206$ cycles).
- **Sensor Pruning:** 7 sensors exhibit near-zero variance across all operating cycles:
  $$\text{Dropped Sensors} = \{s_1, s_5, s_6, s_{10}, s_{16}, s_{18}, s_{19}\}$$
  Pruning these eliminates uninformative dimensions and reduces overfitting risk.
- **Operating Conditions:** Subset FD001 operates under a single nominal operating condition (sea-level, steady state); therefore, multi-regime condition normalization is not required.

### 3. RUL Target Formulation
- **Raw RUL:** Defined as the total life of engine $i$ minus current cycle $t$:
  $$\text{RUL}_{\text{raw}}(i, t) = \text{max\_cycle}_i - t$$
- **Capped Piecewise Target:** Turbofan components do not exhibit measurable mechanical degradation during initial flight hours. In accordance with established literature, RUL is capped at **125 cycles**:
  $$\text{RUL}(i, t) = \min\left(125, \; \text{max\_cycle}_i - t\right)$$

### 4. Engine-Level Splitting
- **Partition:** 80 training engines (16,138 cycles) and 20 validation engines (4,493 cycles).
- **Leakage Prevention:** Split strictly by `unit_nr`. No engine appears in both sets, avoiding synthetic cross-correlation across identical physical units.

### 5. Feature Engineering (71 Features)
From the 14 retained sensors, features are generated on-the-fly per engine:
1. **Cycle Index:** Temporal marker of cumulative operating hours ($1$ feature).
2. **Raw Sensors:** Current instantaneous values ($14$ features).
3. **Rolling Mean (10-cycle window):** Moving average smoothing high-frequency measurement noise ($14$ features).
4. **Rolling Standard Deviation (10-cycle window):** Moving dispersion capturing rising vibration/instability ($14$ features).
5. **Rolling Linear Slope (10-cycle window):** First-derivative trend measuring degradation velocity ($14$ features).
6. **Healthy Baseline Drift (`_dev`):** Deviation of current sensor reading from the engine's initial healthy baseline (mean of first 10 cycles) ($14$ features).

$$\text{Total Features} = 1 + 14 + 14 + 14 + 14 + 14 = 71$$

### 6. Data Leakage Prevention Checks
- **Backward-Facing Windows:** Rolling windows and baseline metrics use expanding/backward windows only.
- **Empirical Truncation Invariance Test:** Evaluated features computed on shortened histories against full histories; identical rows yielded an exact diff of 0.
- **Target Independence:** Target RUL was completely isolated from the feature computation matrix.

### 7. Model Training & Cross-Validation
- **Search Strategy:** Systematic Grid Search over key tree/regularization hyperparameters.
- **Cross-Validation Scheme:** 3-fold `GroupKFold` grouped on engine IDs within the 80 training engines.
- **Leakage-Safe Scaling:** For Elastic Net, `StandardScaler` was encapsulated within an `sklearn.pipeline.Pipeline`, ensuring scaling statistics were fitted strictly on training folds.

### 8. Official NASA Test Evaluation
- **Final Fit:** The champion model selected from validation was retrained on the complete 100-engine training corpus.
- **Inference Setup:** For each of the 100 test engines, features were derived from its available operational trajectory, and a single prediction was generated at its final observed cycle ($t_{\text{last}}$).
- **Ground Truth:** Predictions were evaluated against the true NASA ground truth (`RUL_FD001.txt`, capped at 125).

---

## 5. Results & Benchmark Evaluation

### Overall Model Comparison

The table below summarizes performance across the 20 held-out validation engines and the official 100 NASA test engines:

| Model Architecture | Val RMSE (cycles) | Val MAE (cycles) | Val $R^2$ | Test RMSE (cycles) | Test MAE (cycles) | Test $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Mean Predictor)** | 41.72 | 36.93 | 0.00 | 41.94 | 34.83 | -0.10 | Naive Baseline |
| **Elastic Net** | 15.04 | 12.01 | 0.87 | 16.13 | 13.14 | 0.84 | Regularized Linear |
| **Random Forest** | 10.34 | 6.77 | 0.94 | 11.44 | 8.27 | 0.92 | Ensemble Bagging |
| **XGBoost (Best Model)** | **9.60** | **6.05** | **0.95** | **10.14** | **7.40** | **0.94** | **Champion Model** |

---

### Impact of Feature Engineering

To validate the hypothesis that temporal and drift features significantly boost classical machine learning performance, models were benchmarked on raw sensor values versus engineered feature sets:

| Model | RMSE (Raw Sensors Only) | RMSE (With Engineered Features) | Error Reduction (%) |
| :--- | :---: | :---: | :---: |
| **Random Forest** | 15.98 | **10.34** | **-35.3%** |
| **XGBoost** | 15.53 | **9.60** | **-38.2%** |
| **Elastic Net** | 17.96 | **15.04** | **-16.3%** |

> [!TIP]
> **Key Finding:** Feature engineering reduced prediction error by **~38%** for gradient boosting, confirming that explicit temporal drift features effectively bridge the gap between classical tree models and sequential dynamics.

---

### Feature Importance & Lifecycle Error Dynamics

#### 1. Dominant Features & Sensors
- **Most Predictive Feature Type:** Deviation from healthy start (`_dev`), which isolates physical degradation from engine-to-engine manufacturing variation.
- **Top 5 Informative Sensors:**
  1. `s11` — High-Pressure Compressor (HPC) Static Pressure
  2. `s15` — Bypass Duct Static Pressure
  3. `s20` — High-Pressure Turbine (HPT) Coolant Bleed
  4. `s4` — Low-Pressure Turbine (LPT) Total Temperature
  5. `s21` — Low-Pressure Turbine (LPT) Coolant Bleed

#### 2. Early-Life vs. Late-Life Error Behavior
- **Early-Life Regimes ($\text{RUL} = 125$):** XGBoost RMSE = **7.33 cycles**. Most engines are healthy and stably predicted at the cap.
- **Late-Life Regimes ($\text{RUL} < 125$):** XGBoost RMSE = **11.45 cycles**. Error increases slightly because physical degradation exhibits non-linear acceleration as thermal and mechanical breakdown compounds.

---

## 6. Conclusion & Future Scope

### Conclusion
Classical machine learning pipelines—when paired with domain-informed temporal degradation features—deliver state-of-the-art predictive performance on industrial turbofan degradation datasets. **XGBoost** proved to be the champion model, attaining an official test **RMSE of 10.14 cycles** and an **$R^2$ of 0.94**, closely followed by Random Forest (RMSE 11.44).

### Future Scope (Phase 2 & Beyond)
1. **Multi-Condition Subsets:** Extend methodology to C-MAPSS subsets FD002 and FD004, which feature 6 operational regimes requiring flight-envelope condition normalization.
2. **Deep Sequential Architectures:** Implement sequence-aware neural networks (LSTM, GRU, Temporal Convolutional Networks, and Transformers) to automatically learn temporal dynamics.
3. **Uncertainty Quantification:** Implement conformal prediction, quantile regression, or Bayesian neural networks to provide certified confidence intervals for maintenance safety margins.
4. **Asymmetric NASA Scoring Optimization:** Direct loss function alignment using custom asymmetric loss functions that penalize dangerous over-estimation ($d > 0$) more severely than conservative under-estimation.

---

## 7. Core Concepts & Simple Explanations

| Term / Concept | Plain-Language Explanation |
| :--- | :--- |
| **RUL (Remaining Useful Life)** | The number of operational flight cycles an engine can safely perform before failure. Example: If an engine fails at cycle 200 and is currently at cycle 150, its RUL is 50. |
| **Rolling Mean** | The average sensor reading over the previous 10 cycles. It removes instantaneous sensor noise and exposes the underlying trend. |
| **Rolling Slope** | The slope of a best-fit straight line through the last 10 readings. It indicates whether sensor readings are trending up or down, and at what speed. |
| **Healthy Baseline Drift (`_dev`)** | The difference between the current sensor reading and the engine's initial healthy baseline (first 10 cycles). It captures accumulated physical wear. |
| **Random Forest** | An ensemble of hundreds of decision trees trained on random subsets of data and features; the final prediction is the average of all individual trees. |
| **XGBoost** | An ensemble of sequential decision trees where each subsequent tree is trained specifically to correct the residual errors made by preceding trees. |
| **Elastic Net** | A regularized linear regression model combining $L_1$ (Lasso) and $L_2$ (Ridge) penalties to shrink coefficients and eliminate non-informative features. |
| **RMSE & MAE** | **RMSE** measures typical error magnitude in cycles with high penalties for large mistakes; **MAE** measures the unweighted average error in cycles. |
| **Engine-Level Validation** | Ensuring all records for a given engine reside exclusively in either the training set or the validation set, ensuring models are tested strictly on unseen units. |

---

## 8. Comprehensive Viva Questions & Answers

### Q1: Why is Remaining Useful Life (RUL) formulated as a regression problem rather than classification?
> **Answer:**  
> RUL represents a continuous quantity—the exact number of remaining operational cycles until failure. Formulating it as regression provides granular scheduling visibility (e.g., "replace part in 14 cycles"). Classification into discrete buckets (e.g., "healthy" vs. "failing") loses critical temporal precision necessary for maintenance logistics.

---

### Q2: How is the RUL target calculated from the C-MAPSS training data?
> **Answer:**  
> In the training set, engines are operated until complete failure. The maximum cycle observed for engine $i$ is denoted $\text{max\_cycle}_i$. At any current cycle $t$, raw RUL is calculated as:
> $$\text{RUL}_{\text{raw}} = \text{max\_cycle}_i - t$$
> At the final cycle ($t = \text{max\_cycle}_i$), RUL equals 0. A piecewise cap of 125 cycles is subsequently applied.

---

### Q3: Why is the RUL target capped at 125 cycles?
> **Answer:**  
> During the early operational lifespan of an engine, components experience negligible mechanical wear. Sensor telemetry during cycle 20 is indistinguishable from cycle 80. Without a cap, a model is forced to predict arbitrary differences between healthy states. Capping at 125 reflects the physical reality that degradation only manifests measurably in the final ~125 cycles of life.

---

### Q4: Why must the train/validation split be performed at the engine level rather than random row-level splitting?
> **Answer:**  
> Consecutive cycles from the same engine share virtually identical operational conditions and degradation signatures. A random row split would place cycle $t$ in train and cycle $t+1$ from the same engine in validation, causing severe data leakage. Engine-level splitting (`GroupKFold`) ensures the model is evaluated on completely unseen engines.

---

### Q5: Which sensors were removed during preprocessing and why?
> **Answer:**  
> Sensors **s1, s5, s6, s10, s16, s18, and s19** were dropped. Across all engines and operational cycles, their standard deviations are virtually 0 (constant readings). Because they contain no variance and no degradation signal, removing them prevents noise and redundant computation.

---

### Q6: Why was operating-condition normalization omitted for subset FD001?
> **Answer:**  
> FD001 is simulated under a single operating regime (sea-level, steady altitude/Mach). Unlike FD002 or FD004 (which alternate across 6 operational regimes requiring k-means or z-score normalization per regime), FD001 sensor variations are solely due to wear and noise.

---

### Q7: How do you guarantee zero data leakage during rolling feature engineering?
> **Answer:**  
> All rolling windows (mean, standard deviation, linear regression slope) are computed backwards using only time steps $\le t$. In addition, an empirical audit verified that features computed on a truncated engine history ($t \le 50$) match the identical rows in a full history ($t \le 200$) with 0 difference.

---

### Q8: What occurs during the initial cycles (e.g., cycles 1–9) where a 10-cycle window is incomplete?
> **Answer:**  
> Expanding windows are used (`min_periods=1`). At cycle 1, the mean is the instantaneous value, while standard deviation and slope default safely to 0. By cycle 10, the full window is active. No forward rows are ever imputed.

---

### Q9: Which engineered features provided the highest predictive gain?
> **Answer:**  
> The **deviation from healthy start (`_dev`)** yielded the highest feature importance. Because each engine has slight initial calibration variances, measuring the drift relative to its own first 10 cycles isolates mechanical wear from initial hardware offsets.

---

### Q10: Why does Elastic Net require feature standardization, whereas Random Forest and XGBoost do not?
> **Answer:**  
> Elastic Net optimizes an objective function with $L_1$ and $L_2$ coefficient penalties ($\lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2$). If features are on different scales (e.g., temperatures in hundreds vs. pressure ratios near 1), unscaled penalties unfairly penalize large-scale features. Tree-based models split points based on order rankings and are completely invariant to monotonic feature scaling.

---

### Q11: How were hyperparameters tuned without contaminating the validation set?
> **Answer:**  
> Hyperparameters were tuned strictly on the 80 training engines using 3-fold `GroupKFold` grid search. The 20 validation engines were set aside as a held-out benchmark solely for model comparison. The test set was untouched until final model verification.

---

### Q12: Why was model prediction error lower in early life than in late life?
> **Answer:**  
> In early life, the true target is capped at a constant value (125 cycles). The model easily learns to predict ~125 for healthy sensor profiles (RMSE 7.33). In late life, engines degrade non-linearly, and physical wear accelerates at different rates across units, resulting in higher variance and error (RMSE 11.45).

---

### Q13: What is the champion model and how do the contenders compare?
> **Answer:**  
> **XGBoost** is the best model with an official test RMSE of **10.14 cycles** and MAE of **7.40 cycles** ($R^2 = 0.94$). Random Forest performed comparably well (Test RMSE 11.44, MAE 8.27), while Elastic Net had higher error (Test RMSE 16.13) due to linear modeling constraints.

---

### Q14: What is the NASA Asymmetric Scoring Function?
> **Answer:**  
> The NASA scoring metric penalizes over-estimation ($\hat{y} > y$, predicting an engine has more life than it actually does) much more severely than under-estimation ($\hat{y} < y$):
> $$s = \sum_{i=1}^{N} \begin{cases} e^{-\frac{d_i}{13}} - 1 & \text{for } d_i < 0 \text{ (early / conservative)} \\ e^{\frac{d_i}{10}} - 1 & \text{for } d_i \ge 0 \text{ (late / catastrophic)} \end{cases}$$
> where $d_i = \hat{y}_i - y_i$. Late predictions risk in-flight catastrophic engine failure.

---

### Q15: Is including the operational `cycle` feature considered data leakage?
> **Answer:**  
> No. At any real-world operational inspection point, the total flight cycles logged since overhaul is an observed, documented operational metadata parameter. It contains no future information about when the engine will eventually fail.

---

### Q16: How was inference executed on the official NASA test dataset?
> **Answer:**  
> For each of the 100 test engines, sensor histories are observed up to an unrevealed cycle $t_{\text{last}}$. Feature engineering was applied across the available history, and a single prediction was generated for $t_{\text{last}}$. Predictions were compared against NASA's ground-truth RUL vector capped at 125.

---

### Q17: What are the primary technical limitations of this Phase 1 study?
> **Answer:**  
> 1. Evaluated solely on the single-condition FD001 dataset.
> 2. Utilizes simulated C-MAPSS data rather than noisy real-world airline telemetry.
> 3. Employs a single fixed train/validation partition rather than repeated nested cross-validation.
> 4. Fixed-size rolling windows introduce lag during abrupt degradation onset.
> 5. Point predictions lack probabilistic uncertainty intervals.