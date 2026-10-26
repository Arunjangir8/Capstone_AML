# Project Report — Remaining Useful Life Prediction for Industrial Engines

### Predictive Maintenance with Sensor Features and Classical Machine Learning (Phase 1)

- **Course:** Advanced Machine Learning (AML)
- **Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation (subset FD001)
- **Models:** Linear Regression, Decision Tree, Random Forest
- **Final test:** Official NASA test set (100 engines)

> All numbers in this report come from the notebooks in `notebooks/` (01 – 06), seed `42`, RUL target capped at `125` cycles.

---

## Contents
1. Abstract
2. Problem Statement
3. Objectives
4. Method
5. Results
6. Conclusion and Future Scope
7. Key Terms in Simple Words
8. Viva Questions and Answers

---

## 1. Abstract

We predict the **Remaining Useful Life (RUL)** of aircraft engines — how many more cycles an engine can run before it
fails — from its sensor readings. We use the **NASA C-MAPSS FD001** data: 100 training engines recorded until failure and
100 test engines whose recording stops before failure.

We removed **7 sensors that never change**, capped the RUL target at **125 cycles**, and built **71 features** for each
cycle using only the engine's past: moving averages, standard deviations, trend slopes and the drift of each sensor from
the engine's own healthy start.

We trained three models from the course — **Linear Regression, Decision Tree and Random Forest** — tuned them with
3-fold cross-validation on 80 training engines, and compared them on 20 validation engines. The best model,
**Random Forest**, reached **RMSE 11.44 cycles, MAE 8.27 cycles and R² 0.92** on the official NASA test set, compared
with RMSE 41.94 for a simple "always predict the average" baseline. The engineered features reduced the Random Forest error
by **35 %** compared with raw sensor values.

---

## 2. Problem Statement

Given the sensor history of an engine up to the current cycle *t*, predict its RUL (number of cycles left before failure).
RUL is a number, so this is a **regression** problem.

Rules:
1. **No future information:** at cycle *t* we may only use data from cycles ≤ *t*.
2. **Work on new engines:** the model must predict well for engines it has never seen.

---

## 3. Objectives

- [x] Load and check the data (missing values, duplicates, useless sensors).
- [x] Explore the data (EDA) with charts.
- [x] Build the RUL target (capped at 125 cycles).
- [x] Build leakage-free features from each engine's own past.
- [x] Train and tune Linear Regression, Decision Tree and Random Forest with cross-validation.
- [x] Compare errors in early life and late life.
- [x] Test the best model on the 100 official NASA test engines.

---

## 4. Method

### 4.1 Data check
- 100 training engines (20,631 rows) and 100 test engines.
- 26 columns: engine id, cycle, 3 operating settings, 21 sensors.
- **0 missing values and 0 duplicate rows** → no cleaning needed.

### 4.2 Exploratory Data Analysis (notebook 01)
- Engines live between **128 and 362 cycles** (average ≈ 206).
- **Variance Threshold:** 7 sensors never change (≈ 0 variance) → removed: s1, s5, s6, s10, s16, s18, s19. **14 sensors kept.**
- FD001 has **one operating condition**, so the 3 settings are dropped and no condition normalisation is needed.
- **Pearson correlation:** s11, s4, s12, s7, s15, s21, s20 are most related to RUL (|r| ≈ 0.6–0.7).
- Sensors are noisy but clearly drift near failure → looking at the **trend over several cycles** helps.

### 4.3 RUL target (notebook 02)
- RUL = last cycle of the engine − current cycle.
- Capped at **125**: RUL = min(125, last cycle − current cycle). A healthy engine looks the same at 300 or 150 cycles left,
  so we call every healthy engine "125".

### 4.4 Train / validation split
- **80 engines for training, 20 for validation**, split **by engine** (all rows of one engine stay together).
- Splitting rows randomly would put almost identical rows of the same engine in both sets → **data leakage**.

### 4.5 Feature engineering (71 features)
For each of the 14 sensors, using the last 10 cycles of the **same engine**:

| Feature | Meaning |
|---|---|
| `_mean` | moving average → removes noise |
| `_std` | standard deviation → is the sensor becoming unstable? |
| `_slope` | slope of a straight line through the last 10 readings → the **trend** |
| `_dev` | moving average − the engine's own healthy start (first 10 cycles) → how much the engine has **drifted** |

Plus `cycle` and the 14 raw sensors: **1 + 14 + 14 × 4 = 71 features.**

### 4.6 Leakage checks
- Features recomputed with only the first 100 cycles of an engine are identical → **no future data is used**.
- No missing values; the target is not a feature; training and validation engines do not overlap.

### 4.7 Models and tuning (notebooks 03 – 05)

| Model | Idea | Settings tried |
|---|---|---|
| **Linear Regression** | weighted sum of features (OLS), features standardised | none to tune |
| **Decision Tree** | yes/no questions on features, prediction = average RUL in the leaf (CART) | `max_depth`, `min_samples_leaf` (pre-pruning), `ccp_alpha` (post-pruning) |
| **Random Forest** | average of 150 trees, each trained on a bootstrap sample (bagging) | `max_depth`, `min_samples_leaf` |

- Tuning: **GridSearchCV** with **3-fold cross-validation** (`GroupKFold`: one engine is never in two folds).
- The 20 validation engines are used only to compare the tuned models.

### 4.8 Official test
- The best model (chosen on validation) is retrained on all 100 training engines.
- Each test engine is predicted once, at its **last recorded cycle**, and compared with `RUL_FD001.txt` (capped at 125).

---

## 5. Results

### 5.1 Model comparison

| Model | Val RMSE | Val MAE | Val R² | Test RMSE | Test MAE | Test R² |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline (average) | 41.72 | 36.93 | 0.00 | 41.94 | 34.83 | −0.10 |
| Linear Regression | 15.29 | 11.94 | 0.87 | 15.85 | 12.84 | 0.84 |
| Decision Tree | 14.13 | 9.18 | 0.89 | 14.96 | 10.80 | 0.86 |
| **Random Forest (best)** | **10.34** | **6.77** | **0.94** | **11.44** | **8.27** | **0.92** |

- **Linear Regression** can only fit a straight line, but the real pattern is "flat, then a fast drop".
- **Decision Tree** learns the non-linear pattern but overfits easily: training error keeps falling as the tree gets deeper,
  while validation error stops improving after depth ≈ 7–10. Pruning was needed.
- **Random Forest** averages 150 different trees → much lower variance → best model.

### 5.2 Effect of feature engineering (validation RMSE)

| Model | Raw sensors only | With engineered features | Improvement |
|:---|:---:|:---:|:---:|
| Linear Regression | 17.37 | 15.29 | 12.0 % |
| Decision Tree | 17.73 | 14.13 | 20.3 % |
| **Random Forest** | 15.98 | **10.34** | **35.3 %** |

### 5.3 Most useful features
- Feature type: **`_dev` (drift from the engine's own healthy start) ≈ 85 %** of Random Forest importance, followed by moving averages.
- Single most important feature: **`s11_dev`** (drift of the HPC outlet static pressure), ≈ 74 % of importance.
- Other useful sensors: s4 (LPT outlet temperature), s15 (bypass ratio), s17 (bleed enthalpy), s2 (LPC outlet temperature), s9 (core speed).

### 5.4 Early vs late life (validation RMSE)

| Model | Early (RUL > 100) | Late (RUL ≤ 100) |
|:---|:---:|:---:|
| Linear Regression | 12.23 | 17.86 |
| Decision Tree | 10.01 | 17.33 |
| Random Forest | 8.01 | 12.26 |

Errors are larger in late life. Early life looks easier only because of the cap: most early rows have target 125.

---

## 6. Conclusion and Future Scope

**Conclusion:** with simple engineered features and classical ML, **Random Forest** predicts the RUL of the 100 official test
engines with **RMSE 11.44 and MAE 8.27 cycles (R² 0.92)**, against RMSE 41.94 for the baseline. The most useful idea was to
measure how far each sensor has drifted from the engine's **own healthy start**.

**Limitations:** only FD001 (one operating condition, one fault type, simulated data); one 80/20 split; moving-average windows
react with a small delay; the cap of 125 is a modelling choice.

**Future scope:** apply the method to FD002 – FD004 (several operating conditions) and use repeated cross-validation for
more reliable comparisons.

---

## 7. Key Terms in Simple Words

| Term | Meaning |
|:---|:---|
| **RUL** | Cycles left before failure. Fails at cycle 200, now at cycle 150 → RUL = 50. |
| **Moving average** | Average of the last 10 readings — removes noise. |
| **Trend slope** | Slope of a straight line through the last 10 readings — is the sensor going up or down, and how fast? |
| **Drift (`_dev`)** | Current moving average − the engine's own healthy start value. |
| **Linear Regression** | Prediction = weighted sum of features; weights found by OLS (minimise squared errors). |
| **Decision Tree** | A flow chart of yes/no questions; the leaf predicts the average RUL of its training rows. |
| **Pruning** | Limiting tree growth (`max_depth`, `min_samples_leaf`) or cutting weak branches (`ccp_alpha`) to stop overfitting. |
| **Random Forest** | Many trees on random samples (bagging); the prediction is their average. |
| **MAE / RMSE / R²** | Average error / typical error with big mistakes counted more / share of variation explained. |
| **Cross-validation** | Split training data into k folds, train on k−1, test on the remaining one, repeat. |
| **Data leakage** | The model sees information during training that it would not have in real life. |

---

## 8. Viva Questions and Answers

**Q1. Why is RUL a regression problem?**
RUL is a number of cycles, not a category. Regression gives an exact estimate ("about 14 cycles left").

**Q2. How is the RUL target calculated?**
Training engines run until failure. RUL = last cycle of the engine − current cycle, so it is 0 at the last cycle. Then it is capped at 125.

**Q3. Why cap RUL at 125?**
A healthy engine shows no wear in its sensors, so 300 and 150 cycles left look the same. The cap tells the model "healthy = 125"
and lets it focus on the degradation phase.

**Q4. Why split by engine instead of by row?**
Rows of the same engine are almost identical. A random row split would put the same engine in train and validation → data
leakage and an over-optimistic score.

**Q5. Which sensors were removed and why?**
s1, s5, s6, s10, s16, s18, s19 — they (almost) never change, so their variance is ≈ 0 (Variance Threshold). They carry no information.

**Q6. Why no operating-condition normalisation?**
FD001 has only one operating condition (setting_3 is always 100). Datasets FD002/FD004 have six and would need it.

**Q7. How do you make sure features do not use the future?**
All rolling windows look backwards only (cycles ≤ *t*). We also recomputed features on a shortened history and got exactly the same values.

**Q8. What happens in the first 9 cycles, before the 10-cycle window is full?**
The window uses the cycles available so far. At cycle 1 the moving average is the reading itself, and std and slope are 0.

**Q9. Which features helped most?**
The `_dev` features (drift from the engine's own healthy start), ≈ 85 % of the Random Forest importance. Each engine starts
slightly differently, so comparing it with its own start shows the real wear.

**Q10. Why does Linear Regression need scaling but the trees do not?**
Scaling makes the linear weights comparable. Trees only compare a feature with a threshold, so units do not matter.

**Q11. Why is Linear Regression the weakest model?**
It can only draw a straight line. The sensors stay flat for a long time and then change fast near failure — a non-linear pattern.

**Q12. How does a Decision Tree choose a split in regression?**
CART tries features and thresholds and picks the split that reduces the **MSE** the most. Gini and Entropy are the same idea for classification.

**Q13. How did you handle overfitting in the Decision Tree?**
Pruning, chosen by cross-validation: `max_depth` and `min_samples_leaf` (pre-pruning) and `ccp_alpha` (cost-complexity post-pruning).
The depth plot in notebook 04 shows training error falling while validation error stops improving.

**Q14. Why is Random Forest better than one Decision Tree?**
One tree has high variance (small data changes → a different tree). Random Forest trains many trees on bootstrap samples
with random feature subsets and averages them → lower variance (bagging).

**Q15. How were hyper-parameters tuned without touching the validation set?**
GridSearchCV with 3-fold GroupKFold on the 80 training engines only. Validation engines were used once to compare the
tuned models; the test set only at the very end.

**Q16. Why are errors larger in late life?**
Near failure, engines wear out at different speeds, so RUL is harder to predict. Early life looks easy because the target there is mostly the constant 125.

**Q17. Is using `cycle` as a feature leakage?**
No. The engine's age (cycles flown so far) is known at prediction time; it says nothing about when the engine will fail.

**Q18. How was the official test set evaluated?**
The best model was retrained on all 100 training engines. For each test engine we built features from its own history and
predicted once at its last recorded cycle, then compared with `RUL_FD001.txt` (capped at 125).

**Q19. What does the web app do?**
It loads the best model (Random Forest). You can enter the current cycle and 14 sensor readings to get a predicted RUL and a
health status, or pick one of the 100 NASA test engines to compare the prediction with the true RUL.

**Q20. What are the limitations?**
Only FD001, simulated data, one train/validation split, rolling windows react with a small delay, and the cap of 125 is a modelling choice.
