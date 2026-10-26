# Academic Documentation

All numbers come from `predictive_maintenance_aml_phase1.ipynb` (NASA C-MAPSS FD001, seed 42, RUL cap 125).

## Abstract
This project predicts the Remaining Useful Life (RUL) of turbofan engines from sensor data using classical machine
learning. On NASA's C-MAPSS FD001 dataset (100 training engines run to failure and 100 test engines stopped before
failure), we removed 7 constant sensors and built an RUL target capped at 125 cycles. From the 14 useful sensors we
created simple degradation features inside each engine using only past cycles: 10-cycle rolling means, standard
deviations and slopes, plus each sensor's drift from the engine's healthy starting level. Random Forest, XGBoost and
Elastic Net were tuned on 80 training engines and compared on 20 held-out validation engines. The best model,
XGBoost, achieved **RMSE 10.14 and MAE 7.40 cycles (R² 0.94)** on the official test set, compared with RMSE 41.94 for
a baseline that always predicts the average. The engineered features reduced the error by about 35–38 % compared with
raw sensor values.

## Problem Statement
Given an engine's sensor history up to the current cycle, predict how many cycles remain before it fails. The model
must work on engines it has never seen and must use only information available at that moment.

## Objectives
1. Load, check and explore the data.
2. Build the RUL target.
3. Create degradation features without data leakage.
4. Train and compare Random Forest, XGBoost and Elastic Net against a baseline.
5. Compare errors in early and late life.
6. Test the best model on the official NASA test set.

## Methodology
1. **Data checks.** No missing values and no duplicates. 100 engines in train and in test.
2. **EDA.** Lifetimes range from 128 to 362 cycles. Sensors are noisy but drift near failure. s1, s5, s6, s10, s16,
   s18 and s19 are constant, so they were dropped. There is only one operating condition, so no normalisation by
   condition is needed.
3. **Target.** RUL = last cycle − current cycle, capped at 125.
4. **Split by engine.** 80 engines for training and 20 for validation. No engine appears in both.
5. **Features (71).** The cycle number, 14 raw sensors, and for each sensor the 10-cycle rolling mean, rolling std,
   rolling slope, and deviation from the healthy start (mean of the first 10 cycles).
6. **Leakage checks.** Features computed on a shortened history are identical to those computed on the full history
   (so no future data is used). There are no missing values and the target is not a feature.
7. **Models.** Grid search with 3-fold GroupKFold on the training engines. The scaler is inside the pipeline (for
   Elastic Net).
8. **Testing.** The best validation model is retrained on all 100 training engines and makes one prediction per test
   engine at its last cycle.

## Results

| Model | Val RMSE | Val MAE | Val R² | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 41.72 | 36.93 | 0.00 | 41.94 | 34.83 | −0.10 |
| Random Forest | 10.34 | 6.77 | 0.94 | 11.44 | 8.27 | 0.92 |
| **XGBoost** | **9.60** | **6.05** | **0.95** | **10.14** | **7.40** | **0.94** |
| Elastic Net | 15.04 | 12.01 | 0.87 | 16.13 | 13.14 | 0.84 |

| Model | RMSE raw sensors only | RMSE with engineered features |
|---|---:|---:|
| Random Forest | 15.98 | 10.34 |
| XGBoost | 15.53 | 9.60 |
| Elastic Net | 17.96 | 15.04 |

The most important feature type was the deviation from the healthy start. The most informative sensors were s11,
s15, s20, s4 and s21. Errors were higher in late life (XGBoost RMSE 11.45) than in early life (7.33), because of the cap
(see Q12).

## Conclusion and Future Scope
Simple engineered degradation features allow classical models to predict RUL with about 10 cycles RMSE. XGBoost
performed best, slightly ahead of Random Forest. Future work: the other C-MAPSS subsets, repeated cross-validation,
uncertainty estimates, and deep learning models in Phase 2.

## Simple Explanations
* **RUL:** how many more cycles the engine can run. If it fails at cycle 200 and we are at cycle 150, RUL = 50.
* **Rolling mean:** the average of the last 10 readings. It smooths out noise.
* **Rolling slope:** the slope of a straight line through the last 10 readings. It tells us whether the sensor is
  going up or down, and how fast.
* **Random Forest:** many decision trees, each trained on random samples; the answer is their average.
* **XGBoost:** trees built one after another, each correcting the previous trees' mistakes.
* **Elastic Net:** a linear formula with a penalty that removes useless features and keeps the weights small.
* **RMSE:** typical error size in cycles (big mistakes count more). **MAE:** average error in cycles.
* **Engine-level validation:** all rows of an engine go either to training or to validation, so we always test on
  unseen engines.

## Viva Questions and Answers
1. **Why is this regression?** RUL is a number of cycles, not a category.
2. **How is RUL calculated?** Last cycle of the engine minus the current cycle, so it reaches 0 at failure. It is
   then capped at 125.
3. **Why cap at 125?** Healthy engines look the same whether 150 or 300 cycles remain, so the model cannot know the
   difference. The cap treats every healthy engine as "125".
4. **Why split by engine?** Rows of one engine are almost identical. A random row split would put near-copies in
   both sets and give over-optimistic scores.
5. **Which sensors were removed?** s1, s5, s6, s10, s16, s18 and s19. They are constant or nearly constant, so they
   carry no information.
6. **Why no operating-condition normalisation?** FD001 has only one operating condition.
7. **How do you avoid using future data?** Rolling windows look only backwards, inside each engine. We also check
   that features on a shortened history match those on the full history.
8. **What happens at the first cycles?** The window uses the cycles available so far. Std and slope are 0 at cycle 1.
9. **Which features helped most?** The deviation from the engine's healthy start, then the rolling means.
10. **Why does Elastic Net need scaling?** Its penalty depends on the size of the weights, which depends on the
    feature units. Trees do not need scaling.
11. **How were hyper-parameters tuned?** Grid search with GroupKFold on the training engines only. The validation
    engines were used only to compare the models.
12. **Why is the error lower early in life?** Because of the cap: most early rows have target 125, which is easy to
    predict. The true difficulty is knowing when degradation will start.
13. **Which model is best?** XGBoost (test RMSE 10.14). Random Forest is close (11.44), so the difference is small.
14. **What is the NASA score?** An extra metric that punishes late predictions (over-estimating RUL) more than early
    ones, because predicting failure too late is dangerous. It is optional, and this simple version reports RMSE, MAE and R² instead.
15. **Is using `cycle` leakage?** No. The current cycle is known at prediction time.
16. **How was the test set evaluated?** Features were built from each test engine's own history. We took the last
    cycle, predicted once per engine, and compared with the NASA RUL file (capped at 125). Nothing was trained on the
    test data.
17. **Limitations?** Only FD001, simulated data, one validation split, rolling-window lag, and no uncertainty
    estimates.
