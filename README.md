# Project 7: Remaining Useful Life Prediction for Industrial Engines

**Predictive Maintenance via Sensor Features and Machine Learning**  
*Course: Advanced Machine Learning (AML)*  
*Dataset: NASA C-MAPSS Turbofan Engine Degradation Simulation (Subset FD001)*  

---

## 👥 Team Contributors (Group of 5)
* **Sarvesh Choudhary** (`sarveshDhayal`)
* **Arun Jangir** (`Arunjangir8`)
* **Keshav Chhabra** (`keshavvchhabra`)
* **Group Member 4**
* **Group Member 5**

---

## 📌 Project Overview
The objective of this project is to estimate the **Remaining Useful Life (RUL)** — the remaining operating cycles before failure — of aircraft gas turbine engines using multivariate time-series sensor data from NASA's C-MAPSS simulation benchmark.

In **Phase 1 (Classical Machine Learning)**, we explore the *'Hybrid' Logic*: transforming raw sensor degradation trajectories into informative temporal features (rolling statistics and healthy baseline drifts) and training regression algorithms (Random Forest, XGBoost, and Elastic Net) with strict engine-level validation to prevent data leakage.

---

## 📊 Key Results (Official NASA Test Set)

Models evaluated on the 100 official test engines at their final observed cycles against true RUL ground truth:

| Model | Test RMSE (cycles) | Test MAE (cycles) | Test $R^2$ | Status |
|:---|:---:|:---:|:---:|:---|
| **Baseline (Mean Predictor)** | 41.94 | 34.83 | -0.10 | Reference Benchmark |
| **Elastic Net** | 16.13 | 13.14 | 0.84 | Linear Baseline with Regularization |
| **Random Forest** | 11.44 | 8.27 | 0.92 | Non-linear Tree Ensemble |
| **XGBoost (Best Model)** | **10.14** | **7.40** | **0.94** | **Selected Final Model** |

### 🔍 Key Findings:
- **Feature Engineering Impact:** Engineered degradation features reduced prediction error by **~38%** compared to training on raw sensor readings alone (RMSE dropped from 15.53 to 9.60 on validation).
- **Dominant Features:** The drift/deviation from an engine's own initial healthy baseline (`_dev`) accounted for over 80% of tree model feature importance.
- **Top Informative Sensors:** High-pressure compressor static pressure (`s11`), bypass ratio (`s15`), coolant bleeds (`s20`, `s21`), and LPT temperature (`s4`).

---

## 📁 Repository Structure
```
Capstone_AML/
├── predictive_maintenance_aml_phase1.ipynb   # Executed master notebook with outputs & plots
├── ACADEMIC_REPORT.md                        # Formal academic report, methodology & viva Q&A
├── README.md                                 # Project overview & documentation
├── build_notebook.py                         # Automated script to regenerate clean notebook
├── requirements.txt                          # Python dependencies
├── results/                                  # Saved models, evaluation tables & visual charts
│   ├── final_model_xgboost.joblib            # Trained XGBoost model artifact
│   ├── random_forest.joblib                  # Trained Random Forest model
│   ├── elastic_net.joblib                    # Trained Elastic Net pipeline
│   ├── test_predictions.csv                  # Per-engine test predictions (100 engines)
│   ├── validation_results.csv                # Validation metrics comparison
│   ├── early_vs_late.csv                     # Degradation stage performance breakdown
│   ├── raw_vs_engineered.csv                 # Feature ablation study
│   └── *.png                                 # EDA, feature importance, and test result plots
└── detailed_version/                         # Extended builder with additional analyses
```

---

## 🚀 How to Run & Reproduce

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Arunjangir8/Capstone_AML.git
cd Capstone_AML

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# macOS only (required for XGBoost OpenMP runtime)
brew install libomp
```

### 2. Dataset Setup
Download the NASA C-MAPSS dataset from [NASA Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) and set the path:
```bash
export CMAPSS_DATA_DIR="/path/to/CMAPSSData"
```

### 3. Launch Notebook
```bash
jupyter notebook predictive_maintenance_aml_phase1.ipynb
```
The notebook executes end-to-end in approximately 2 minutes on standard hardware.
