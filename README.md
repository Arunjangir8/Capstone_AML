# Project 7: Remaining Useful Life Prediction for Industrial Engines

**Predictive Maintenance via Sensor Features and Machine Learning (Phase 1)**
*Course: Advanced Machine Learning (AML) · Dataset: NASA C-MAPSS Turbofan Engine Degradation (subset FD001)*

## 👥 Team
* **Sarvesh Choudhary** (`sarveshDhayal`)
* **Arun Jangir** (`Arunjangir8`)
* **Keshav Chhabra** (`keshavvchhabra`)
* **Mayank Yadav** (`mayankgamerz25`)
* **Rohit Godara** (`rohit-godara`)


## 📌 What is this project?
Every engine starts healthy, slowly wears out and finally fails. From its **sensor readings** we predict the
**Remaining Useful Life (RUL)** — how many more cycles (flights) it can run before failure.

We build simple degradation features (moving average, std, trend slope, drift from healthy start) and compare three
models: **Linear Regression → Decision Tree → Random Forest**, with a fair **engine-level** split.

## 📁 Folder structure
```
Capstone_AML/
├── notebooks/                       ← open these in order
│   ├── 01_EDA_Data_Exploration.ipynb                     look at the data, charts
│   ├── 02_Data_Preparation_and_Feature_Engineering.ipynb  RUL target, split, 71 features
│   ├── 03_Model_Linear_Regression.ipynb                  model 1 (straight line)
│   ├── 04_Model_Decision_Tree.ipynb                      model 2 (yes/no questions, pruning)
│   ├── 05_Model_Random_Forest.ipynb                      model 3 (many trees) — best
│   ├── 06_Final_Comparison_and_Test_Results.ipynb        all models side by side
│   └── utils.py                                          small shared helpers (paths, metrics, plots)
├── app/app.py                       ← Streamlit web app (enter sensor values → predicted RUL)
├── data/
│   ├── raw/                         NASA C-MAPSS text files (train / test / RUL)
│   └── processed/                   feature tables made by notebook 02
├── outputs/
│   ├── figures/                     every chart, named by notebook number (01_..., 05_...)
│   ├── models/                      trained models (.joblib)
│   └── results/                     predictions and result tables (.csv)
├── docs/                            project report + viva Q&A, project brief, NASA papers
└── requirements.txt
```
Hidden cells (**⋯**) in the notebooks contain only plotting or long helper code — click to expand.

## 📊 Results (official NASA test set, 100 engines)

| Model | Test RMSE | Test MAE | Test R² |
|:---|:---:|:---:|:---:|
| Baseline (always predict the average) | 41.94 | 34.83 | −0.10 |
| Linear Regression | 15.85 | 12.84 | 0.84 |
| Decision Tree | 14.96 | 10.80 | 0.86 |
| **Random Forest (best)** | **11.44** | **8.27** | **0.92** |

* Engineered features reduced the Random Forest error by **35 %** vs raw sensors (validation RMSE 15.98 → 10.34).
* The drift from each engine's own healthy start (`_dev` features) gives ~85 % of the Random Forest importance.

## 🚀 How to run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. run the notebooks in order (02 creates the features, 03-05 save the models)
cd notebooks && jupyter notebook

# 2. start the web app (from the project folder)
streamlit run app/app.py
```
Or run all notebooks at once: `cd notebooks && for f in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace $f; done`

## 🌐 Web app
* **Enter sensor values:** type the current cycle and 14 sensor readings (or pick a Healthy / Mid-life / Near-failure
  example) → predicted RUL from the best model + health status.
* **NASA test engine:** pick one of the 100 test engines → predicted vs true RUL and its sensor history.
* **Compare models:** test results of all models.
