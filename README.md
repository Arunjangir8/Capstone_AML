# Project 7: Remaining Useful Life Prediction for Industrial Engines

**Predictive Maintenance via Sensor Features and Machine Learning**
*Course: Advanced Machine Learning (AML) · Dataset: NASA C-MAPSS Turbofan Engine Degradation (subset FD001)*

## 👥 Team
* **Sarvesh Choudhary** (`sarveshDhayal`)
* **Arun Jangir** (`Arunjangir8`)
* **Keshav Chhabra** (`keshavvchhabra`)
* **Group Member 4**
* **Group Member 5**

## 📌 What is this project?
Every engine starts healthy, slowly wears out and finally fails. From its **21 sensor readings** we predict the
**Remaining Useful Life (RUL)** — how many more cycles (flights) it can run before failure.

**Phase 1 (done):** build engineered degradation features (rolling mean, std, slope, drift from healthy start) and
train classical ML models (Elastic Net, Random Forest, XGBoost) with a fair **engine-level** split.

## 📁 Folder structure
```
Capstone_AML/
├── notebooks/                       ← open these in order
│   ├── 01_EDA_Data_Exploration.ipynb                     look at the data, charts
│   ├── 02_Data_Preparation_and_Feature_Engineering.ipynb  RUL target, split, 71 features
│   ├── 03_Model_Elastic_Net.ipynb                        model 1 (linear)
│   ├── 04_Model_Random_Forest.ipynb                      model 2 (bagging trees)
│   ├── 05_Model_XGBoost.ipynb                            model 3 (boosting trees) — best
│   ├── 06_Final_Comparison_and_Test_Results.ipynb        all models side by side
│   └── utils.py                                          small shared helpers (paths, metrics, plots)
├── app/app.py                       ← Streamlit web app (predict any test engine)
├── data/
│   ├── raw/                         NASA C-MAPSS text files (train / test / RUL)
│   └── processed/                   feature tables made by notebook 02 (not in git)
├── outputs/
│   ├── figures/                     every chart, named by notebook number (01_..., 05_...)
│   ├── models/                      trained models (.joblib)
│   └── results/                     predictions and result tables (.csv)
├── docs/                            academic report + viva Q&A, project brief, NASA papers
└── requirements.txt
```
Hidden cells (**⋯**) in the notebooks contain only plotting or long helper code — click to expand.

## 📊 Results (official NASA test set, 100 engines)

| Model | Test RMSE | Test MAE | Test R² |
|:---|:---:|:---:|:---:|
| Baseline (always predict the average) | 41.94 | 34.83 | −0.10 |
| Elastic Net | 16.13 | 13.14 | 0.84 |
| Random Forest | 11.44 | 8.27 | 0.92 |
| **XGBoost (best)** | **10.14** | **7.40** | **0.94** |

* Engineered features reduced the error by **~38 %** vs raw sensors (XGBoost validation RMSE 15.53 → 9.60).
* The drift from each engine's own healthy start (`_dev` features) gives ~80 % of the tree models' importance.

## 🚀 How to run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install libomp            # macOS only, needed by XGBoost

# 1. run the notebooks in order (02 creates the features, 03-05 save the models)
cd notebooks && jupyter notebook

# 2. start the web app (from the project folder)
streamlit run app/app.py
```
Or run all notebooks at once: `cd notebooks && for f in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace $f; done`

## 🔜 Next: Phase 2
The project brief also asks for a **sequence model (LSTM / 1D-CNN)** trained on raw sensor windows, compared with
XGBoost using the same split and target → planned as `07_Model_LSTM.ipynb`.
