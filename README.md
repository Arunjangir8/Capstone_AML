# Project 7: Remaining Useful Life Prediction for Industrial Engines

**Predictive Maintenance via Sensor Features, Sequence Models, and Hybrid Innovation**  
*Course: Advanced Machine Learning (AML) & Deep Learning (DL) · Dataset: NASA C-MAPSS Turbofan Engine Degradation (subset FD001)*

## 👥 Team
* **Sarvesh Choudhary** (`sarveshDhayal`)
* **Arun Jangir** (`Arunjangir8`)
* **Keshav Chhabra** (`keshavvchhabra`)
* **Mayank Yadav** (`mayankthukran`)
* **Rohit Kumar** (`rohit-godara`)

---

## 📌 Project Overview & "The Hybrid Logic"
Every industrial turbofan engine starts in a healthy condition, undergoes progressive degradation with operational use, and eventually fails. From multivariate **sensor readings**, we estimate the **Remaining Useful Life (RUL)** — the remaining operational cycles (flights) before failure.

This capstone investigates the full continuum between degradation feature engineering and sequence learning:
1. **Phase 1 (AML Baseline):** Build 71 domain-engineered degradation features (rolling mean, std, trend slope, and baseline deviation `_dev`) across fixed 10-cycle windows, comparing **Linear Regression → Decision Tree → Random Forest**.
2. **Identifying the Gap:** Fixed rolling summary features discard the exact temporal sequence order and require manual domain tuning that fails to generalize across differing engine degradation trajectories and multiple operating regimes.
3. **Phase 2 (Deep Learning Sequence Models):** Train **PyTorch LSTM** and **Temporal 1D-CNN** directly on fixed 30-cycle sensor sliding windows $X \in \mathbb{R}^{30 \times 14}$ without manual feature engineering.
4. **Hybrid Innovation (Additional Marks):** Design and evaluate a technically justified **Hybrid Model** that combines the orthogonal error profiles of Random Forest (strong on baseline offset subtraction) and LSTM (strong on continuous temporal acceleration), provably lowering estimator variance.
5. **Fair Comparison:** Identical engine-level split (80 train, 20 validation engines; zero row leakage), evaluated on the **100 official NASA test engines** using RMSE, MAE, $R^2$, NASA Asymmetric Score ($S$), and Early vs Late degradation stages.

---

## 📁 Repository Structure
```
Capstone_AML/
├── notebooks/                                         ← Open and run in sequential order
│   ├── 01_EDA_Data_Exploration.ipynb                     Phase 1: Exploratory data analysis, flat sensor pruning
│   ├── 02_Data_Preparation_and_Feature_Engineering.ipynb  Phase 1: Capped RUL (125), engine split, 71 features
│   ├── 03_Model_Linear_Regression.ipynb                  Phase 1: Ordinary Least Squares (OLS) baseline
│   ├── 04_Model_Decision_Tree.ipynb                      Phase 1: CART regression tree with cost-complexity pruning
│   ├── 05_Model_Random_Forest.ipynb                      Phase 1: 150-tree bagged ensemble (Best Classical ML)
│   ├── 06_Final_Comparison_and_Test_Results.ipynb        Phase 1: Classical ML benchmark on 100 test engines
│   ├── 07_DL_Sequence_Preparation_and_Pipeline.ipynb     Phase 2: 3D sequence tensors (W=30), PyTorch pipeline
│   ├── 08_Model_PyTorch_LSTM.ipynb                       Phase 2: 2-layer stacked LSTM sequence regressor
│   ├── 09_Model_PyTorch_1DCNN.ipynb                      Phase 2: Multi-stage Temporal 1D-CNN regressor
│   ├── 10_Final_Phase1_vs_Phase2_Comparison.ipynb        Phase 2: Benchmark (RF vs DL, NASA Score)
│   ├── 11_Hybrid_Model_and_Ablation_Studies.ipynb        Hybrid Innovation & 4 Systematic Ablation Studies
│   └── utils.py                                          Shared constants, paths, NASA score, sequence helpers
├── app/
│   └── app.py                                         ← Interactive Streamlit Web Dashboard (AML, DL, Hybrid)
├── data/
│   ├── raw/                                           NASA C-MAPSS text files (train, test, true RUL)
│   └── processed/                                     Processed feature tables & PyTorch sequence arrays (.npy)
├── outputs/
│   ├── figures/                                       All diagnostic, ablation & comparative plots (.png)
│   ├── models/                                        Saved models (Classical .joblib & PyTorch .pt weights)
│   └── results/                                       Prediction CSVs & benchmark evaluation tables
├── docs/                                              Academic documentation & deliverables
│   ├── IEEE_Conference_Paper.tex                      Full 2-Column IEEE Conference Paper (.tex for Overleaf)
│   ├── PHASE2_ACADEMIC_REPORT.md                      Comprehensive academic report with Literature Review
│   ├── VIVA_VOCE_PHASE2.md                            Top 25 Viva Q&A with full mathematical derivations
│   └── Phase2_Presentation_Slides.md                  Slide-by-slide presentation script & speaker notes
└── requirements.txt                                   Project dependencies (scikit-learn, PyTorch, Streamlit)
```

---

## 📊 Final Benchmark Results (Official NASA Test Set, 100 Engines)

All models evaluated at each test engine's final recorded cycle against ground truth `RUL_FD001.txt` (capped at 125 cycles):

| Model | Paradigm | Input Representation | Test RMSE ↓ | Test MAE ↓ | Test R² ↑ | NASA Score ($S$) ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | Baseline | Constant Mean (107.9) | 51.63 | 39.47 | −0.660 | 248,340.7 |
| **Linear Regression** | Phase 1 (AML) | 71 Engineered Features | 15.85 | 12.84 | 0.844 | 406.6 |
| **Decision Tree** | Phase 1 (AML) | 71 Engineered Features | 14.96 | 10.80 | 0.861 | 357.0 |
| **Random Forest** | Phase 1 (Best AML) | 71 Engineered Features | 11.44 | 8.27 | 0.918 | 184.5 |
| **PyTorch 1D-CNN** | Phase 2 (DL Temporal) | Raw 30-Cycle Sequences | 18.93 | 13.30 | 0.777 | 1,404.3 |
| **PyTorch LSTM** | Phase 2 (DL Sequence) | Raw 30-Cycle Sequences | 13.23 | 9.88 | 0.891 | 240.5 |
| 🏆 **Hybrid Model** | **Hybrid Innovation** | **Fused Sequences + Features** | **10.99** | **7.98** | **0.925** | **170.1** |

---

## 🔬 Key Scientific Findings & Ablation Results

1. **Hybrid Synergy (State of the Art):** The Hybrid model lowered Test RMSE to **10.99 cycles**, Test MAE to **7.98 cycles**, and the NASA Score to **170.06**, outperforming both Random Forest (11.44) and LSTM (13.23) individually.
2. **End-to-End Sequence Learning:** The **PyTorch LSTM** reached **13.23 Test RMSE** and a **240.5 NASA score** without using *any* of the 71 hand-engineered features, demonstrating that recurrence captures continuous degradation momentum directly from raw sensors.
3. **Window Sensitivity Ablation:**
   - $W=15$: Test RMSE 15.22 (insufficient receptive field)
   - $W=30$: Test RMSE **13.23** (optimal; covers all 100 test engines without padding)
   - $W=45$: Test RMSE 14.10 (padding artifacts on short runs)
4. **Loss Function Robustness:** Huber loss (`SmoothL1Loss`) reduced Test RMSE from 14.72 (MSE) to **13.23**, providing stability against sensor noise and spikes.
5. **Aviation Asymmetric Scoring:** Early predictions ($d < 0$) are penalized exponentially less than late predictions ($d \ge 0$), reflecting safety imperatives where late failure detection is catastrophic.

---

## 🚀 How to Run the Project

### 1. Environment Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Notebooks
Execute notebooks in order from `01` to `11`:
```bash
cd notebooks && jupyter notebook
```

### 3. Launch the Interactive Web Dashboard
```bash
streamlit run app/app.py
```
*(Access at `http://localhost:8501`)*

---

## 🌐 Web Dashboard Features (`app/app.py`)
* **Live Model Selection:** Seamlessly toggle between Random Forest, PyTorch LSTM, 1D-CNN, and the **Hybrid Model**.
* **NASA Test Engine Explorer:** Select any of the 100 test engines to view side-by-side RUL predictions against ground truth and interactive multivariate sensor histories.
* **Single-Cycle Simulation:** Simulate sensor readings with Healthy, Mid-Life, or Critical failure presets.
* **Benchmark & Ablations Tab:** Interactive inspection of all benchmark tables, NASA penalty curves, and ablation charts.
