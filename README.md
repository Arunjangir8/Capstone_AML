# Capstone_AML — Predictive Maintenance, Phase 1 (Classical ML)

RUL prediction for NASA C-MAPSS FD001 turbofan engines with Random Forest, XGBoost and Elastic Net.

- `predictive_maintenance_aml_phase1.ipynb` — the notebook (already executed, with outputs)
- `ACADEMIC_REPORT.md` — abstract, methodology, results, viva Q&A
- `results/` — figures, saved models, metrics, test predictions
- `build_notebook.py` — regenerates the notebook
- `detailed_version/` — builder for an extended version (more features and analyses)

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install libomp            # macOS only (XGBoost)
export CMAPSS_DATA_DIR=../CMAPSSData   # folder with train_FD001.txt, test_FD001.txt, RUL_FD001.txt
jupyter notebook predictive_maintenance_aml_phase1.ipynb
```
Runs in about 2 minutes on a laptop.
