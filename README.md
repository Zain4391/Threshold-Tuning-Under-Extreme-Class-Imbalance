# Threshold tuning under extreme class imbalance

Code and raw results for: *Threshold Tuning Under Extreme Class Imbalance: Separating Imbalance Ratio, Training Positives and Validation Positives* (Zain Rasool Hashmi, FAST-NUCES, Karachi).

## Setup
```
pip install -r requirements.txt
```
Datasets (credit-card fraud, Covertype, protein_homo) are downloaded automatically on first use and cached in `cache/`.

## Run
`run_all.bat` runs all 21 experiments (A, B, C, D and strategy versions S_A, S_C, S_D on three datasets). Output goes to `v2_<experiment>_<dataset>/` (`raw_*.csv` = every run, `summary_*.csv` = means, plus figures). Single runs: see `exp_ac.py --help` and `exp_strat.py --help`.

## Reproduce paper figures and tables
```
python analysis/final_analysis.py
```

## Files
- `data.py` – data loading, controlled subsampling, ratio-matched test set
- `pipeline.py` – models, F1- and cost-optimal threshold tuning, metrics
- `exp_ac.py` – Experiments A–D
- `exp_strat.py` – resampling / class-weight strategies
- `v2_*` – raw results used in the paper
