"""
pipeline.py  --  Step 2: models, threshold selection, metrics, expected cost.

Design decision
---------------
The number of validation positives is controlled explicitly, so the threshold
is chosen on that validation set itself:

    fit on TRAIN  ->  score VALIDATION  ->  pick the F1-maximising threshold
    on VALIDATION  ->  evaluate on the fixed TEST set.

Repeating this over many random seeds replaces nested cross-validation. (State
this in the paper's method section.)

Sign conventions
----------------
benefit_f1        = F1(tuned threshold) - F1(threshold 0.5)      > 0 means tuning helps
benefit_cost_c<k> = cost(threshold 0.5) - cost(tuned threshold)  > 0 means tuning helps
cost = (k * FN + FP) / n_test  for a FN:FP cost ratio k  (cost per test sample)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import Data, make_split

COST_RATIOS = (10, 50, 200)  # FN:FP cost ratios (our own settings, not from a paper)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_model(name: str, seed: int, class_weight=None):
    if name == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed, class_weight=class_weight),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed, class_weight=class_weight
        )
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, early_stopping=False,
            random_state=seed, class_weight=class_weight
        )
    raise ValueError(f"unknown model: {name}")


def fit_model(name: str, X_tr, y_tr, seed: int, class_weight=None):
    model = make_model(name, seed, class_weight)
    model.fit(X_tr, y_tr)
    return model


def positive_scores(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------
# Threshold selection (uses VALIDATION data only)
# --------------------------------------------------------------------------
def tune_threshold_f1(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Threshold that maximises F1 on the validation set.

    Predictions use  p >= threshold. When several thresholds tie for the best F1
    (common with few positives), the median of the tied thresholds is returned
    so the choice is not biased to the smallest one.
    """
    if y_val.sum() == 0:
        raise ValueError("validation set has no positives")
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    # precision/recall have one more entry than thresholds; drop the last point
    prec, rec = prec[:-1], rec[:-1]
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
    best = f1.max()
    tied = thr[np.isclose(f1, best)]
    return float(np.median(tied))


def tune_threshold_cost(y_val: np.ndarray, p_val: np.ndarray, k: float) -> float:
    """Threshold that minimises validation cost  k*FN + FP  (FN costs k times a FP).

    Predictions use p >= threshold. Ties: median of the tied thresholds. If predicting
    everything negative is cheapest, a threshold just above the largest score is returned.
    """
    y_val = np.asarray(y_val)
    order = np.argsort(-p_val, kind="stable")
    ps, ys = p_val[order], y_val[order]
    tp_cum, fp_cum = np.cumsum(ys), np.cumsum(1 - ys)
    # last index of each distinct score = threshold at that score
    last = np.r_[ps[1:] != ps[:-1], True]
    thr = ps[last]
    n_pos = ys.sum()
    cost = k * (n_pos - tp_cum[last]) + fp_cum[last]
    thr = np.r_[thr, np.nextafter(ps[0], np.inf)]   # all-negative option
    cost = np.r_[cost, k * n_pos]
    best = cost.min()
    return float(np.median(thr[np.isclose(cost, best)]))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins on [0, 1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1], right=False), 0, n_bins - 1)
    ece, n = 0.0, len(y)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(ece)


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    out = {
        "f1": f1,
        "mcc": mcc,
        "gmean": float(np.sqrt(rec * spec)),
        "precision": prec,
        "recall": rec,
        "fp": int(fp),
        "fn": int(fn),
    }
    n = len(y)
    for k in COST_RATIOS:
        out[f"cost_c{k}"] = (k * fn + fp) / n
    return out


def score_row(model, X_va, y_va, X_te, y_te, meta: dict) -> dict:
    """One results row: default threshold vs validation-tuned threshold on TEST."""
    p_va = positive_scores(model, X_va)
    p_te = positive_scores(model, X_te)
    thr = tune_threshold_f1(y_va, p_va)

    d = threshold_metrics(y_te, p_te, 0.5)
    t = threshold_metrics(y_te, p_te, thr)

    row = dict(meta)
    row["chosen_threshold"] = thr
    for k, v in d.items():
        row[f"{k}_default"] = v
    for k, v in t.items():
        row[f"{k}_tuned"] = v

    # cost-optimal thresholds (one per cost ratio), also chosen on VALIDATION only
    for k in COST_RATIOS:
        tk = tune_threshold_cost(y_va, p_va, k)
        m = threshold_metrics(y_te, p_te, tk)
        row[f"thr_cost{k}"] = tk
        row[f"cost_c{k}_ctuned"] = m[f"cost_c{k}"]
        row[f"benefit_ctuned_c{k}"] = row[f"cost_c{k}_default"] - m[f"cost_c{k}"]

    # threshold-free metrics on TEST
    row["pr_auc"] = float(average_precision_score(y_te, p_te))
    row["roc_auc"] = float(roc_auc_score(y_te, p_te))
    row["brier"] = float(brier_score_loss(y_te, p_te))
    row["ece"] = ece_score(y_te, p_te)

    # benefits (positive = tuning helps)
    row["benefit_f1"] = row["f1_tuned"] - row["f1_default"]
    row["benefit_mcc"] = row["mcc_tuned"] - row["mcc_default"]
    for k in COST_RATIOS:
        row[f"benefit_cost_c{k}"] = row[f"cost_c{k}_default"] - row[f"cost_c{k}_tuned"]
    return row


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------
def run_once(data: Data, model_name: str, n_train_pos: int, n_val_pos: int,
             ratio: int, seed: int) -> dict:
    X_tr, y_tr, X_va, y_va = make_split(data, n_train_pos, n_val_pos, ratio, seed)
    model = fit_model(model_name, X_tr, y_tr, seed)
    meta = dict(model=model_name, n_train_pos=n_train_pos, n_val_pos=n_val_pos,
                ratio=ratio, seed=seed)
    return score_row(model, X_va, y_va, data.X_test, data.y_test, meta)


def run_many(data: Data, model_name: str, n_train_pos: int, n_val_pos: int,
             ratio: int, seeds) -> pd.DataFrame:
    rows = [run_once(data, model_name, n_train_pos, n_val_pos, ratio, s) for s in seeds]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def selftest() -> None:
    from data import build_data

    # --- threshold selection uses only the validation arrays it is given ---
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    thr = tune_threshold_f1(y, p)
    assert 0.6 < thr <= 0.7, thr  # best split puts both positives above it

    # --- metrics sanity ---
    m = threshold_metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.1, 0.8]), 0.5)
    assert m["fp"] == 1 and m["fn"] == 1 and abs(m["f1"] - 0.5) < 1e-12
    assert abs(m["cost_c10"] - (10 * 1 + 1) / 4) < 1e-12
    assert abs(ece_score(np.array([1, 1, 0, 0]), np.array([1.0, 1.0, 0.0, 0.0]))) < 1e-12

    # --- no test data is used for fitting or threshold selection ---
    data = build_data("synthetic")
    X_tr, y_tr, X_va, y_va = make_split(data, 60, 20, 100, seed=1)
    model = fit_model("lr", X_tr, y_tr, 1)
    row_a = score_row(model, X_va, y_va, data.X_test, data.y_test, {})
    # replace the TEST labels with garbage: chosen threshold must not change
    rng = np.random.default_rng(0)
    y_garbage = rng.integers(0, 2, size=len(data.y_test))
    row_b = score_row(model, X_va, y_va, data.X_test, y_garbage, {})
    assert row_a["chosen_threshold"] == row_b["chosen_threshold"]
    # and the model's predictions do not depend on TEST labels
    assert np.array_equal(positive_scores(model, data.X_test),
                          positive_scores(model, data.X_test))

    # --- cost-optimal threshold equals brute force ---
    rng = np.random.default_rng(1)
    yb = (rng.random(300) < 0.1).astype(int)
    pb = np.clip(0.3 * yb + rng.random(300) * 0.7, 0, 1)
    for k in (1, 10, 50):
        t = tune_threshold_cost(yb, pb, k)
        c_t = k * ((pb < t) & (yb == 1)).sum() + ((pb >= t) & (yb == 0)).sum()
        c_best = min(k * ((pb < s) & (yb == 1)).sum() + ((pb >= s) & (yb == 0)).sum()
                     for s in np.r_[np.unique(pb), 2.0])
        assert c_t == c_best, (k, c_t, c_best)

    # --- full row for all models ---
    for name in ("lr", "rf", "hgb"):
        r = run_once(data, name, 60, 20, 100, seed=1)
        assert 0.0 <= r["f1_tuned"] <= 1.0 and 0.0 <= r["brier"] <= 1.0
    print("pipeline.py selftest: all checks passed")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--source", default="fraud", choices=["fraud", "synthetic"])
    ap.add_argument("--model", default="lr", choices=["lr", "rf", "hgb"])
    ap.add_argument("--n-train-pos", type=int, default=100)
    ap.add_argument("--n-val-pos", type=int, default=50)
    ap.add_argument("--ratio", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    if args.selftest:
        selftest()
    else:
        from data import build_data, describe

        data = build_data(args.source)
        describe(data)
        df = run_many(data, args.model, args.n_train_pos, args.n_val_pos,
                      args.ratio, range(args.seeds))
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 50)
        cols = ["model", "n_train_pos", "n_val_pos", "ratio", "seed",
                "chosen_threshold", "f1_default", "f1_tuned", "benefit_f1",
                "pr_auc", "brier", "ece", "benefit_cost_c50"]
        print(df[cols].round(4).to_string(index=False))
        df.to_csv("pipeline_example.csv", index=False)
        print("saved pipeline_example.csv")