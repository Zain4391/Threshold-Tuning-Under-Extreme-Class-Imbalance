"""
data.py  --  Step 1: data loading and controlled subsampling.

Conventions
-----------
* Label 1 = minority (fraud), label 0 = majority.
* "ratio" = negatives per positive. ratio=100 means 1:100.
* A fixed TEST set is cut once from the full data (stratified, natural class
  distribution) and is never used for fitting or threshold selection.
* The remaining POOL is used to draw TRAIN and VALIDATION sets of exactly
  controlled size:
      TRAIN : n_train_pos positives, n_train_pos * ratio negatives
      VAL   : n_val_pos   positives, n_val_pos   * ratio negatives
  TRAIN and VAL never share rows.

Paired design (important for experiment B)
------------------------------------------
For a given seed, the TRAIN set depends only on (seed, n_train_pos, ratio).
It does NOT change when n_val_pos changes. The VAL set is drawn from the rows
left after TRAIN, using a separate random stream. So when only n_val_pos is
varied, any change in results is caused by the validation set alone.

Usage
-----
    python data.py                 # load fraud data (needs internet), print counts
    python data.py --synthetic     # stand-in data, no internet
    python data.py --selftest      # run the asserts on synthetic data
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


@dataclass
class Data:
    X_pool: np.ndarray
    y_pool: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    name: str = ""

    @property
    def pool_pos(self) -> int:
        return int(self.y_pool.sum())

    @property
    def pool_neg(self) -> int:
        return int((self.y_pool == 0).sum())


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def labels_to_binary(target) -> np.ndarray:
    """Turn a two-valued label array (e.g. 'False'/'True' or '0'/'1', possibly quoted)
    into 0/1, where 1 = the RARER label (the minority/fraud class)."""
    y_str = np.char.strip(np.char.replace(np.asarray(target).astype(str), "'", ""))
    labels, counts = np.unique(y_str, return_counts=True)
    print("label values found:", dict(zip(labels.tolist(), counts.tolist())))
    if len(labels) != 2:
        raise ValueError(f"expected 2 label values, found {labels}")
    minority = labels[np.argmin(counts)]
    y = (y_str == minority).astype(int)
    if y.sum() == 0:
        raise ValueError("no positives found; label parsing failed")
    return y


def load_fraud(use_cache: bool = True):
    """Credit-card fraud data from OpenML (data_id=42397). Needs internet the
    first time; afterwards it is read from ./cache/fraud.npz."""
    path = os.path.join(CACHE_DIR, "fraud.npz")
    if use_cache and os.path.exists(path):
        z = np.load(path)
        return z["X"], z["y"]

    # Optional fallback: a local copy of creditcard.csv (column "Class", 1 = fraud)
    csv = os.path.join(CACHE_DIR, "creditcard.csv")
    if os.path.exists(csv):
        import pandas as pd

        df = pd.read_csv(csv)
        X = df.drop(columns="Class").to_numpy(dtype="float64")
        y = df["Class"].to_numpy(dtype=int)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, X=X, y=y)
        return X, y

    from sklearn.datasets import fetch_openml

    # parser="liac-arff" avoids a scikit-learn/pandas version bug in the pandas parser.
    bunch = fetch_openml(data_id=42397, as_frame=False, parser="liac-arff")
    X = np.asarray(bunch.data, dtype="float64")
    y = labels_to_binary(bunch.target)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def load_synthetic(n_samples: int = 200_000, pos_rate: float = 0.0025, seed: int = 0):
    """Stand-in data with fraud-like imbalance, for testing the code offline.
    NOT for reporting results in the paper."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=12,
        n_redundant=4,
        weights=[1 - pos_rate, pos_rate],
        class_sep=1.0,
        flip_y=0.0,
        random_state=seed,
    )
    return X, y.astype(int)


def _cached(name, fetch):
    path = os.path.join(CACHE_DIR, f"{name}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["X"], z["y"]
    X, y = fetch()
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError(f"{name}: label parsing failed")
    print(f"{name}: {len(y):,} rows, {int(y.sum()):,} positives (1:{(len(y) - y.sum()) / y.sum():.0f})")
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def load_covtype():
    """Forest Covertype (UCI id 31). Positive = class 4 (Cottonwood/Willow), rest = negative."""
    def fetch():
        from sklearn.datasets import fetch_covtype

        d = fetch_covtype()
        return d.data, (d.target == 4).astype(int)
    return _cached("covtype", fetch)


def load_imblearn(name):
    """Dataset from the imbalanced-learn benchmark suite (Zenodo 61452), e.g. protein_homo."""
    def fetch():
        from imblearn.datasets import fetch_datasets

        d = fetch_datasets(filter_data=(name,))[name]
        return d.data, (d.target == 1).astype(int)
    return _cached(name, fetch)


def build_data(source: str = "fraud", test_size: float = 0.30, seed: int = 0) -> Data:
    """Load a dataset and cut the fixed stratified TEST set.

    source: fraud | covtype | protein_homo | synthetic | any other imbalanced-learn
    suite name (e.g. mammography, us_crime), written plainly."""
    if source == "fraud":
        X, y = load_fraud()
    elif source == "synthetic":
        X, y = load_synthetic()
    elif source == "covtype":
        X, y = load_covtype()
    else:
        X, y = load_imblearn(source)

    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return Data(X_pool, y_pool, X_test, y_test, name=source)


# --------------------------------------------------------------------------
# Controlled subsampling
# --------------------------------------------------------------------------
def max_total_positives(data: Data, ratio: int) -> int:
    """Largest n_train_pos + n_val_pos that the pool can support at this ratio."""
    return min(data.pool_pos, data.pool_neg // ratio)


def make_split(data: Data, n_train_pos: int, n_val_pos: int, ratio: int, seed: int):
    """Return (X_tr, y_tr, X_va, y_va) with exactly controlled class counts."""
    if n_train_pos < 1 or n_val_pos < 1:
        raise ValueError("n_train_pos and n_val_pos must be >= 1")
    if ratio < 1:
        raise ValueError("ratio must be >= 1")

    pos_idx = np.flatnonzero(data.y_pool == 1)
    neg_idx = np.flatnonzero(data.y_pool == 0)

    need_pos = n_train_pos + n_val_pos
    need_neg = need_pos * ratio
    if need_pos > len(pos_idx):
        raise ValueError(
            f"Not enough positives in pool: need {need_pos}, have {len(pos_idx)}."
        )
    if need_neg > len(neg_idx):
        raise ValueError(
            f"Not enough negatives in pool: need {need_neg}, have {len(neg_idx)}."
        )

    # TRAIN: own random stream -> identical for any n_val_pos at the same seed.
    rng_tr = np.random.default_rng([seed, 0])
    tr_pos = rng_tr.choice(pos_idx, size=n_train_pos, replace=False)
    tr_neg = rng_tr.choice(neg_idx, size=n_train_pos * ratio, replace=False)

    # VAL: separate stream, drawn only from rows not used by TRAIN.
    rng_va = np.random.default_rng([seed, 1])
    rest_pos = np.setdiff1d(pos_idx, tr_pos, assume_unique=True)
    rest_neg = np.setdiff1d(neg_idx, tr_neg, assume_unique=True)
    va_pos = rng_va.choice(rest_pos, size=n_val_pos, replace=False)
    va_neg = rng_va.choice(rest_neg, size=n_val_pos * ratio, replace=False)

    tr_idx = np.concatenate([tr_pos, tr_neg])
    va_idx = np.concatenate([va_pos, va_neg])
    rng_tr.shuffle(tr_idx)
    rng_va.shuffle(va_idx)

    return (
        data.X_pool[tr_idx],
        data.y_pool[tr_idx],
        data.X_pool[va_idx],
        data.y_pool[va_idx],
    )


def split_indices(data: Data, n_train_pos: int, n_val_pos: int, ratio: int, seed: int):
    """Same draw as make_split but returns pool row indices (used by the tests)."""
    pos_idx = np.flatnonzero(data.y_pool == 1)
    neg_idx = np.flatnonzero(data.y_pool == 0)
    rng_tr = np.random.default_rng([seed, 0])
    tr_pos = rng_tr.choice(pos_idx, size=n_train_pos, replace=False)
    tr_neg = rng_tr.choice(neg_idx, size=n_train_pos * ratio, replace=False)
    rng_va = np.random.default_rng([seed, 1])
    rest_pos = np.setdiff1d(pos_idx, tr_pos, assume_unique=True)
    rest_neg = np.setdiff1d(neg_idx, tr_neg, assume_unique=True)
    va_pos = rng_va.choice(rest_pos, size=n_val_pos, replace=False)
    va_neg = rng_va.choice(rest_neg, size=n_val_pos * ratio, replace=False)
    return np.concatenate([tr_pos, tr_neg]), np.concatenate([va_pos, va_neg])


def test_at_ratio(data: Data, ratio: int, seed: int = 12345):
    """TEST set re-sampled to the SAME ratio as TRAIN/VAL.

    Keeps ALL test positives and draws ratio * n_test_pos test negatives
    (fixed seed, so every run sees the identical test set).

    Why: F1-optimal thresholds depend on class prevalence. If TRAIN/VAL are drawn
    at 1:100 but TEST is at the natural 1:576, a threshold tuned on VAL does not
    transfer to TEST, and the comparison mixes prior shift with the effect we want
    to study. Use the natural TEST set only as a separate, clearly labelled analysis.
    """
    pos = np.flatnonzero(data.y_test == 1)
    neg = np.flatnonzero(data.y_test == 0)
    need = len(pos) * ratio
    if need > len(neg):
        raise ValueError(
            f"Test set has {len(neg)} negatives, need {need} for ratio 1:{ratio}."
        )
    rng = np.random.default_rng(seed)
    idx = np.concatenate([pos, rng.choice(neg, size=need, replace=False)])
    return data.X_test[idx], data.y_test[idx]


def describe(data: Data, ratios=(10, 50, 100, 200, 500)) -> None:
    n_test_pos = int(data.y_test.sum())
    print(f"Dataset: {data.name}")
    print(f"  pool : {len(data.y_pool):,} rows, {data.pool_pos:,} positives, "
          f"{data.pool_neg:,} negatives "
          f"(1:{data.pool_neg / max(data.pool_pos, 1):.0f})")
    print(f"  test : {len(data.y_test):,} rows, {n_test_pos:,} positives "
          f"(1:{(len(data.y_test) - n_test_pos) / max(n_test_pos, 1):.0f})")
    print("  feasible n_train_pos + n_val_pos by ratio (1:ratio):")
    for r in ratios:
        print(f"    1:{r:<4d} -> at most {max_total_positives(data, r)} positives in total")


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def selftest() -> None:
    data = build_data("synthetic")
    describe(data)
    ratio, n_tr, n_va = 100, 100, 40

    Xtr, ytr, Xva, yva = make_split(data, n_tr, n_va, ratio, seed=3)
    # 1. exact counts
    assert ytr.sum() == n_tr and (ytr == 0).sum() == n_tr * ratio
    assert yva.sum() == n_va and (yva == 0).sum() == n_va * ratio

    # 2. no overlap between TRAIN and VAL (pool indices)
    tr_idx, va_idx = split_indices(data, n_tr, n_va, ratio, seed=3)
    assert len(np.intersect1d(tr_idx, va_idx)) == 0
    assert len(np.unique(tr_idx)) == len(tr_idx) and len(np.unique(va_idx)) == len(va_idx)

    # 3. pool rows and TEST rows are disjoint by construction (separate arrays);
    #    check sizes add up to the full data
    assert len(data.y_pool) + len(data.y_test) == 200_000

    # 4. reproducible with the same seed, different with another seed
    a = make_split(data, n_tr, n_va, ratio, seed=3)
    b = make_split(data, n_tr, n_va, ratio, seed=3)
    c = make_split(data, n_tr, n_va, ratio, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])

    # 5. paired design: TRAIN is identical when only n_val_pos changes
    d = make_split(data, n_tr, 10, ratio, seed=3)
    assert np.array_equal(a[0], d[0]) and np.array_equal(a[1], d[1])
    assert d[3].sum() == 10

    # 6. clear error when the pool is too small
    try:
        make_split(data, 10_000, 10, ratio, seed=0)
    except ValueError as e:
        assert "Not enough" in str(e)
    else:
        raise AssertionError("expected ValueError")

    # 7. label parsing handles the formats seen so far
    for raw in (["False"] * 98 + ["True"] * 2, ["0"] * 98 + ["1"] * 2,
                ["'0'"] * 98 + ["'1'"] * 2):
        yy = labels_to_binary(raw)
        assert yy.sum() == 2 and len(yy) == 100

    print("data.py selftest: all checks passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        describe(build_data("synthetic" if args.synthetic else "fraud"))