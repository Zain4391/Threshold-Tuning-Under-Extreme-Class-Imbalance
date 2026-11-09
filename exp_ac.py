"""
exp_ac.py  --  Experiments A, B and C in one script.

Vary exactly ONE quantity, keep the other two fixed:

  --vary train   Experiment A: vary TRAIN positives   (fixed ratio, fixed VAL positives)
  --vary val     Experiment B: vary VALIDATION positives (fixed ratio, fixed TRAIN positives)
  --vary ratio   Experiment C: vary imbalance RATIO   (fixed TRAIN and VAL positives)
  --vary ratio_n Experiment D: vary RATIO with fixed dataset SIZE (fixed TRAIN and VAL ROWS),
                 so the number of positives shrinks as the ratio grows, as in a benchmark
                 whose datasets all have the same size.

The TEST set is always re-sampled to the same ratio as TRAIN/VAL (all test
positives kept), so thresholds are never distorted by a change of prevalence.
benefit_f1 = F1(validation-tuned threshold) - F1(threshold 0.5); > 0 means tuning helps.

Examples (run from the folder containing data.py and pipeline.py)
--------
    python exp_ac.py --vary train --seeds 30 --out results_A
    python exp_ac.py --vary ratio --seeds 30 --out results_C
    python exp_ac.py --vary val   --seeds 30 --out results_B2
    python exp_ac.py --vary ratio_n --seeds 30 --out results_D
    python exp_ac.py --vary train --source synthetic --seeds 3     # quick test, no internet
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from data import build_data, describe, make_split, max_total_positives, test_at_ratio
from pipeline import COST_RATIOS, fit_model, score_row

DEFAULTS = {
    #          grid                           fixed values used for the other two quantities
    "train": dict(grid=[10, 20, 40, 80, 160], ratio=50, n_val_pos=80),
    "val":   dict(grid=[5, 10, 20, 40, 80], ratio=100, n_train_pos=100),
    "ratio": dict(grid=[10, 20, 50, 100, 200, 500], n_train_pos=100, n_val_pos=80),
    # rows are fixed; positives = rows / (1 + ratio). With 4000 + 3000 rows the lowest ratio (1:20)
    # needs about 333 positives, which the fraud pool (344) can just support.
    "ratio_n": dict(grid=[20, 50, 100, 200, 500], n_train_rows=4000, n_val_rows=3000),
}
XLABEL = {
    "train": "Training positives (log scale)",
    "val": "Validation positives (log scale)",
    "ratio": "Negatives per positive, 1:ratio (log scale)",
    "ratio_n": "Negatives per positive, 1:ratio (log scale)",
}


def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def make_settings(vary, grid, n_train_pos, n_val_pos, ratio, n_train_rows=None, n_val_rows=None):
    """List of (x, n_train_pos, n_val_pos, ratio) with exactly one quantity varying
    (for 'ratio_n': ratio varies and the number of ROWS is fixed, so positives change)."""
    out = []
    for x in grid:
        if vary == "train":
            out.append((x, x, n_val_pos, ratio))
        elif vary == "val":
            out.append((x, n_train_pos, x, ratio))
        elif vary == "ratio":
            out.append((x, n_train_pos, n_val_pos, x))
        else:  # ratio_n
            ntr = max(1, round(n_train_rows / (1 + x)))
            nva = max(1, round(n_val_rows / (1 + x)))
            out.append((x, ntr, nva, x))
    return out


def run(data, models, settings, seeds, vary, out_csv):
    tests = {}  # one fixed ratio-matched test set per ratio
    for _, _, _, r in settings:
        if r not in tests:
            tests[r] = test_at_ratio(data, r)
            yt = tests[r][1]
            print(f"  test set at 1:{r}: {len(yt):,} rows, {int(yt.sum())} positives")

    rows, t0 = [], time.time()
    for i, seed in enumerate(seeds, 1):
        fitted = {}  # (model, n_train_pos, ratio) -> fitted model; TRAIN does not depend on n_val_pos
        for model_name in models:
            for x, ntr, nva, r in settings:
                X_tr, y_tr, X_va, y_va = make_split(data, ntr, nva, r, seed)
                key = (model_name, ntr, r)
                if key not in fitted:
                    fitted[key] = fit_model(model_name, X_tr, y_tr, seed)
                X_te, y_te = tests[r]
                meta = dict(model=model_name, vary=vary, x=x, n_train_pos=ntr,
                            n_val_pos=nva, ratio=r, seed=seed)
                rows.append(score_row(fitted[key], X_va, y_va, X_te, y_te, meta))
        pd.DataFrame(rows).to_csv(out_csv, index=False)  # save after every seed
        el = time.time() - t0
        print(f"  seed {seed} done ({i}/{len(seeds)}), elapsed {el:.0f}s, "
              f"eta {el / i * (len(seeds) - i):.0f}s", flush=True)
    return pd.DataFrame(rows)


def summarise(df, n_boot=2000):
    recs = []
    for (model, x), g in df.groupby(["model", "x"]):
        lo, hi = bootstrap_ci(g["benefit_f1"].to_numpy(), n_boot)
        rec = {
            "model": model, "x": x, "runs": len(g),
            "n_train_pos": int(g["n_train_pos"].iloc[0]), "n_val_pos": int(g["n_val_pos"].iloc[0]),
            "f1_default_mean": g["f1_default"].mean(),
            "f1_tuned_mean": g["f1_tuned"].mean(),
            "benefit_f1_mean": g["benefit_f1"].mean(),
            "benefit_f1_sd": g["benefit_f1"].std(ddof=1),
            "benefit_f1_ci_lo": lo, "benefit_f1_ci_hi": hi,
            "threshold_mean": g["chosen_threshold"].mean(),
            "threshold_sd": g["chosen_threshold"].std(ddof=1),
            "pr_auc_mean": g["pr_auc"].mean(),
            "brier_mean": g["brier"].mean(),
            "ece_mean": g["ece"].mean(),
        }
        for k in COST_RATIOS:
            rec[f"benefit_cost_c{k}_mean"] = g[f"benefit_cost_c{k}"].mean()
            rec[f"benefit_ctuned_c{k}_mean"] = g[f"benefit_ctuned_c{k}"].mean()
        recs.append(rec)
    return pd.DataFrame(recs).sort_values(["model", "x"]).reset_index(drop=True)


def _axis(ax, summary, vary):
    ax.set_xscale("log")
    ax.set_xticks(sorted(summary["x"].unique()))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(XLABEL[vary])


def make_plots(summary, outdir, vary, fixed_text, tag):
    colors = {"lr": "#1f77b4", "rf": "#d62728", "hgb": "#2ca02c"}
    names = {"lr": "Logistic regression", "rf": "Random Forest", "hgb": "Gradient boosting"}

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m, g in summary.groupby("model"):
        ax.plot(g["x"], g["benefit_f1_mean"], marker="o", color=colors.get(m), label=names.get(m, m))
        ax.fill_between(g["x"], g["benefit_f1_ci_lo"], g["benefit_f1_ci_hi"],
                        color=colors.get(m), alpha=0.2)
    ax.axhline(0, color="black", lw=0.8)
    _axis(ax, summary, vary)
    ax.set_ylabel("F1 gain from threshold tuning")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig_{tag}_benefit.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m, g in summary.groupby("model"):
        ax.plot(g["x"], g["f1_default_mean"], marker="o", color=colors.get(m),
                label=f"{names.get(m, m)}, threshold 0.5")
        ax.plot(g["x"], g["f1_tuned_mean"], marker="s", ls="--", color=colors.get(m),
                label=f"{names.get(m, m)}, tuned")
    _axis(ax, summary, vary)
    ax.set_ylabel("Test F1")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig_{tag}_f1.png"), dpi=300)
    plt.close(fig)

    print("\nCaption text you can copy:")
    print(f"  Fig. (1). Mean F1 gain from validation-tuned thresholding versus {XLABEL[vary].split(' (')[0].lower()} "
          f"({fixed_text}). Shaded bands are 95% bootstrap confidence intervals over seeds.")
    print(f"  Fig. (2). Test F1 at the default threshold (solid) and at the validation-tuned threshold "
          f"(dashed) versus {XLABEL[vary].split(' (')[0].lower()} ({fixed_text}).")


def describe_results(summary):
    print("\nWhat the table shows (descriptive only, no causal claim):")
    for m, g in summary.groupby("model"):
        g = g.sort_values("x")
        best = g.loc[g["benefit_f1_mean"].idxmax()]
        pos = g[g["benefit_f1_ci_lo"] > 0]
        print(f"  {m}: largest mean F1 gain is {best.benefit_f1_mean:+.4f} at x={best.x:g} "
              f"(95% CI {best.benefit_f1_ci_lo:+.4f} to {best.benefit_f1_ci_hi:+.4f}).")
        if len(pos):
            print(f"      Gain is positive (CI above 0) at x = {', '.join(f'{v:g}' for v in pos.x)}.")
        else:
            print("      No setting has a gain whose CI lies above 0.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vary", required=True, choices=["train", "val", "ratio", "ratio_n"])
    ap.add_argument("--source", default="fraud")
    ap.add_argument("--models", nargs="+", default=["lr", "rf", "hgb"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--grid", type=int, nargs="+", default=None)
    ap.add_argument("--ratio", type=int, default=None)
    ap.add_argument("--n-train-pos", type=int, default=None)
    ap.add_argument("--n-val-pos", type=int, default=None)
    ap.add_argument("--n-train-rows", type=int, default=None)
    ap.add_argument("--n-val-rows", type=int, default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = DEFAULTS[args.vary]
    grid = sorted(args.grid or d["grid"])
    ratio = args.ratio or d.get("ratio")
    ntr = args.n_train_pos or d.get("n_train_pos")
    nva = args.n_val_pos or d.get("n_val_pos")
    ntr_rows = args.n_train_rows or d.get("n_train_rows")
    nva_rows = args.n_val_rows or d.get("n_val_rows")
    tag = {"train": "A", "val": "B", "ratio": "C", "ratio_n": "D"}[args.vary]
    out = args.out or f"results_{tag}"
    os.makedirs(out, exist_ok=True)

    data = build_data(args.source)
    describe(data)
    settings = make_settings(args.vary, grid, ntr, nva, ratio, ntr_rows, nva_rows)

    for x, a, b, r in settings:  # feasibility check before any work
        if a + b > max_total_positives(data, r):
            raise SystemExit(f"Setting x={x} needs {a + b} positives at 1:{r}; pool supports "
                             f"{max_total_positives(data, r)}. Shrink --grid or the fixed values.")

    fixed = {"train": f"ratio 1:{ratio}, {nva} validation positives",
             "val": f"ratio 1:{ratio}, {ntr} training positives",
             "ratio": f"{ntr} training and {nva} validation positives",
             "ratio_n": f"{ntr_rows} training rows and {nva_rows} validation rows"}[args.vary]
    print(f"\nExperiment {tag} (vary {args.vary}): grid {grid}, fixed: {fixed}, "
          f"models {args.models}, {args.seeds} seeds\n")

    df = run(data, args.models, settings, list(range(args.seeds)), args.vary,
             os.path.join(out, f"raw_{tag}.csv"))
    summary = summarise(df, args.boot)
    summary.to_csv(os.path.join(out, f"summary_{tag}.csv"), index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    show = ["model", "x", "n_train_pos", "n_val_pos", "runs", "f1_default_mean", "f1_tuned_mean", "benefit_f1_mean",
            "benefit_f1_ci_lo", "benefit_f1_ci_hi", "threshold_mean", "threshold_sd",
            "benefit_cost_c10_mean", "benefit_cost_c50_mean", "benefit_cost_c200_mean",
            "benefit_ctuned_c10_mean", "benefit_ctuned_c50_mean", "benefit_ctuned_c200_mean"]
    print(f"\nSUMMARY {tag} (x = {args.vary}); paste this table back:")
    print(summary[show].round(4).to_string(index=False))

    make_plots(summary, out, args.vary, fixed, tag)
    describe_results(summary)
    print(f"\nFiles in ./{out}/: raw_{tag}.csv, summary_{tag}.csv, fig_{tag}_benefit.png, fig_{tag}_f1.png")