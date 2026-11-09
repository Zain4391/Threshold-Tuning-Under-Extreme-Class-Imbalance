"""
exp_strat.py  --  Step 5: imbalance-handling strategies versus threshold tuning.

Same settings as exp_ac.py (--vary train|val|ratio|ratio_n). For every setting and seed the SAME
TRAIN and VALIDATION sets are used by every strategy (paired design), and the TEST set is the
ratio-matched one.

Strategies (applied to TRAIN only; VALIDATION and TEST are never resampled):
  none    plain model
  cw      class weights ("balanced")
  rus     random undersampling to 1:1
  ros     random oversampling to 1:1
  smote   SMOTE to 1:1
  bsmote  Borderline-SMOTE to 1:1      (optional)
  stomek  SMOTE + Tomek links          (optional)
  adasyn  ADASYN                       (optional)

For each strategy we report F1 at threshold 0.5 and at the validation-tuned threshold.
Key paired comparison: F1 of a strategy at 0.5 minus F1 of the plain model with a tuned
threshold ("vs_none_tuned"). > 0 means resampling beats threshold tuning.

Examples
--------
    python exp_strat.py --vary ratio_n --seeds 30 --out results_S_D
    python exp_strat.py --vary ratio --grid 20 50 100 200 --seeds 30 --out results_S_C
    python exp_strat.py --vary train --source covtype --seeds 30 --out results_S_A_cov
    python exp_strat.py --vary ratio_n --strategies none cw rus ros smote bsmote stomek adasyn
    python exp_strat.py --vary train --source synthetic --seeds 2      # quick test
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from data import build_data, describe, make_split, max_total_positives, test_at_ratio
from exp_ac import DEFAULTS, XLABEL, bootstrap_ci, make_settings
from pipeline import fit_model, score_row

ALL = ["none", "cw", "rus", "ros", "smote", "bsmote", "stomek", "adasyn"]
DEFAULT_STRATS = ["none", "cw", "rus", "ros", "smote"]
LABEL = {"none": "plain", "cw": "class weights", "rus": "undersampling", "ros": "oversampling",
         "smote": "SMOTE", "bsmote": "Borderline-SMOTE", "stomek": "SMOTE-Tomek", "adasyn": "ADASYN"}


def resample(strategy, X, y, seed):
    """Return (X, y, class_weight). Raises on impossible cases (caller skips them)."""
    if strategy == "none":
        return X, y, None
    if strategy == "cw":
        return X, y, "balanced"
    n_pos = int(y.sum())
    k = max(1, min(5, n_pos - 1))
    if strategy == "rus":
        from imblearn.under_sampling import RandomUnderSampler
        s = RandomUnderSampler(random_state=seed)
    elif strategy == "ros":
        from imblearn.over_sampling import RandomOverSampler
        s = RandomOverSampler(random_state=seed)
    elif strategy == "smote":
        from imblearn.over_sampling import SMOTE
        s = SMOTE(k_neighbors=k, random_state=seed)
    elif strategy == "bsmote":
        from imblearn.over_sampling import BorderlineSMOTE
        s = BorderlineSMOTE(k_neighbors=k, m_neighbors=10, random_state=seed)
    elif strategy == "stomek":
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
        s = SMOTETomek(smote=SMOTE(k_neighbors=k, random_state=seed), random_state=seed)
    elif strategy == "adasyn":
        from imblearn.over_sampling import ADASYN
        s = ADASYN(n_neighbors=k, random_state=seed)
    else:
        raise ValueError(strategy)
    Xr, yr = s.fit_resample(X, y)
    return Xr, yr, None


def run(data, models, strategies, settings, seeds, vary, out_csv):
    tests = {}
    for _, _, _, r in settings:
        if r not in tests:
            tests[r] = test_at_ratio(data, r)
            print(f"  test set at 1:{r}: {len(tests[r][1]):,} rows, {int(tests[r][1].sum())} positives")
    rows, t0, warned = [], time.time(), set()
    for i, seed in enumerate(seeds, 1):
        for x, ntr, nva, r in settings:
            X_tr, y_tr, X_va, y_va = make_split(data, ntr, nva, r, seed)
            X_te, y_te = tests[r]
            for strat in strategies:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        Xr, yr, cw = resample(strat, X_tr, y_tr, seed)
                except Exception as e:  # e.g. too few neighbours
                    if (strat, x) not in warned:
                        print(f"  WARNING: {strat} failed at x={x}: {str(e)[:80]} (skipped)")
                        warned.add((strat, x))
                    continue
                for model_name in models:
                    model = fit_model(model_name, Xr, yr, seed, class_weight=cw)
                    meta = dict(model=model_name, strategy=strat, vary=vary, x=x,
                                n_train_pos=ntr, n_val_pos=nva, ratio=r, seed=seed)
                    rows.append(score_row(model, X_va, y_va, X_te, y_te, meta))
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        el = time.time() - t0
        print(f"  seed {seed} done ({i}/{len(seeds)}), elapsed {el:.0f}s, "
              f"eta {el / i * (len(seeds) - i):.0f}s", flush=True)
    return pd.DataFrame(rows)


def summarise(df, n_boot=2000):
    base = (df[df.strategy == "none"].set_index(["model", "x", "seed"])["f1_tuned"]
            if "none" in set(df.strategy) else None)
    recs = []
    for (model, x, strat), g in df.groupby(["model", "x", "strategy"]):
        lo, hi = bootstrap_ci(g["benefit_f1"].to_numpy(), n_boot)
        rec = dict(model=model, x=x, strategy=strat, runs=len(g),
                   n_train_pos=int(g.n_train_pos.iloc[0]), n_val_pos=int(g.n_val_pos.iloc[0]),
                   f1_default=g.f1_default.mean(), f1_tuned=g.f1_tuned.mean(),
                   benefit_f1=g.benefit_f1.mean(), benefit_lo=lo, benefit_hi=hi,
                   threshold_mean=g.chosen_threshold.mean(),
                   pr_auc=g.pr_auc.mean(), brier=g.brier.mean(), ece=g.ece.mean(),
                   cost50_default=g.cost_c50_default.mean(), cost50_tuned=g.cost_c50_tuned.mean(),
                   cost50_ctuned=g.cost_c50_ctuned.mean())
        if base is not None:
            ref = base.reindex(list(zip(g.model, g.x, g.seed))).to_numpy()
            d = g.f1_default.to_numpy() - ref          # strategy at 0.5 minus plain model, tuned
            ok = ~np.isnan(d)
            rec["vs_none_tuned"] = float(d[ok].mean()) if ok.any() else np.nan
            if ok.sum() > 1:
                rec["vs_lo"], rec["vs_hi"] = bootstrap_ci(d[ok], n_boot)
        recs.append(rec)
    out = pd.DataFrame(recs)
    out["_o"] = out.strategy.map({s: i for i, s in enumerate(ALL)})
    return out.sort_values(["model", "x", "_o"]).drop(columns="_o").reset_index(drop=True)


def make_plots(summary, outdir, vary, fixed_text, tag):
    for m, g in summary.groupby("model"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, col, ttl in zip(axes, ["f1_default", "f1_tuned"], ["threshold 0.5", "tuned threshold"]):
            for s, h in g.groupby("strategy"):
                h = h.sort_values("x")
                ax.plot(h.x, h[col], marker="o", label=LABEL.get(s, s))
            ax.set_xscale("log")
            ax.set_xticks(sorted(summary.x.unique()))
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.set_xlabel(XLABEL[vary])
            ax.set_title(ttl, fontsize=9)
        axes[0].set_ylabel("Test F1")
        axes[1].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"fig_{tag}_{m}.png"), dpi=300)
        plt.close(fig)
    print("\nCaption text you can copy:")
    print(f"  Fig. (3). Test F1 of each imbalance-handling strategy at threshold 0.5 (left) and at the "
          f"validation-tuned threshold (right) versus {XLABEL[vary].split(' (')[0].lower()} ({fixed_text}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vary", required=True, choices=["train", "val", "ratio", "ratio_n"])
    ap.add_argument("--source", default="fraud")
    ap.add_argument("--models", nargs="+", default=["lr", "rf", "hgb"])
    ap.add_argument("--strategies", nargs="+", default=DEFAULT_STRATS, choices=ALL)
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
    ntr, nva = args.n_train_pos or d.get("n_train_pos"), args.n_val_pos or d.get("n_val_pos")
    ntr_rows, nva_rows = args.n_train_rows or d.get("n_train_rows"), args.n_val_rows or d.get("n_val_rows")
    tag = "S_" + {"train": "A", "val": "B", "ratio": "C", "ratio_n": "D"}[args.vary]
    out = args.out or f"results_{tag}_{args.source}"
    os.makedirs(out, exist_ok=True)

    data = build_data(args.source)
    describe(data)
    settings = make_settings(args.vary, grid, ntr, nva, ratio, ntr_rows, nva_rows)
    ok = [s for s in settings if s[1] + s[2] <= max_total_positives(data, s[3])]
    for s in settings:
        if s not in ok:
            print(f"  skipping x={s[0]}: needs {s[1] + s[2]} positives at 1:{s[3]}, "
                  f"pool supports {max_total_positives(data, s[3])}")
    if not ok:
        raise SystemExit("No feasible setting. Use --grid / --n-train-pos / --n-val-pos with smaller values.")
    settings = ok

    fixed = {"train": f"ratio 1:{ratio}, {nva} validation positives",
             "val": f"ratio 1:{ratio}, {ntr} training positives",
             "ratio": f"{ntr} training and {nva} validation positives",
             "ratio_n": f"{ntr_rows} training rows and {nva_rows} validation rows"}[args.vary]
    print(f"\nStrategy experiment {tag} on {args.source}: x={[s[0] for s in settings]}, fixed: {fixed}, "
          f"strategies {args.strategies}, models {args.models}, {args.seeds} seeds\n")

    df = run(data, args.models, args.strategies, settings, list(range(args.seeds)), args.vary,
             os.path.join(out, f"raw_{tag}.csv"))
    summary = summarise(df, args.boot)
    summary.to_csv(os.path.join(out, f"summary_{tag}.csv"), index=False)

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    cols = ["model", "x", "strategy", "n_train_pos", "f1_default", "f1_tuned", "benefit_f1",
            "benefit_lo", "benefit_hi", "vs_none_tuned", "vs_lo", "vs_hi",
            "cost50_default", "cost50_tuned", "cost50_ctuned", "ece"]
    cols = [c for c in cols if c in summary.columns]
    print(f"\nSUMMARY {tag} on {args.source}; paste this table back:")
    print(summary[cols].round(4).to_string(index=False))

    make_plots(summary, out, args.vary, fixed, tag)
    print(f"\nFiles in ./{out}/: raw_{tag}.csv, summary_{tag}.csv, fig_{tag}_lr.png, fig_{tag}_rf.png")