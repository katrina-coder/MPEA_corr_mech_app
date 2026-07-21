"""
step6_ablation_C.py  —  Change one thing at a time and measure it
══════════════════════════════════════════════════════════════════
Pipeline C bundles five design decisions. Sherif's original changed all of them
at once, which means none of the resulting R² differences can be attributed to
any single choice. This script varies ONE decision at a time against the
Pipeline C baseline, so you can see what each is actually worth — and drop the
ones that aren't earning their place.

Variants tested
───────────────
  baseline                 step4's settings as shipped
  n_estimators = 300       Sherif's forest size (vs our 100)
  min_samples_leaf = 2     Sherif's regularisation (vs sklearn default 1)
  class_weight = None      turns off rare-phase balancing
  phase = ground truth     turns OFF stacking — regressors see the database's
                           true phase labels, as Pipelines A and B do. Expect
                           R² to go UP; that rise is the optimism that stacking
                           removes, NOT a real improvement.
  conc fill = zero         the old behaviour: missing molarity becomes 0 M

Also reports INFERENCE COST, which is what actually matters for tree count:
NSGA-II calls .predict() roughly pop_size x n_generations times per run, so
prediction latency — not training time — is what the user waits through.

Usage
─────
    python3 step6_ablation_C.py                  # all variants
    python3 step6_ablation_C.py --quick          # 3 folds, faster
    python3 step6_ablation_C.py --only baseline "n_estimators = 300"

Output: ablation_results.csv + a summary table on stdout.
Runtime: roughly 10-25 min for the full set; the 300-tree variant dominates.
"""

import argparse
import contextlib
import io
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

import step4_retrain_models_C as C          # config constants are patched in place


# ══════════════════════════════════════════════════════════════════════════════
VARIANTS = {
    'baseline':               {},
    'n_estimators = 300':     {'N_ESTIMATORS_REG': 300, 'N_ESTIMATORS_CLF': 300},
    'min_samples_leaf = 2':   {'MIN_SAMPLES_LEAF': 2},
    'class_weight = None':    {'CLASS_WEIGHT': None},
    'phase = ground truth':   {'_phase_source': 'true'},
    'conc fill = zero':       {'_conc_fill': 'zero'},
}

CONFIG_KEYS = ['N_ESTIMATORS_REG', 'N_ESTIMATORS_CLF', 'MIN_SAMPLES_LEAF',
               'CLASS_WEIGHT', 'N_FOLDS']

# Representative NSGA-II workload: 50 population x 200 generations
NSGA_EVALS = 50 * 200


def build_dataframe(conc_fill):
    """conc_fill='median' → per-electrolyte median (Pipeline C).
       conc_fill='zero'   → missing molarity becomes 0 M (old behaviour)."""
    df, medians = C.load_dataframe()
    if conc_fill == 'zero':
        conc = pd.to_numeric(df['Concentration in M'], errors='coerce').fillna(0.0)
        df['conc_norm'] = conc / C.MAX_CONCENTRATION
    return df, medians


def run_variant(name, overrides, quick=False):
    saved = {k: getattr(C, k) for k in CONFIG_KEYS}
    saved_dir = C.MODEL_DIR
    tmp = tempfile.mkdtemp(prefix='ablation_')

    phase_source = overrides.pop('_phase_source', 'oof')
    conc_fill    = overrides.pop('_conc_fill', 'median')

    try:
        for k, v in overrides.items():
            setattr(C, k, v)
        if quick:
            C.N_FOLDS = 3
        C.MODEL_DIR = tmp

        df, _ = build_dataframe(conc_fill)
        X54 = C.base_matrix(df)

        t0 = time.perf_counter()
        # Classifiers are always trained (we report their metrics either way);
        # phase_source only decides what the REGRESSORS are fed.
        with contextlib.redirect_stdout(io.StringIO()):
            oof_pred, _oof_proba, phase_metrics = C.train_phase_classifiers(df, X54)

        if phase_source == 'true':
            phase_feat = df[C.PHASE_COLS].fillna(0).to_numpy(dtype=float)
        else:
            phase_feat = oof_pred

        X58 = np.hstack([X54, phase_feat])
        X66 = np.hstack([X58,
                         df[C.ELECTROLYTES].to_numpy(dtype=float),
                         df[['conc_norm']].to_numpy(dtype=float)])

        is_mech = df['OG property'] == 'mechanical'
        is_corr = (df['OG property'] == 'corrosion') & df['Electrolyte'].isin(C.ELECTROLYTES)

        reg = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for prop, (col, fname, logs) in C.MECH_TARGETS.items():
                m = C.train_regressor(prop, col, X58, df[col], is_mech, fname, logs)
                if m: reg[prop] = m
            for prop, (col, fname, logs) in C.CORR_TARGETS.items():
                m = C.train_regressor(prop, col, X66, df[col], is_corr, fname, logs)
                if m: reg[prop] = m
        train_secs = time.perf_counter() - t0

        # ── Inference cost: what NSGA-II actually pays ───────────────────────
        from joblib import load
        batch = X58[:200]
        rf = load(os.path.join(tmp, 'hardness_regressor.joblib'))
        rf.predict(batch)                                    # warm up
        t1 = time.perf_counter()
        for _ in range(5):
            rf.predict(batch)
        per_row_ms = (time.perf_counter() - t1) / (5 * len(batch)) * 1000
        # 11 models hit per candidate (7 regressors + 4 classifiers)
        nsga_secs = per_row_ms / 1000 * NSGA_EVALS * 11

        row = {'variant': name, 'train_s': round(train_secs, 1),
               'est_NSGA_run_s': round(nsga_secs, 1)}
        for prop in ['Hardness', 'Yield Strength', 'Tensile', 'Elongation',
                     'Ecorr', 'Epit', 'icorr']:
            row[prop] = round(reg[prop]['cv_r2_mean'], 4) if prop in reg else np.nan
        row['mech_mean_R2'] = round(np.nanmean(
            [row[p] for p in ['Hardness', 'Yield Strength', 'Tensile', 'Elongation']]), 4)
        row['corr_mean_R2'] = round(np.nanmean(
            [row[p] for p in ['Ecorr', 'Epit', 'icorr']]), 4)
        for ph in C.PHASE_COLS:
            row[f'{ph}_balacc'] = round(phase_metrics[ph]['balanced_accuracy'], 4)
            row[f'{ph}_auc']    = round(phase_metrics[ph]['roc_auc'], 4)
        row['phase_mean_balacc'] = round(
            np.mean([phase_metrics[p]['balanced_accuracy'] for p in C.PHASE_COLS]), 4)
        return row

    finally:
        for k, v in saved.items():
            setattr(C, k, v)
        C.MODEL_DIR = saved_dir
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='3 folds instead of 5')
    ap.add_argument('--only', nargs='+', default=None, help='subset of variant names')
    ap.add_argument('--out', default='ablation_results.csv')
    args = ap.parse_args()

    names = args.only or list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variant(s): {unknown}\nAvailable: {list(VARIANTS)}")

    print(f"Ablation over {len(names)} variant(s)"
          f"{'  [quick: 3 folds]' if args.quick else ''}\n")

    rows = []
    for i, n in enumerate(names, 1):
        print(f"  [{i}/{len(names)}] {n} …", flush=True)
        rows.append(run_variant(n, dict(VARIANTS[n]), quick=args.quick))

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)

    base = res[res.variant == 'baseline']
    show = ['variant', 'mech_mean_R2', 'corr_mean_R2', 'phase_mean_balacc',
            'train_s', 'est_NSGA_run_s']

    print("\n" + "=" * 92)
    print("  ABLATION SUMMARY   (deltas are vs baseline)")
    print("=" * 92)
    print(res[show].to_string(index=False))

    if len(base):
        b = base.iloc[0]
        print("\n  Deltas vs baseline:")
        for _, r in res.iterrows():
            if r.variant == 'baseline':
                continue
            print(f"    {r.variant:<26} "
                  f"mech {r.mech_mean_R2 - b.mech_mean_R2:+.4f}   "
                  f"corr {r.corr_mean_R2 - b.corr_mean_R2:+.4f}   "
                  f"phase bal-acc {r.phase_mean_balacc - b.phase_mean_balacc:+.4f}   "
                  f"NSGA run x{r.est_NSGA_run_s / max(b.est_NSGA_run_s, 1e-9):.2f}")

    print(f"""
  How to read this
  ────────────────
  • A change is worth keeping if its R² delta clearly exceeds the fold-to-fold
    noise. Typical std across folds here is ~0.03-0.10, so a +0.005 gain is
    noise, not a result.
  • 'n_estimators = 300' should be judged on the NSGA run multiplier, not on
    training time. If it buys < 0.01 R² for ~3x slower optimisation, drop it.
  • 'class_weight = None' will likely RAISE raw accuracy while LOWERING
    balanced accuracy on HCP — that is the imbalance trap, visible directly.
  • 'phase = ground truth' will likely raise mechanical R². That is not a
    better model; it is the optimism stacking exists to remove, since the app
    can never supply true phase labels. Treat the gap as a measurement of how
    much A and B were over-reporting.

  Full per-property numbers: {args.out}
""")


if __name__ == '__main__':
    main()
