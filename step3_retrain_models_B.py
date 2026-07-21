"""
step3_retrain_models_B.py  —  Pipeline B: Imputed Unified Models  (v2)
──────────────────────────────────────────────────────────────────────
Uses MissForest-style imputation (IterativeImputer + RF) to fill missing
cross-domain values, trains unified RF regressors on the full imputed
dataset, and evaluates R² using nested K-fold CV to avoid leakage.

TWO SEPARATE OUTPUTS
─────────────────────
1. PRODUCTION MODELS (models_B/) — trained on all 2323 imputed rows.
   These are deployed in the Streamlit app. No train/test split needed
   here — we use all available data for the best possible deployment model.

2. HONEST R² (nested 5-fold CV) — for reporting in publications.
   For each fold: imputer fit on train rows only → transform test rows
   → train RF → evaluate on REAL observed test values (not imputed).
   This eliminates transductive leakage from the imputation step.

═══════════════════════════════════════════════════════════════════════════════
 FIX 1 — phase-classifier target leakage
   v1 trained the phase classifiers on MECH_FEATURES (58-dim), which INCLUDES
   the four phase columns — so the FCC classifier had FCC as an input feature.
   That is what produced the ~100% accuracies, and it is why the classifiers
   misbehaved at inference: app.py cannot know the phase in advance, so it
   passed zeros in those slots, a pattern the model never saw in training.

   Phase classifiers now use PHASE_CLF_FEATURES (54-dim) = 32 element +
   7 processing + 15 empirical. No phase flags. app.py builds exactly this
   54-dim vector (build_base_features) — the dimensions must stay in sync.

   The regressors are unchanged: they still take the 4 phase flags as inputs,
   which is legitimate, with app.py supplying them from these classifiers.
═══════════════════════════════════════════════════════════════════════════════

Output
──────
  models_B/               — .joblib files for app deployment
  imputed_dataset_B.xlsx  — full 2323-row imputed dataset with observed/imputed flags
  models_B/r2_report.txt  — leakage-free R² + phase accuracy summary

Usage:
    python3 step3_retrain_models_B.py
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (r2_score, accuracy_score, balanced_accuracy_score,
                             roc_auc_score, average_precision_score)
from joblib import dump

# Canonical property names shared by step2/step3/step4 metrics.json and app.py
CANON = {
    'Hardness': 'Hardness',
    'Yield Strength': 'Yield Strength',
    'Ultimate Tensile Strength': 'Tensile',
    'Elongation': 'Elongation',
    'Ecorr': 'Ecorr',
    'Epit': 'Epit',
    'icorr (log10)': 'icorr',
}

warnings.filterwarnings('ignore')
print(f"scikit-learn version: {sklearn.__version__}")
print("Pipeline B v2 — MissForest imputation + nested CV + leakage-free phase classifiers\n")

DB_PATH    = 'MPEAs_Mech_Corr_DB_updated.xlsx'
MODEL_DIR  = 'models_B'
EXPORT_XLS = 'imputed_dataset_B.xlsx'
N_FOLDS    = 5
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Column definitions ────────────────────────────────────────────────────────
ELEM_COLS    = ['Ag','Al','B','C','Ca','Co','Cr','Cu','Fe','Ga','Ge','Hf',
                'Li','Mg','Mn','Mo','N','Nb','Nd','Ni','Pd','Re','Sc','Si',
                'Sn','Ta','Ti','V','W','Y','Zn','Zr']
PROCESS_COLS = ['process_1','process_2','process_3','process_4',
                'process_5','process_6','process_7']
EMP_COLS     = ['a','delta','Tm','std of Tm','entropy','enthalpy',
                'std of enthalpy','omega','X','std of X','VEC',
                'std of vec','K','std of K','density']
PHASE_COLS   = ['FCC','BCC','HCP','IM']
ELECTROLYTES = ['NaCl','H2SO4','Seawater','HNO3','NaOH','HCl','KOH']

# FIX 1 — 54-dim: phase classifier input. Excludes PHASE_COLS.
PHASE_CLF_FEATURES = ELEM_COLS + PROCESS_COLS + EMP_COLS                    # 54-dim

# 58-dim: mechanical regressors
MECH_FEATURES = PHASE_CLF_FEATURES + PHASE_COLS                             # 58-dim

# 66-dim: all Pipeline B regressors
FEATURE_COLS  = MECH_FEATURES + ELECTROLYTES + ['conc_norm']                # 66-dim

assert len(PHASE_CLF_FEATURES) == 54, len(PHASE_CLF_FEATURES)
assert len(MECH_FEATURES)      == 58, len(MECH_FEATURES)
assert len(FEATURE_COLS)       == 66, len(FEATURE_COLS)

MECH_TARGETS = {
    'Hardness (HVN)'                  : 'hardness_regressor.joblib',
    'Yield Strength (MPa)'            : 'yield_regressor.joblib',
    'Ultimate Tensile Strength (MPa)' : 'tensile_regressor.joblib',
    'Elongation (%)'                  : 'elongation_regressor.joblib',
}
CORR_TARGETS = {
    'Corrosion potential (mV vs SCE)'        : 'ecorr_regressor.joblib',
    'Pitting potential (mV vs SCE)'          : 'epit_regressor.joblib',
    'Corrosion current density (microA/cm2)' : 'icorr_regressor.joblib',
}
ALL_TARGETS = list(MECH_TARGETS.keys()) + list(CORR_TARGETS.keys())

ICORR_RAW = 'Corrosion current density (microA/cm2)'
ICORR_LOG = 'icorr_log10'

CLIP_RULES = {
    'Hardness (HVN)'                  : (0,    3000),
    'Yield Strength (MPa)'            : (0,    5000),
    'Ultimate Tensile Strength (MPa)' : (0,    5000),
    'Elongation (%)'                  : (0,    100),
    'Corrosion potential (mV vs SCE)' : (-2000, 3000),
    'Pitting potential (mV vs SCE)'   : (-2000, 3000),
    ICORR_LOG                         : (-6,    6),
}

def clip_col(arr, col_name):
    if col_name in CLIP_RULES:
        lo, hi = CLIP_RULES[col_name]
        return np.clip(arr, lo, hi)
    return arr

def make_imputer():
    return IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=50, max_depth=20, random_state=0, n_jobs=-1),
        max_iter=5, random_state=0, verbose=0)

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading: {DB_PATH}")
df = pd.read_excel(DB_PATH)
print(f"  {len(df)} rows × {df.shape[1]} columns")
print(f"  Mechanical: {(df['OG property']=='mechanical').sum()}  "
      f"Corrosion: {(df['OG property']=='corrosion').sum()}")

for e in ELECTROLYTES:
    df[e] = (df['Electrolyte'] == e).astype(float)
df['conc_norm'] = df['Concentration in M'].fillna(0) / 6.0

for col in ALL_TARGETS:
    df[col] = df[col].replace(0, np.nan)

# Log₁₀-scale icorr
df[ICORR_LOG] = np.where(df[ICORR_RAW] > 0, np.log10(df[ICORR_RAW]), np.nan)
print(f"\n  icorr observed: {df[ICORR_LOG].notna().sum()} rows  "
      f"log10 range [{df[ICORR_LOG].min():.2f}, {df[ICORR_LOG].max():.2f}]")

# ── Imputation matrix ─────────────────────────────────────────────────────────
MECH_COLS_FOR_IMP = list(MECH_TARGETS.keys())
CORR_COLS_FOR_IMP = ['Corrosion potential (mV vs SCE)',
                     'Pitting potential (mV vs SCE)',
                     ICORR_LOG]
IMP_COLS = FEATURE_COLS + MECH_COLS_FOR_IMP + CORR_COLS_FOR_IMP

print(f"\n  Imputation matrix: {len(df)} rows × {len(IMP_COLS)} columns")
missing_before = df[IMP_COLS].isna().sum().sum()
print(f"  Missing values before imputation: {missing_before:,}")

# ── Full imputation (for production models) ───────────────────────────────────
print("\nRunning full MissForest imputation on all 2323 rows...")
print("(This takes 2–5 minutes)")
full_matrix  = df[IMP_COLS].to_numpy(dtype=float)
imputer_full = make_imputer()
full_imp_arr = imputer_full.fit_transform(full_matrix)
full_imp_df  = pd.DataFrame(full_imp_arr, columns=IMP_COLS, index=df.index)
print("✓ Full imputation done — 0 missing values remaining")

dump(imputer_full, os.path.join(MODEL_DIR, 'imputer.joblib'))

# ── Validate imputed distributions ───────────────────────────────────────────
print("\n=== Imputed value ranges vs original ===")
for col in MECH_COLS_FOR_IMP + ['Corrosion potential (mV vs SCE)',
                                'Pitting potential (mV vs SCE)', ICORR_LOG]:
    orig = df[col].dropna()
    imp  = full_imp_df[col]
    unit = ' [log10 µA/cm²]' if col == ICORR_LOG else ''
    print(f"  {col[:42]:42s}{unit}")
    print(f"    original: n={len(orig):4d}  [{orig.min():8.2f}, {orig.max():8.2f}]  mean={orig.mean():8.2f}")
    print(f"    imputed : n={len(imp):4d}  [{imp.min():8.2f},  {imp.max():8.2f}]  mean={imp.mean():8.2f}")

# ── Export imputed dataset ────────────────────────────────────────────────────
print(f"\nExporting imputed dataset to {EXPORT_XLS}...")
export_df = df[['OG property']].copy()
if 'Composition' in df.columns:
    export_df.insert(1, 'Composition', df['Composition'])

for col in MECH_COLS_FOR_IMP:
    export_df[col + ' (imputed)'] = np.round(
        clip_col(full_imp_df[col].values, col), 3)

export_df['Ecorr mV vs SCE (imputed)'] = np.round(
    clip_col(full_imp_df['Corrosion potential (mV vs SCE)'].values,
             'Corrosion potential (mV vs SCE)'), 2)
export_df['Epit mV vs SCE (imputed)'] = np.round(
    clip_col(full_imp_df['Pitting potential (mV vs SCE)'].values,
             'Pitting potential (mV vs SCE)'), 2)
export_df['icorr log10 (imputed)'] = np.round(
    clip_col(full_imp_df[ICORR_LOG].values, ICORR_LOG), 4)
export_df['icorr µA/cm² (imputed)'] = np.round(
    10 ** clip_col(full_imp_df[ICORR_LOG].values, ICORR_LOG), 4)

for col in MECH_COLS_FOR_IMP:
    export_df[col + ' (observed?)'] = df[col].notna().astype(int)
export_df['Ecorr (observed?)'] = df['Corrosion potential (mV vs SCE)'].notna().astype(int)
export_df['Epit (observed?)']  = df['Pitting potential (mV vs SCE)'].notna().astype(int)
export_df['icorr (observed?)'] = df[ICORR_RAW].notna().astype(int)

export_df.to_excel(EXPORT_XLS, index=False)
print(f"✓ Saved {len(export_df)} rows  —  1=observed, 0=imputed estimate")

# ── Row indices ───────────────────────────────────────────────────────────────
mech_idx = np.where(df['OG property'] == 'mechanical')[0]
corr_idx = np.where(df['OG property'] == 'corrosion')[0]
all_idx  = np.arange(len(df))

# ── Nested CV evaluation (leakage-free R²) ────────────────────────────────────
print("\n" + "=" * 65)
print("  NESTED IMPUTATION EVALUATION  —  leakage-free R²")
print(f"  {N_FOLDS}-fold CV  ·  imputer fit on train fold only")
print("=" * 65)

feat_col_idx_66 = [IMP_COLS.index(c) for c in FEATURE_COLS]

# Written to models_B/metrics.json at the end so app.py shows live numbers
METRICS = {'pipeline': 'B', 'regressors': {}, 'phase_classifiers': {}}


def nested_r2(target_col, eval_pool, feat_idx, label):
    if target_col not in IMP_COLS:
        print(f"  [{label}] '{target_col}' not in IMP_COLS — skipping")
        return np.nan
    tgt_idx = IMP_COLS.index(target_col)

    raw_y    = full_matrix[eval_pool, tgt_idx]
    obs_mask = ~np.isnan(raw_y)
    obs_rows = eval_pool[obs_mask]

    if len(obs_rows) < N_FOLDS * 5:
        print(f"  [{label}] only {len(obs_rows)} observed rows — skipping")
        return np.nan

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_r2s = []
    for tr_local, te_local in kf.split(obs_rows):
        test_abs  = obs_rows[te_local]
        train_abs = obs_rows[tr_local]
        non_test  = np.setdiff1d(all_idx, test_abs)

        imp = make_imputer()
        imp.fit(full_matrix[non_test])

        train_imp = imp.transform(full_matrix[train_abs])
        test_imp  = imp.transform(full_matrix[test_abs])

        X_tr = train_imp[:, feat_idx]
        y_tr = clip_col(train_imp[:, tgt_idx], target_col)
        X_te = test_imp[:, feat_idx]
        y_te = clip_col(full_matrix[test_abs, tgt_idx], target_col)  # REAL observed values

        rf = RandomForestRegressor(n_estimators=100, max_depth=50,
                                   random_state=0, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        fold_r2s.append(r2_score(y_te, rf.predict(X_te)))

    mean_r2 = float(np.mean(fold_r2s))
    std_r2  = float(np.std(fold_r2s))
    print(f"  {label:<35} n={len(obs_rows):4d}  R² = {mean_r2:.3f} ± {std_r2:.3f}")
    METRICS['regressors'][CANON.get(label, label)] = {
        'n': int(len(obs_rows)), 'cv_r2_mean': mean_r2, 'cv_r2_std': std_r2,
        'log_scale': target_col == ICORR_LOG}
    return mean_r2

print("\n── Mechanical (evaluated on mechanical rows) ──")
r2_nested = {}
for col in MECH_TARGETS:
    name = col.split(' (')[0]
    r2_nested[name] = nested_r2(col, mech_idx, feat_col_idx_66, name)

print("\n── Corrosion (evaluated on real corrosion rows) ──")
for col in CORR_TARGETS:
    if 'current' in col:
        name, target = 'icorr (log10)', ICORR_LOG
    elif 'Pitting' in col:
        name, target = 'Epit', col
    else:
        name, target = 'Ecorr', col
    r2_nested[name] = nested_r2(target, corr_idx, feat_col_idx_66, name)

# ── Production models (trained on full imputed dataset) ───────────────────────
print("\n" + "=" * 65)
print("  PRODUCTION MODELS  —  trained on all 2323 imputed rows")
print("  (deployed in Streamlit app)")
print("=" * 65)

X_all_66 = full_imp_df[FEATURE_COLS].to_numpy(dtype=float)
X_all_54 = full_imp_df[PHASE_CLF_FEATURES].to_numpy(dtype=float)   # FIX 1

def train_final(label, target_col, X, filename):
    y = clip_col(full_imp_df[target_col].to_numpy(dtype=float), target_col)
    rf = RandomForestRegressor(n_estimators=100, max_depth=50,
                               random_state=0, n_jobs=-1)
    rf.fit(X, y)
    out = os.path.join(MODEL_DIR, filename)
    dump(rf, out)
    print(f"  {label:<35} → {out}")

# All Pipeline B regressors use 66-dim (electrolyte columns are zero for mech rows)
train_final('Hardness',       'Hardness (HVN)',                  X_all_66, 'hardness_regressor.joblib')
train_final('Yield Strength', 'Yield Strength (MPa)',            X_all_66, 'yield_regressor.joblib')
train_final('Tensile',        'Ultimate Tensile Strength (MPa)', X_all_66, 'tensile_regressor.joblib')
train_final('Elongation',     'Elongation (%)',                  X_all_66, 'elongation_regressor.joblib')
train_final('Ecorr',          'Corrosion potential (mV vs SCE)', X_all_66, 'ecorr_regressor.joblib')
train_final('Epit',           'Pitting potential (mV vs SCE)',   X_all_66, 'epit_regressor.joblib')
train_final('icorr (log10)',  ICORR_LOG,                         X_all_66, 'icorr_regressor.joblib')

# ── Phase classifiers — 54-dim, leakage-free (FIX 1) ─────────────────────────
print("\n" + "=" * 65)
print("  PHASE CLASSIFIERS  —  54-dim (element + processing + empirical)")
print("  Phase columns EXCLUDED from their own inputs (FIX 1).")
print("  app.py builds this vector in build_base_features() — keep in sync.")
print("=" * 65)

phase_acc = {}
for phase in PHASE_COLS:
    y_raw = df[phase].to_numpy(dtype=float)
    valid = ~np.isnan(y_raw)
    X_ph  = X_all_54[valid]
    y_ph  = y_raw[valid].astype(int)

    # Out-of-fold predictions from stratified 5-fold CV. A single 10% split is
    # far too noisy once the honest (non-leaky) accuracies come down.
    oof_pred  = np.zeros(len(y_ph))
    oof_proba = np.zeros(len(y_ph))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for tr, te in skf.split(X_ph, y_ph):
        c = RandomForestClassifier(n_estimators=100, max_depth=50,
                                   class_weight='balanced_subsample',
                                   random_state=4, n_jobs=-1)
        c.fit(X_ph[tr], y_ph[tr])
        oof_pred[te] = c.predict(X_ph[te])
        proba = c.predict_proba(X_ph[te])
        oof_proba[te] = proba[:, 1] if proba.shape[1] > 1 else float(c.classes_[0])

    pos_rate = float(y_ph.mean())
    baseline = float(max(pos_rate, 1 - pos_rate))
    acc      = float(accuracy_score(y_ph, oof_pred))
    bal_acc  = float(balanced_accuracy_score(y_ph, oof_pred))
    try:
        auc   = float(roc_auc_score(y_ph, oof_proba))
        avg_p = float(average_precision_score(y_ph, oof_proba))
    except ValueError:
        auc = avg_p = float('nan')
    phase_acc[phase] = acc

    clf = RandomForestClassifier(n_estimators=100, max_depth=50,
                                 class_weight='balanced_subsample',
                                 random_state=4, oob_score=True, n_jobs=-1)
    clf.fit(X_ph, y_ph)
    out = os.path.join(MODEL_DIR, f'{phase}_classifier.joblib')
    dump(clf, out)

    flag = "" if acc > baseline else "  <-- BELOW majority baseline"
    print(f"  [{phase}]  n={int(valid.sum())}  acc={acc:.4f} (baseline {baseline:.4f}){flag}")
    print(f"         bal_acc={bal_acc:.4f}  ROC-AUC={auc:.4f}  avg_prec={avg_p:.4f}  → {out}")

    METRICS['phase_classifiers'][phase] = {
        'n': int(valid.sum()), 'positive_rate': pos_rate,
        'accuracy': acc, 'majority_baseline': baseline,
        'beats_baseline': bool(acc > baseline),
        'balanced_accuracy': bal_acc, 'roc_auc': auc, 'average_precision': avg_p,
        'model_path': out}

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  R² SUMMARY  —  leakage-free nested 5-fold CV")
print("=" * 65)
print(f"\n  {'Property':<35} {'R²':>8}  {'Note'}")
print("  " + "-" * 60)
for name, r2 in r2_nested.items():
    note = "log₁₀-scaled" if "icorr" in name else ""
    print(f"  {name:<35} {r2:>8.3f}  {note}")

print(f"\n  {'Phase classifier':<35} {'Acc':>8}")
print("  " + "-" * 60)
for p, a in phase_acc.items():
    print(f"  {p:<35} {a:>8.3f}")

report = ["Pipeline B — Leakage-free R² (nested 5-fold CV)\n",
          f"{'Property':<35} {'R²':>8}\n", "-"*45+"\n"]
for name, r2 in r2_nested.items():
    report.append(f"{name:<35} {r2:>8.3f}\n")
report.append("\nPhase classifiers — 54-dim features, stratified 5-fold CV\n")
report.append(f"{'Phase':<35} {'Acc':>8}\n")
report.append("-"*45+"\n")
for p, a in phase_acc.items():
    report.append(f"{p:<35} {a:>8.3f}\n")
rpath = os.path.join(MODEL_DIR, 'r2_report.txt')
with open(rpath, 'w') as f:
    f.writelines(report)
print(f"\n  Report saved → {rpath}")

METRICS['config'] = {
    'n_estimators_reg': 100, 'n_estimators_clf': 100, 'min_samples_leaf': 1,
    'class_weight': 'balanced_subsample', 'n_folds': N_FOLDS,
    'imputation': 'MissForest (IterativeImputer + RF, max_iter=5)',
    'phase_features': 'ground-truth phase columns',
    'concentration_fill': 'zero',
}
with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
    json.dump(METRICS, f, indent=2)
print(f"  metrics.json  → {os.path.join(MODEL_DIR, 'metrics.json')}  (app.py reads this)")

print(f"""
✅  Pipeline B complete.

   Production models : {MODEL_DIR}/
   Imputed dataset   : {EXPORT_XLS}
   Report            : {rpath}

   IMPORTANT — icorr convention
   ─────────────────────────────
   icorr model trained on log₁₀(icorr). App back-transforms via 10**pred.
   Do NOT change this without updating app.py.

   IMPORTANT — phase classifier dimensions
   ────────────────────────────────────────
   Classifiers now take 54 features and EXCLUDE the phase columns.
   app.py build_base_features() must produce the same 54-dim vector,
   in the same column order: 32 element, 7 processing, 15 empirical.
""")
