"""
step3_retrain_models_B.py  —  Pipeline B: Imputed Unified Models
─────────────────────────────────────────────────────────────────
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

PHASE CLASSIFIERS use 58-dim mechanical features (no electrolyte/concentration)
because crystal structure depends on composition + processing, not test environment.
app.py calls classifiers with 58-dim base58 — must match exactly.

Output
──────
  models_B/               — .joblib files for app deployment
  imputed_dataset_B.xlsx  — full 2323-row imputed dataset with observed/imputed flags
  models_B/r2_report.txt  — leakage-free R² summary for paper

Usage:
    python3 step3_retrain_models_B.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, accuracy_score
from joblib import dump

warnings.filterwarnings('ignore')
print(f"scikit-learn version: {sklearn.__version__}")
print("Pipeline B — MissForest imputation + nested CV evaluation\n")

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

# 58-dim: used for mechanical regressors AND phase classifiers
MECH_FEATURES = ELEM_COLS + PROCESS_COLS + EMP_COLS + PHASE_COLS

# 66-dim: used for corrosion regressors
FEATURE_COLS  = MECH_FEATURES + ELECTROLYTES + ['conc_norm']

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
print(f"✓ Full imputation done — 0 missing values remaining")

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
corr_idx  = np.where(df['OG property'] == 'corrosion')[0]
all_idx   = np.arange(len(df))

# ── Nested CV evaluation (leakage-free R²) ────────────────────────────────────
print("\n" + "=" * 65)
print("  NESTED IMPUTATION EVALUATION  —  leakage-free R²")
print(f"  {N_FOLDS}-fold CV  ·  imputer fit on train fold only")
print("=" * 65)

feat_col_idx_66 = [IMP_COLS.index(c) for c in FEATURE_COLS]
feat_col_idx_58 = [IMP_COLS.index(c) for c in MECH_FEATURES]

def nested_r2(target_col, eval_pool, feat_idx, label):
    if target_col not in IMP_COLS:
        print(f"  [{label}] '{target_col}' not in IMP_COLS — skipping")
        return np.nan
    tgt_idx = IMP_COLS.index(target_col)

    raw_y = full_matrix[eval_pool, tgt_idx]
    obs_mask = ~np.isnan(raw_y)
    obs_rows  = eval_pool[obs_mask]

    if len(obs_rows) < N_FOLDS * 5:
        print(f"  [{label}] only {len(obs_rows)} observed rows — skipping")
        return np.nan

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_r2s = []
    for tr_local, te_local in kf.split(obs_rows):
        test_abs     = obs_rows[te_local]
        train_abs    = obs_rows[tr_local]
        non_test     = np.setdiff1d(all_idx, test_abs)

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
X_all_58 = full_imp_df[MECH_FEATURES].to_numpy(dtype=float)

def train_final(label, target_col, X, filename):
    y = clip_col(full_imp_df[target_col].to_numpy(dtype=float), target_col)
    rf = RandomForestRegressor(n_estimators=100, max_depth=50,
                               random_state=0, n_jobs=-1)
    rf.fit(X, y)
    out = os.path.join(MODEL_DIR, filename)
    dump(rf, out)
    print(f"  {label:<35} → {out}")

# Mechanical regressors use 66-dim (incl. electrolyte zeros for mech rows)
train_final('Hardness',       'Hardness (HVN)',                         X_all_66, 'hardness_regressor.joblib')
train_final('Yield Strength', 'Yield Strength (MPa)',                   X_all_66, 'yield_regressor.joblib')
train_final('Tensile',        'Ultimate Tensile Strength (MPa)',         X_all_66, 'tensile_regressor.joblib')
train_final('Elongation',     'Elongation (%)',                          X_all_66, 'elongation_regressor.joblib')
train_final('Ecorr',          'Corrosion potential (mV vs SCE)',         X_all_66, 'ecorr_regressor.joblib')
train_final('Epit',           'Pitting potential (mV vs SCE)',           X_all_66, 'epit_regressor.joblib')
train_final('icorr (log10)',  ICORR_LOG,                                 X_all_66, 'icorr_regressor.joblib')

# ── Phase classifiers — 58-dim ONLY ──────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE CLASSIFIERS  —  58-dim mechanical features")
print("  (app.py calls classifiers with 58-dim base58 — must match)")
print("=" * 65)
for phase in ['FCC', 'BCC', 'HCP', 'IM']:
    y_raw = df[phase].to_numpy(dtype=float)
    valid = ~np.isnan(y_raw)
    X_ph  = X_all_58[valid]
    y_ph  = y_raw[valid]
    X_tr, X_te, y_tr, y_te = train_test_split(X_ph, y_ph, test_size=0.1, random_state=14)
    clf = RandomForestClassifier(n_estimators=100, max_depth=50,
                                 random_state=4, oob_score=True)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    out = os.path.join(MODEL_DIR, f'{phase}_classifier.joblib')
    dump(clf, out)
    print(f"  [{phase}]  n={valid.sum()}  Accuracy={acc:.4f}  → {out}")

# ── R² summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  R² SUMMARY  —  leakage-free nested 5-fold CV")
print("=" * 65)
print(f"\n  {'Property':<35} {'R²':>8}  {'Note'}")
print("  " + "-" * 60)
for name, r2 in r2_nested.items():
    note = "log₁₀-scaled" if "icorr" in name else ""
    print(f"  {name:<35} {r2:>8.3f}  {note}")

report = ["Pipeline B — Leakage-free R² (nested 5-fold CV)\n",
          f"{'Property':<35} {'R²':>8}\n", "-"*45+"\n"]
for name, r2 in r2_nested.items():
    report.append(f"{name:<35} {r2:>8.3f}\n")
rpath = os.path.join(MODEL_DIR, 'r2_report.txt')
with open(rpath, 'w') as f:
    f.writelines(report)
print(f"\n  R² report saved → {rpath}")

print(f"""
✅  Pipeline B complete.

   Production models : {MODEL_DIR}/
   Imputed dataset   : {EXPORT_XLS}
   R² report         : {rpath}

   IMPORTANT — icorr convention
   ─────────────────────────────
   icorr model trained on log₁₀(icorr). App back-transforms via 10**pred.
   Do NOT change this without updating app.py.

   Phase classifiers trained on 58-dim mechanical features.
   App calls classifiers with 58-dim base58 — dimensions match.
""")
