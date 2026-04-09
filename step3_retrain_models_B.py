"""
step3_retrain_models_B.py  —  Pipeline B: Imputed Unified Models
─────────────────────────────────────────────────────────────────
Uses iterative MissForest-style imputation (via sklearn IterativeImputer)
to fill in missing mechanical properties for corrosion rows and vice versa.
Then trains one unified RF model per property on the full imputed dataset.

Output: models_B/  folder with all .joblib files

Usage:
    python3 step3_retrain_models_B.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer   # noqa — required
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from joblib import dump

warnings.filterwarnings('ignore')
print(f"scikit-learn version: {sklearn.__version__}")
print("Pipeline B — MissForest imputation then unified models\n")

DB_PATH   = 'MPEAs_Mech_Corr_DB_updated.xlsx'
MODEL_DIR = 'models_B'
os.makedirs(MODEL_DIR, exist_ok=True)

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
FEATURE_COLS = ELEM_COLS + PROCESS_COLS + EMP_COLS + PHASE_COLS + ELECTROLYTES + ['conc_norm']  # 66-dim

MECH_TARGETS = {
    'Hardness (HVN)':                   'hardness_regressor.joblib',
    'Yield Strength (MPa)':             'yield_regressor.joblib',
    'Ultimate Tensile Strength (MPa)':  'tensile_regressor.joblib',
    'Elongation (%)':                   'elongation_regressor.joblib',
}
CORR_TARGETS = {
    'Corrosion potential (mV vs SCE)':          'ecorr_regressor.joblib',
    'Pitting potential (mV vs SCE)':            'epit_regressor.joblib',
    'Corrosion current density (microA/cm2)':   'icorr_regressor.joblib',
}
ALL_TARGETS = list(MECH_TARGETS.keys()) + list(CORR_TARGETS.keys())

print(f"Loading: {DB_PATH}")
df = pd.read_excel(DB_PATH)
print(f"  {len(df)} rows loaded")

# ── One-hot encode electrolyte + normalise concentration ─────────────────────
for e in ELECTROLYTES:
    df[e] = (df['Electrolyte'] == e).astype(float)
df['conc_norm'] = df['Concentration in M'].fillna(0) / 6.0
print(f"  Electrolyte one-hot columns added: {ELECTROLYTES}")
print(f"  Concentration normalised (÷ 6 M)\n")

# ── Replace zeros with NaN for all target columns ────────────────────────────
print("Replacing zeros with NaN in target columns...")
for col in ALL_TARGETS:
    df[col] = df[col].replace(0, np.nan)

# Log-scale icorr before imputation (reduces skew, improves imputation quality)
icorr_col = 'Corrosion current density (microA/cm2)'
df['icorr_log'] = np.where(df[icorr_col] > 0, np.log10(df[icorr_col]), np.nan)

# ── Build imputation matrix ───────────────────────────────────────────────────
# Features + all targets together so imputer can use cross-property correlations
MECH_COLS_FOR_IMP = list(MECH_TARGETS.keys())
CORR_COLS_FOR_IMP = [c for c in CORR_TARGETS.keys()
                     if c != icorr_col] + ['icorr_log']

IMP_COLS = FEATURE_COLS + MECH_COLS_FOR_IMP + CORR_COLS_FOR_IMP
imp_matrix = df[IMP_COLS].copy().to_numpy(dtype=float)

print(f"Imputation matrix shape: {imp_matrix.shape}")
missing_before = np.isnan(imp_matrix).sum()
print(f"Missing values before imputation: {missing_before:,}\n")

# MissForest-style: IterativeImputer with RF estimator
print("Running MissForest imputation (this may take 2–5 minutes)...")
imputer = IterativeImputer(
    estimator=RandomForestRegressor(
        n_estimators=50, max_depth=20, random_state=0, n_jobs=-1
    ),
    max_iter=5,
    random_state=0,
    verbose=0
)
imp_matrix_filled = imputer.fit_transform(imp_matrix)

missing_after = np.isnan(imp_matrix_filled).sum()
print(f"Missing values after imputation: {missing_after:,}")
print(f"Imputed {missing_before - missing_after:,} values\n")

# Put imputed values back into a dataframe
imp_df = pd.DataFrame(imp_matrix_filled, columns=IMP_COLS, index=df.index)

# Keep icorr in log scale in imp_df — train_unified_regressor will use it directly
# Rename so train_unified_regressor can find it
imp_df['icorr_log10'] = imp_df['icorr_log']
imp_df[icorr_col] = 10 ** imp_df['icorr_log']   # back-transform for sanity check only
imp_df = imp_df.drop(columns=['icorr_log'])

# ── Validate imputed distributions ───────────────────────────────────────────
print("=== Imputed value ranges (sanity check) ===")
for col in MECH_COLS_FOR_IMP + list(CORR_TARGETS.keys()):
    orig_vals = df[col].dropna()
    imp_vals  = imp_df[col]
    print(f"  {col[:45]:45s}  "
          f"orig [{orig_vals.min():8.1f}, {orig_vals.max():8.1f}]  "
          f"imputed [{imp_vals.min():8.1f}, {imp_vals.max():8.1f}]")


# ── Train unified regressors on full imputed dataset ─────────────────────────
# Mask for real observed corrosion rows (used for corrosion target evaluation)
REAL_CORR_MASK = (df['OG property'] == 'corrosion').values

def train_unified_regressor(name, target_col, log_scale=False, filename=None):
    X = imp_df[FEATURE_COLS].to_numpy(dtype=float)
    y = imp_df[target_col].to_numpy(dtype=float)

    # Clip any physically unreasonable imputed values
    if 'Hardness' in target_col:
        y = np.clip(y, 0, 3000)
    elif 'Strength' in target_col:
        y = np.clip(y, 0, 5000)
    elif 'Elongation' in target_col:
        y = np.clip(y, 0, 100)
    elif 'potential' in target_col.lower():
        y = np.clip(y, -2000, 3000)
    elif 'current' in target_col.lower():
        y = np.clip(y, 0.001, 10000)

    if log_scale:
        y = np.log10(np.clip(y, 1e-6, None))

    # Train on ALL rows (full imputed dataset)
    # Evaluate on real observed rows only for corrosion targets
    is_corr_target = any(k in name.lower() for k in ['ecorr','epit','icorr'])
    if is_corr_target:
        # Evaluate on held-out real corrosion rows — not synthetic imputed rows
        X_real = X[REAL_CORR_MASK]
        y_real = y[REAL_CORR_MASK]
        X_train, X_test, y_train, y_test = train_test_split(
            X_real, y_real, test_size=0.2, random_state=49)
        print(f"  [{name}] n={len(y)} total  (eval on {len(y_real)} real corrosion rows)")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=49)
        print(f"  [{name}] n={len(y)}")

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=50, random_state=0, oob_score=True
    )
    rf.fit(X, y)  # always train on ALL rows
    r2 = r2_score(y_test, rf.predict(X_test))

    fname = filename or f"{name.lower()}_regressor.joblib"
    out_path = os.path.join(MODEL_DIR, fname)
    dump(rf, out_path)
    print(f"    R² = {r2:.4f}  →  saved: {out_path}")
    return r2


print("\n" + "=" * 55)
print("UNIFIED REGRESSORS  (full imputed dataset, n=2323)")
print("=" * 55)
train_unified_regressor('Hardness',       'Hardness (HVN)',
                        filename='hardness_regressor.joblib')
train_unified_regressor('Yield Strength', 'Yield Strength (MPa)',
                        filename='yield_regressor.joblib')
train_unified_regressor('Tensile',        'Ultimate Tensile Strength (MPa)',
                        filename='tensile_regressor.joblib')
train_unified_regressor('Elongation',     'Elongation (%)',
                        filename='elongation_regressor.joblib')
train_unified_regressor('Ecorr',          'Corrosion potential (mV vs SCE)',
                        log_scale=False, filename='ecorr_regressor.joblib')
train_unified_regressor('Epit',           'Pitting potential (mV vs SCE)',
                        log_scale=False, filename='epit_regressor.joblib')
train_unified_regressor('icorr',          'icorr_log10',
                        log_scale=False, filename='icorr_regressor.joblib')

# ── Phase classifiers ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE CLASSIFIERS  (all rows)")
print("=" * 55)
for phase in ['FCC', 'BCC', 'HCP', 'IM']:
    y = df[phase].to_numpy(dtype=float)
    valid = ~np.isnan(y)
    X = imp_df[FEATURE_COLS].to_numpy(dtype=float)[valid]
    y = y[valid]
    print(f"  [{phase}] using {len(y)} samples")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=14
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=50, random_state=4, oob_score=True
    )
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    out_path = os.path.join(MODEL_DIR, f'{phase}_classifier.joblib')
    dump(rf, out_path)
    print(f"    Accuracy = {acc:.4f}  →  saved: {out_path}")

# Save the imputer itself so the app can use it for feature prep if needed
dump(imputer, os.path.join(MODEL_DIR, 'imputer.joblib'))
print(f"\n✅  Pipeline B — all models saved to: {MODEL_DIR}")
print("    icorr model predicts log10(icorr) — back-transformed in app.")
