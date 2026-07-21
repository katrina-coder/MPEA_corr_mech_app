"""
step2_retrain_models_A.py  —  Pipeline A: Separate Models  (v5)
──────────────────────────────────────────────────────────────
Mechanical models: 58 features (32 elem + 7 proc + 15 emp + 4 phase)
Corrosion models:  66 features (58 + 7 electrolyte one-hot + 1 concentration)
Phase classifiers: 54 features (32 elem + 7 proc + 15 emp)   ← FIX 1

Evaluation: same 5-fold CV as Pipeline B for fair comparison.
Final models trained on full observed subset (no imputation).

═══════════════════════════════════════════════════════════════════════════════
 FIX 1 — phase-classifier target leakage
   v4 trained the phase classifiers on MECH_FEATURES, which includes the four
   phase columns. The FCC classifier therefore received FCC as an input feature
   and simply read off the answer — hence Acc ≈ 100%. app.py then passed zeros
   in those four slots at inference (it cannot know the phase in advance), i.e.
   an input pattern the model had never seen, which is why it so often returned
   0 for all four phases on generated alloys.

   PHASE_CLF_FEATURES (54-dim) now excludes PHASE_COLS. Reported accuracies
   will drop to realistic values — that is the point. Classifier accuracy is
   now evaluated by 5-fold CV rather than a single 10% split, so the numbers
   are stable enough to quote.

   The mechanical/corrosion regressors still take the 4 phase flags as inputs.
   That is legitimate: phase is a genuine predictor of strength and ductility,
   and at inference app.py supplies it from these classifiers.
═══════════════════════════════════════════════════════════════════════════════
"""
import os, json
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (r2_score, accuracy_score, balanced_accuracy_score,
                             roc_auc_score, average_precision_score)
from joblib import dump

# Collected as we go, written to models_A/metrics.json at the end so app.py can
# display live numbers instead of a hard-coded table that goes stale each retrain.
METRICS = {'pipeline': 'A', 'regressors': {}, 'phase_classifiers': {}}

print(f"scikit-learn {sklearn.__version__}  |  Pipeline A v5 (leakage-free phase classifiers)\n")

DB_PATH   = 'MPEAs_Mech_Corr_DB_updated.xlsx'
MODEL_DIR = 'models_A'
N_FOLDS   = 5
os.makedirs(MODEL_DIR, exist_ok=True)

ELEM_COLS    = ['Ag','Al','B','C','Ca','Co','Cr','Cu','Fe','Ga','Ge','Hf','Li','Mg','Mn','Mo','N','Nb','Nd','Ni','Pd','Re','Sc','Si','Sn','Ta','Ti','V','W','Y','Zn','Zr']
PROCESS_COLS = ['process_1','process_2','process_3','process_4','process_5','process_6','process_7']
EMP_COLS     = ['a','delta','Tm','std of Tm','entropy','enthalpy','std of enthalpy','omega','X','std of X','VEC','std of vec','K','std of K','density']
PHASE_COLS   = ['FCC','BCC','HCP','IM']
ELECTROLYTES = ['NaCl','H2SO4','Seawater','HNO3','NaOH','HCl','KOH']

# FIX 1 — 54-dim: everything a phase classifier is allowed to see.
# Crystal structure is predicted from composition + processing route only.
PHASE_CLF_FEATURES = ELEM_COLS + PROCESS_COLS + EMP_COLS                    # 54-dim

MECH_FEATURES = PHASE_CLF_FEATURES + PHASE_COLS                             # 58-dim
CORR_FEATURES = MECH_FEATURES + ELECTROLYTES + ['conc_norm']                # 66-dim

assert len(PHASE_CLF_FEATURES) == 54, len(PHASE_CLF_FEATURES)
assert len(MECH_FEATURES)      == 58, len(MECH_FEATURES)
assert len(CORR_FEATURES)      == 66, len(CORR_FEATURES)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_excel(DB_PATH)
for e in ELECTROLYTES:
    df[e] = (df['Electrolyte'] == e).astype(float)
df['conc_norm'] = df['Concentration in M'].fillna(0) / 6.0  # 0 = no electrolyte (mechanical rows)

mech_df = df[df['OG property'] == 'mechanical'].copy()
corr_df = df[(df['OG property'] == 'corrosion') & (df['Electrolyte'].isin(ELECTROLYTES))].copy()
print(f"Mechanical rows: {len(mech_df)},  Corrosion rows (7 electrolytes): {len(corr_df)}\n")
print(f"Feature dims:  PhaseClf={len(PHASE_CLF_FEATURES)}, "
      f"Mechanical={len(MECH_FEATURES)},  Corrosion={len(CORR_FEATURES)}")

with open(f'{MODEL_DIR}/feature_config.json','w') as f:
    json.dump({'phase_clf_features': PHASE_CLF_FEATURES,   # ← consumed by app.py
               'mech_features': MECH_FEATURES,
               'corr_features': CORR_FEATURES,
               'electrolytes': ELECTROLYTES,
               'phase_cols': PHASE_COLS}, f, indent=2)


def cv_r2(X, y, n_folds=N_FOLDS):
    """5-fold CV R² — same protocol as Pipeline B for fair comparison."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    for tr, te in kf.split(X):
        rf = RandomForestRegressor(n_estimators=100, max_depth=50,
                                   random_state=0, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        scores.append(r2_score(y[te], rf.predict(X[te])))
    return float(np.mean(scores)), float(np.std(scores))


def train_reg(name, sub, target, feats, log_scale=False, fname=None):
    """Evaluate with 5-fold CV, then train final model on ALL observed rows."""
    X = sub[feats].fillna(0).to_numpy(dtype=float)
    y = sub[target].replace(0, np.nan).to_numpy(dtype=float)
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    if log_scale:
        pos = y > 0
        X, y = X[pos], y[pos]
        y = np.log10(y)

    mean_r2, std_r2 = cv_r2(X, y)

    rf = RandomForestRegressor(n_estimators=100, max_depth=50,
                               random_state=0, n_jobs=-1)
    rf.fit(X, y)
    out = f"{MODEL_DIR}/{fname or name + '.joblib'}"
    dump(rf, out)
    print(f"  [{name:15s}] n={len(y):4d}  R²={mean_r2:.4f} ± {std_r2:.4f}  → {out}")
    METRICS['regressors'][name] = {
        'n': int(len(y)), 'cv_r2_mean': mean_r2, 'cv_r2_std': std_r2,
        'log_scale': bool(log_scale), 'model_path': out}
    return mean_r2


def train_clf(name, sub, target, feats, fname=None):
    """FIX 1 — trained on PHASE_CLF_FEATURES (54-dim), which excludes the phase
    columns. Scored on out-of-fold predictions from stratified 5-fold CV.

    Raw accuracy is reported but should NOT be quoted on its own: HCP is
    positive in under 2% of rows, so "never predict HCP" already scores ~0.98.
    Balanced accuracy, ROC-AUC and average precision are all insensitive to
    that imbalance and are the numbers worth putting in a paper.
    """
    X = sub[feats].fillna(0).to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid].astype(int)

    oof_pred  = np.zeros(len(y))
    oof_proba = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        c = RandomForestClassifier(n_estimators=100, max_depth=50,
                                   class_weight='balanced_subsample',
                                   random_state=4, n_jobs=-1)
        c.fit(X[tr], y[tr])
        oof_pred[te] = c.predict(X[te])
        proba = c.predict_proba(X[te])
        oof_proba[te] = proba[:, 1] if proba.shape[1] > 1 else float(c.classes_[0])

    pos_rate = float(y.mean())
    baseline = float(max(pos_rate, 1 - pos_rate))
    acc      = float(accuracy_score(y, oof_pred))
    bal_acc  = float(balanced_accuracy_score(y, oof_pred))
    try:
        auc   = float(roc_auc_score(y, oof_proba))
        avg_p = float(average_precision_score(y, oof_proba))
    except ValueError:
        auc = avg_p = float('nan')

    rf = RandomForestClassifier(n_estimators=100, max_depth=50,
                                class_weight='balanced_subsample',
                                random_state=4, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    out = f"{MODEL_DIR}/{fname or name + '_classifier.joblib'}"
    dump(rf, out)

    flag = "" if acc > baseline else "  <-- BELOW majority baseline"
    print(f"  [{name:15s}] n={len(y):4d}  acc={acc:.4f} (baseline {baseline:.4f}){flag}")
    print(f"  {'':15s}   bal_acc={bal_acc:.4f}  ROC-AUC={auc:.4f}  avg_prec={avg_p:.4f}  → {out}")

    METRICS['phase_classifiers'][name] = {
        'n': int(len(y)), 'positive_rate': pos_rate,
        'accuracy': acc, 'majority_baseline': baseline,
        'beats_baseline': bool(acc > baseline),
        'balanced_accuracy': bal_acc, 'roc_auc': auc, 'average_precision': avg_p,
        'model_path': out}
    return acc


print("\n--- Mechanical regressors (5-fold CV) ---")
r2_hardness = train_reg('Hardness',       mech_df, 'Hardness (HVN)',                  MECH_FEATURES, fname='hardness_regressor.joblib')
r2_yield    = train_reg('Yield Strength', mech_df, 'Yield Strength (MPa)',            MECH_FEATURES, fname='yield_regressor.joblib')
r2_tensile  = train_reg('Tensile',        mech_df, 'Ultimate Tensile Strength (MPa)', MECH_FEATURES, fname='tensile_regressor.joblib')
r2_elong    = train_reg('Elongation',     mech_df, 'Elongation (%)',                  MECH_FEATURES, fname='elongation_regressor.joblib')

print("--- Corrosion regressors (5-fold CV) ---")
r2_ecorr = train_reg('Ecorr', corr_df, 'Corrosion potential (mV vs SCE)',         CORR_FEATURES, fname='ecorr_regressor.joblib')
r2_epit  = train_reg('Epit',  corr_df, 'Pitting potential (mV vs SCE)',           CORR_FEATURES, fname='epit_regressor.joblib')
r2_icorr = train_reg('icorr', corr_df, 'Corrosion current density (microA/cm2)',  CORR_FEATURES, log_scale=True, fname='icorr_regressor.joblib')

print("--- Phase classifiers (54-dim, leakage-free, stratified 5-fold CV) ---")
acc = {}
for p in PHASE_COLS:
    acc[p] = train_clf(p, df, p, PHASE_CLF_FEATURES, fname=f'{p}_classifier.joblib')

print(f"\n✅ Pipeline A saved to: {MODEL_DIR}")
print(f"\n{'=' * 55}")
print(f"  R² SUMMARY (5-fold CV — same protocol as Pipeline B)")
print(f"{'=' * 55}")
for name, r2 in [('Hardness', r2_hardness), ('Yield Strength', r2_yield),
                 ('Tensile', r2_tensile), ('Elongation', r2_elong),
                 ('Ecorr', r2_ecorr), ('Epit', r2_epit), ('icorr (log10)', r2_icorr)]:
    print(f"  {name:<22} {r2:.3f}")
print(f"\n  PHASE CLASSIFIERS (leakage-free, out-of-fold)")
print(f"  {'Phase':<10}{'acc':>8}{'baseline':>10}{'bal_acc':>10}{'ROC-AUC':>10}{'avg_prec':>10}")
for p in PHASE_COLS:
    m = METRICS['phase_classifiers'][p]
    print(f"  {p:<10}{m['accuracy']:>8.3f}{m['majority_baseline']:>10.3f}"
          f"{m['balanced_accuracy']:>10.3f}{m['roc_auc']:>10.3f}{m['average_precision']:>10.3f}")
print("\n  NOTE: these replace the old ~1.00 accuracies, which were an artefact")
print("        of each phase column being present in its own feature vector.")
print("        Do not quote raw accuracy for HCP — the majority baseline is ~0.98.")

METRICS['config'] = {
    'n_estimators_reg': 100, 'n_estimators_clf': 100, 'min_samples_leaf': 1,
    'class_weight': 'balanced_subsample', 'n_folds': N_FOLDS,
    'imputation': 'none', 'phase_features': 'ground-truth phase columns',
    'concentration_fill': 'zero',
}
with open(f'{MODEL_DIR}/metrics.json', 'w') as f:
    json.dump(METRICS, f, indent=2)
print(f"\n  metrics.json → {MODEL_DIR}/metrics.json  (app.py reads this)")
