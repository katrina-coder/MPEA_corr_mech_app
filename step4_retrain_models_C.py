"""
step4_retrain_models_C.py  —  Pipeline C: Stacked, Leakage-Free, No Imputation
══════════════════════════════════════════════════════════════════════════════
Merged pipeline. What it takes from where:

  FROM SHERIF'S PIPELINE C
  ────────────────────────
  • Out-of-fold phase stacking. The strength/hardness/corrosion models are
    trained on PREDICTED phase labels, not the true ones from the database.
    This is the fix for the second, deeper leak: at deployment the app has no
    true phase, only a classifier's guess, so training on ground-truth phase
    inflated the reported R² relative to what users actually get.
  • No imputation anywhere. Corrosion models see only real corrosion
    measurements from supported electrolytes. No invented target values.
  • Electrolyte-median concentration fill. A NaCl test with unrecorded
    molarity is imputed as the median NaCl molarity (0.6 M), not as 0 M,
    which was physically meaningless.

  FROM OUR PIPELINE A/B FIXES
  ───────────────────────────
  • 54-dim phase classifier input. The four phase columns are DROPPED, not
    zeroed. Same leak closed, but no dead columns to feed by accident, and
    smaller/faster models. app.py's build_base_features() emits this exactly.

  DELIBERATE DEPARTURES FROM BOTH
  ────────────────────────────────
  • Final regressors are fit on OUT-OF-FOLD phase features, not full-fit ones.
    Sherif refits the final regressor using phase predictions from a classifier
    that had already seen those rows — near-perfect, so it quietly reintroduces
    optimism. On a genuinely new alloy the classifier's accuracy is its OOF
    accuracy, so OOF features are what training should match. See STACKING NOTE.
  • Pseudo-labelling dropped. In Sherif's run it moved mechanical R² by
    -0.011 to +0.000 — i.e. nothing — while adding a teacher/student stage and
    two hyperparameters. The pseudo-labels are the model's own predictions, so
    they carry no new information about hardness; at best they regularise.
    Use step6_ablation_C.py if you want to re-test that decision on your data.
  • Phase quality is reported with balanced accuracy, ROC-AUC and average
    precision alongside raw accuracy and the majority-class baseline.
    Raw accuracy is actively misleading for HCP: only ~1.9% of rows are
    positive, so "never predict HCP" scores ~98.1%.

Outputs
───────
  models_C/
      hardness_regressor.joblib      yield_regressor.joblib
      tensile_regressor.joblib       elongation_regressor.joblib
      ecorr_regressor.joblib         epit_regressor.joblib
      icorr_regressor.joblib         ← predicts log10(icorr); app does 10**pred
      FCC/BCC/HCP/IM_classifier.joblib   ← 54-dim input
      feature_config.json            metrics.json

Then:  cp generator_net_MPEA.pt models_C/

Usage:
    python3 step4_retrain_models_C.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — every number you might want to argue about lives here
# ══════════════════════════════════════════════════════════════════════════════
DB_PATH   = "MPEAs_Mech_Corr_DB_updated.xlsx"
MODEL_DIR = "models_C"
N_FOLDS      = 5
RANDOM_STATE = 42

# Trees: 100, not Sherif's 300. NSGA-II calls .predict() ~10,000 times per
# optimisation run (pop 50 x 200 generations), so tree count multiplies the
# latency the user actually waits through. 300 trees ≈ 3x slower optimisation
# for a typically ~0.005 R² gain. Measure it yourself with step6_ablation_C.py.
N_ESTIMATORS_REG = 100
N_ESTIMATORS_CLF = 100

# sklearn's default. min_samples_leaf=2 is mild regularisation — cheap, but
# it should earn its place on your data rather than be assumed. Ablation covers it.
MIN_SAMPLES_LEAF = 1

# KEPT, and not for accuracy. HCP is positive in ~1.9% of rows, so an unweighted
# forest learns "never predict HCP" and returns P(HCP) ≈ 0.02 for every alloy.
# That makes 'maximise HCP' a flat objective with nothing for NSGA-II to climb.
# Balancing restores usable spread. Cost: the probabilities are no longer
# calibrated to the true base rate — app.py labels them accordingly.
CLASS_WEIGHT = "balanced_subsample"

MAX_CONCENTRATION = 6.0     # normalisation divisor, matches app.py

# ══════════════════════════════════════════════════════════════════════════════
#  COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
ELEM_COLS = ['Ag','Al','B','C','Ca','Co','Cr','Cu','Fe','Ga','Ge','Hf',
             'Li','Mg','Mn','Mo','N','Nb','Nd','Ni','Pd','Re','Sc','Si',
             'Sn','Ta','Ti','V','W','Y','Zn','Zr']
PROCESS_COLS = ['process_1','process_2','process_3','process_4',
                'process_5','process_6','process_7']
EMP_COLS = ['a','delta','Tm','std of Tm','entropy','enthalpy',
            'std of enthalpy','omega','X','std of X','VEC',
            'std of vec','K','std of K','density']
PHASE_COLS   = ['FCC','BCC','HCP','IM']
ELECTROLYTES = ['NaCl','H2SO4','Seawater','HNO3','NaOH','HCl','KOH']

# 54-dim — phase classifier input. Phase columns are ABSENT, not zeroed.
PHASE_CLF_FEATURES = ELEM_COLS + PROCESS_COLS + EMP_COLS
# 58-dim — mechanical regressors: 54 + 4 PREDICTED phase flags
# 66-dim — corrosion regressors: 58 + 7 electrolyte one-hot + 1 concentration
assert len(PHASE_CLF_FEATURES) == 54, len(PHASE_CLF_FEATURES)

MECH_TARGETS = {
    'Hardness':       ('Hardness (HVN)',                  'hardness_regressor.joblib',   False),
    'Yield Strength': ('Yield Strength (MPa)',            'yield_regressor.joblib',      False),
    'Tensile':        ('Ultimate Tensile Strength (MPa)', 'tensile_regressor.joblib',    False),
    'Elongation':     ('Elongation (%)',                  'elongation_regressor.joblib', False),
}
CORR_TARGETS = {
    'Ecorr': ('Corrosion potential (mV vs SCE)',        'ecorr_regressor.joblib', False),
    'Epit':  ('Pitting potential (mV vs SCE)',          'epit_regressor.joblib',  False),
    'icorr': ('Corrosion current density (microA/cm2)', 'icorr_regressor.joblib', True),
}


def make_regressor(seed):
    return RandomForestRegressor(n_estimators=N_ESTIMATORS_REG, max_depth=50,
                                 min_samples_leaf=MIN_SAMPLES_LEAF,
                                 random_state=seed, n_jobs=-1)


def make_classifier(seed):
    return RandomForestClassifier(n_estimators=N_ESTIMATORS_CLF, max_depth=50,
                                  min_samples_leaf=MIN_SAMPLES_LEAF,
                                  class_weight=CLASS_WEIGHT,
                                  random_state=seed, n_jobs=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_dataframe():
    """Load the DB and build electrolyte one-hots + a physically sane conc_norm.

    Concentration fill (from Sherif): a corrosion test whose molarity was never
    recorded is far more likely to be at the typical molarity for that
    electrolyte than at 0 M. Filling 0 told the model 'pure water', which is
    wrong and put those rows in a region of feature space nothing else occupies.
    """
    df = pd.read_excel(DB_PATH)

    for e in ELECTROLYTES:
        df[e] = (df['Electrolyte'] == e).astype(float)

    conc = pd.to_numeric(df['Concentration in M'], errors='coerce')
    conc = conc.mask(conc <= 0, np.nan)          # 0 M is a missing value, not a reading

    medians = {}
    for e in ELECTROLYTES:
        vals = conc[df['Electrolyte'] == e].dropna()
        medians[e] = float(vals.median()) if len(vals) else 0.5

    filled = conc.copy()
    for e, med in medians.items():
        filled.loc[(df['Electrolyte'] == e) & filled.isna()] = med

    # Mechanical rows have no electrolyte at all and never reach a corrosion
    # model during training; 0 is a harmless placeholder for them.
    filled = filled.fillna(0.0)
    df['conc_norm'] = filled / MAX_CONCENTRATION

    return df, medians


def base_matrix(df):
    """54-dim feature matrix. Medians for missing empirical values.

    Every column is coerced with pd.to_numeric first. Hand-maintained
    workbooks routinely carry stray text or a literal 0 placeholder in
    otherwise-numeric columns, and a single such cell would otherwise make
    the whole column object-dtype and blow up .to_numpy(dtype=float).
    """
    frame = df[PHASE_CLF_FEATURES].apply(pd.to_numeric, errors='coerce')
    for col in EMP_COLS:
        vals = frame[col]
        frame[col] = vals.fillna(vals.median() if vals.notna().any() else 0.0)
    return frame.fillna(0.0).to_numpy(dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — PHASE CLASSIFIERS (level-0 models)
# ══════════════════════════════════════════════════════════════════════════════
def train_phase_classifiers(df, X54):
    """Train the four phase classifiers and return OUT-OF-FOLD predictions.

    Out-of-fold means: every row's phase prediction comes from a model that
    never saw that row during training. Those predictions are what the
    regressors get as input features in stage 2 — so the regressors learn
    against phase labels that are as noisy as the ones the app will hand them.
    """
    print("\n" + "=" * 72)
    print("  STAGE 1 — PHASE CLASSIFIERS  (54-dim input, phase columns absent)")
    print("=" * 72)

    oof_pred  = np.zeros((len(df), 4))
    oof_proba = np.zeros((len(df), 4))
    metrics   = {}

    for i, phase in enumerate(PHASE_COLS):
        y = pd.to_numeric(df[phase], errors='coerce').fillna(0).to_numpy(dtype=int)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for fold, (tr, te) in enumerate(skf.split(X54, y)):
            clf = make_classifier(RANDOM_STATE + i * 100 + fold)
            clf.fit(X54[tr], y[tr])
            oof_pred[te, i]  = clf.predict(X54[te])
            proba = clf.predict_proba(X54[te])
            oof_proba[te, i] = proba[:, 1] if proba.shape[1] > 1 else float(clf.classes_[0])

        # ── Honest metrics, computed on the out-of-fold predictions ──────────
        pos_rate = float(y.mean())
        baseline = float(max(pos_rate, 1 - pos_rate))   # "always guess the common class"
        acc      = float(accuracy_score(y, oof_pred[:, i]))
        bal_acc  = float(balanced_accuracy_score(y, oof_pred[:, i]))
        try:
            auc    = float(roc_auc_score(y, oof_proba[:, i]))
            avg_p  = float(average_precision_score(y, oof_proba[:, i]))
        except ValueError:                              # single class present
            auc = avg_p = float('nan')

        metrics[phase] = {
            'n': int(len(y)), 'positive_rate': pos_rate,
            'accuracy': acc, 'majority_baseline': baseline,
            'beats_baseline': bool(acc > baseline),
            'balanced_accuracy': bal_acc, 'roc_auc': auc, 'average_precision': avg_p,
        }

        flag = "" if acc > baseline else "   <-- BELOW the always-guess-majority baseline"
        print(f"\n  [{phase}]  n={len(y)}  positives={pos_rate:6.2%}")
        print(f"     accuracy          {acc:.4f}   (majority baseline {baseline:.4f}){flag}")
        print(f"     balanced accuracy {bal_acc:.4f}   (0.500 = no skill)")
        print(f"     ROC-AUC           {auc:.4f}   (0.500 = coin flip)")
        print(f"     avg precision     {avg_p:.4f}   (no-skill = {pos_rate:.4f})")

        # Final classifier: refit on everything, this is what ships
        final = make_classifier(RANDOM_STATE + 900 + i)
        final.fit(X54, y)
        out = os.path.join(MODEL_DIR, f'{phase}_classifier.joblib')
        dump(final, out)
        metrics[phase]['model_path'] = out

    print("\n  Reading these numbers:")
    print("    accuracy alone is misleading for rare phases — HCP is positive in")
    print("    <2% of rows, so 'never HCP' already scores ~0.98. Balanced accuracy,")
    print("    ROC-AUC and average precision are immune to that and are the ones")
    print("    to quote in a paper.")

    return oof_pred, oof_proba, metrics


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — REGRESSORS (level-1 models, fed predicted phases)
# ══════════════════════════════════════════════════════════════════════════════
#  STACKING NOTE
#  Both the cross-validation AND the final shipped model are fit on OUT-OF-FOLD
#  phase features. It is tempting to refit the final model using phase
#  predictions from the full classifier (Sherif does this), but that classifier
#  has already seen those rows and predicts them near-perfectly. On a genuinely
#  new alloy — which is the only kind the app ever sees — classifier accuracy is
#  the OOF accuracy. Training on OOF features is therefore the honest match, and
#  it keeps the shipped model consistent with the R² we report for it.
# ══════════════════════════════════════════════════════════════════════════════
def train_regressor(name, target_col, X, y_series, row_mask, filename,
                    log_scale=False):
    y = pd.to_numeric(y_series, errors='coerce').replace(0, np.nan)
    mask = row_mask & y.notna()
    if log_scale:
        mask &= (y > 0)

    idx = np.where(mask.to_numpy())[0]
    Xs  = X[idx]
    ys  = y.to_numpy(dtype=float)[idx]
    if log_scale:
        ys = np.log10(ys)

    if len(ys) < N_FOLDS * 4:
        print(f"  [{name:15s}] only {len(ys)} usable rows — SKIPPED")
        return None

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fold, (tr, te) in enumerate(kf.split(Xs)):
        rf = make_regressor(RANDOM_STATE + 200 + fold)
        rf.fit(Xs[tr], ys[tr])
        scores.append(r2_score(ys[te], rf.predict(Xs[te])))

    final = make_regressor(RANDOM_STATE + 1200)
    final.fit(Xs, ys)
    out = os.path.join(MODEL_DIR, filename)
    dump(final, out)

    mean_r2, std_r2 = float(np.mean(scores)), float(np.std(scores))
    print(f"  [{name:15s}] n={len(ys):4d}  R²={mean_r2:.4f} ± {std_r2:.4f}  → {out}")
    return {'n': int(len(ys)), 'cv_r2_mean': mean_r2, 'cv_r2_std': std_r2,
            'log_scale': bool(log_scale), 'model_path': out}


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"scikit-learn {sklearn.__version__}  |  Pipeline C (merged)\n")

    df, elec_medians = load_dataframe()
    print(f"Loaded {DB_PATH}: {len(df)} rows")
    print(f"  mechanical rows: {(df['OG property'] == 'mechanical').sum()}")
    print(f"  corrosion rows : {(df['OG property'] == 'corrosion').sum()}")
    print(f"  concentration medians used for missing molarity: {elec_medians}")

    X54 = base_matrix(df)
    print(f"  base feature matrix: {X54.shape}")

    oof_pred, oof_proba, phase_metrics = train_phase_classifiers(df, X54)

    # Stage-2 feature matrices, built on OUT-OF-FOLD phase predictions
    X58 = np.hstack([X54, oof_pred])
    X66 = np.hstack([X58,
                     df[ELECTROLYTES].to_numpy(dtype=float),
                     df[['conc_norm']].to_numpy(dtype=float)])
    assert X58.shape[1] == 58 and X66.shape[1] == 66

    is_mech = df['OG property'] == 'mechanical'
    is_corr = (df['OG property'] == 'corrosion') & df['Electrolyte'].isin(ELECTROLYTES)

    print("\n" + "=" * 72)
    print("  STAGE 2a — MECHANICAL REGRESSORS  (58-dim, predicted phases)")
    print("=" * 72)
    reg_metrics = {}
    for name, (col, fname, logs) in MECH_TARGETS.items():
        m = train_regressor(name, col, X58, df[col], is_mech, fname, logs)
        if m: reg_metrics[name] = m

    print("\n" + "=" * 72)
    print("  STAGE 2b — CORROSION REGRESSORS  (66-dim, real labels only)")
    print(f"  No imputation. Supported electrolytes only: {', '.join(ELECTROLYTES)}")
    print("=" * 72)
    for name, (col, fname, logs) in CORR_TARGETS.items():
        m = train_regressor(name, col, X66, df[col], is_corr, fname, logs)
        if m: reg_metrics[name] = m

    # ── Save config + metrics ────────────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, 'feature_config.json'), 'w') as f:
        json.dump({'phase_clf_features': PHASE_CLF_FEATURES,
                   'mech_features': PHASE_CLF_FEATURES + PHASE_COLS,
                   'corr_features': PHASE_CLF_FEATURES + PHASE_COLS + ELECTROLYTES + ['conc_norm'],
                   'electrolytes': ELECTROLYTES, 'phase_cols': PHASE_COLS,
                   'max_concentration': MAX_CONCENTRATION}, f, indent=2)

    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump({
            'pipeline': 'C',
            'description': 'Stacked out-of-fold phase features, no imputation, '
                           '54-dim leakage-free phase classifiers.',
            'regressors': reg_metrics,
            'phase_classifiers': phase_metrics,
            'config': {
                'n_estimators_reg': N_ESTIMATORS_REG,
                'n_estimators_clf': N_ESTIMATORS_CLF,
                'min_samples_leaf': MIN_SAMPLES_LEAF,
                'class_weight': CLASS_WEIGHT,
                'n_folds': N_FOLDS,
                'random_state': RANDOM_STATE,
                'imputation': 'none',
                'phase_features': 'out-of-fold predictions',
                'concentration_fill': 'per-electrolyte median',
                'electrolyte_concentration_medians': elec_medians,
            },
        }, f, indent=2)

    print(f"""
{'=' * 72}
✅  Pipeline C complete → {MODEL_DIR}/

    Remaining step:  cp generator_net_MPEA.pt {MODEL_DIR}/

    Reported R² is what the app will actually deliver: the regressors were
    trained on predicted phase labels, the same kind the app supplies, rather
    than on ground truth the app never has.

    icorr is trained on log10(icorr); app.py back-transforms via 10**pred.
    Phase classifiers take 54 features — app.py build_base_features() matches.
{'=' * 72}
""")
