"""
step2_retrain_models_A.py  —  Pipeline A: Separate Models  (v3)
──────────────────────────────────────────────────────────────
Mechanical models: 58 features (32 elem + 7 proc + 15 emp + 4 phase)
Corrosion models:  66 features (58 + 7 electrolyte one-hot + 1 concentration)

Changes vs v2:
  - Phase (FCC/BCC/HCP/IM) now used as INPUT features (not just outputs)
  - PBS and Hanks electrolytes dropped (n=15 and n=5 — too few to be informative)
  - 7 electrolytes kept: NaCl, H2SO4, Seawater, HNO3, NaOH, HCl, KOH
"""
import os, json
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from joblib import dump

print(f"scikit-learn {sklearn.__version__}  |  Pipeline A v3\n")

DB_PATH   = 'MPEAs_Mech_Corr_DB_updated.xlsx'
MODEL_DIR = 'models_A'
os.makedirs(MODEL_DIR, exist_ok=True)

ELEM_COLS    = ['Ag','Al','B','C','Ca','Co','Cr','Cu','Fe','Ga','Ge','Hf','Li','Mg','Mn','Mo','N','Nb','Nd','Ni','Pd','Re','Sc','Si','Sn','Ta','Ti','V','W','Y','Zn','Zr']
PROCESS_COLS = ['process_1','process_2','process_3','process_4','process_5','process_6','process_7']
EMP_COLS     = ['a','delta','Tm','std of Tm','entropy','enthalpy','std of enthalpy','omega','X','std of X','VEC','std of vec','K','std of K','density']
PHASE_COLS   = ['FCC','BCC','HCP','IM']
ELECTROLYTES = ['NaCl','H2SO4','Seawater','HNO3','NaOH','HCl','KOH']   # 7 — PBS+Hanks dropped
ELEC_COLS    = [f'elec_{e}' for e in ELECTROLYTES]

MECH_FEATURES = ELEM_COLS + PROCESS_COLS + EMP_COLS + PHASE_COLS          # 58
CORR_FEATURES = ELEM_COLS + PROCESS_COLS + EMP_COLS + PHASE_COLS + ELEC_COLS + ['conc_norm']  # 66

df = pd.read_excel(DB_PATH)
for e in ELECTROLYTES:
    df[f'elec_{e}'] = (df['Electrolyte'] == e).astype(float)
df['conc_norm'] = df['Concentration in M'].replace(0, np.nan).fillna(0.5) / 6.0

mech_df = df[df['OG property'] == 'mechanical'].copy()
corr_df = df[(df['OG property'] == 'corrosion') & (df['Electrolyte'].isin(ELECTROLYTES))].copy()
print(f"Mechanical rows: {len(mech_df)},  Corrosion rows (7 electrolytes): {len(corr_df)}\n")
print(f"Feature dims:  Mechanical={len(MECH_FEATURES)},  Corrosion={len(CORR_FEATURES)}")

with open(f'{MODEL_DIR}/feature_config.json','w') as f:
    json.dump({'mech_features': MECH_FEATURES, 'corr_features': CORR_FEATURES,
               'electrolytes': ELECTROLYTES, 'phase_cols': PHASE_COLS}, f)

def train_reg(name, sub, target, feats, log_scale=False, fname=None):
    X = sub[feats].fillna(0).to_numpy(dtype=float)
    y = sub[target].replace(0, np.nan).to_numpy(dtype=float)
    valid = ~np.isnan(y); X, y = X[valid], y[valid]
    if log_scale:
        pos = y > 0; X, y = X[pos], y[pos]; y = np.log10(y)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.1,random_state=49)
    rf = RandomForestRegressor(n_estimators=100,max_depth=50,random_state=0,oob_score=True)
    rf.fit(Xtr,ytr); r2 = r2_score(yte,rf.predict(Xte))
    out = f"{MODEL_DIR}/{fname or name+'.joblib'}"; dump(rf, out)
    print(f"  [{name:15s}] n={len(y):4d}  R²={r2:.4f}  → {out}")

def train_clf(name, sub, target, feats, fname=None):
    X = sub[feats].fillna(0).to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    valid = ~np.isnan(y); X, y = X[valid], y[valid]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.1,random_state=14)
    rf = RandomForestClassifier(n_estimators=100,max_depth=50,random_state=4,oob_score=True)
    rf.fit(Xtr,ytr); acc = accuracy_score(yte,rf.predict(Xte))
    out = f"{MODEL_DIR}/{fname or name+'_classifier.joblib'}"; dump(rf, out)
    print(f"  [{name:15s}] n={len(y):4d}  Acc={acc:.4f}  → {out}")

print("\n--- Mechanical regressors ---")
train_reg('Hardness',       mech_df,'Hardness (HVN)',                   MECH_FEATURES, fname='hardness_regressor.joblib')
train_reg('Yield Strength', mech_df,'Yield Strength (MPa)',             MECH_FEATURES, fname='yield_regressor.joblib')
train_reg('Tensile',        mech_df,'Ultimate Tensile Strength (MPa)',  MECH_FEATURES, fname='tensile_regressor.joblib')
train_reg('Elongation',     mech_df,'Elongation (%)',                   MECH_FEATURES, fname='elongation_regressor.joblib')
print("--- Corrosion regressors ---")
train_reg('Ecorr', corr_df,'Corrosion potential (mV vs SCE)',          CORR_FEATURES, fname='ecorr_regressor.joblib')
train_reg('Epit',  corr_df,'Pitting potential (mV vs SCE)',            CORR_FEATURES, fname='epit_regressor.joblib')
train_reg('icorr', corr_df,'Corrosion current density (microA/cm2)',   CORR_FEATURES, log_scale=True, fname='icorr_regressor.joblib')
print("--- Phase classifiers ---")
for p in PHASE_COLS:
    train_clf(p, df, p, MECH_FEATURES, fname=f'{p}_classifier.joblib')
print(f"\n✅ Pipeline A saved to: {MODEL_DIR}")
