"""
app.py  —  MPEA Mechanical + Corrosion Generative Design Tool
─────────────────────────────────────────────────────────────
NSGAN framework extended to mechanical AND corrosion properties.
Corrosion models include electrolyte type + concentration features
matching Ghorbani et al. (2025) npj Materials Degradation.

═══════════════════════════════════════════════════════════════════════════════
 BUG-FIX REVISION
═══════════════════════════════════════════════════════════════════════════════
 FIX 1 — Phase-classifier target leakage
   The FCC/BCC/HCP/IM classifiers were trained on MECH_FEATURES (58-dim), which
   *includes* the four phase columns — so the FCC classifier had FCC itself as
   an input. Hence the 99.9–100% accuracies. At inference app.py passed zeros
   for those four slots, i.e. a feature pattern never seen in training, which is
   why the classifiers frequently returned 0 for all four phases.
   → Classifiers now consume a 54-dim vector (32 element + 7 processing +
     15 empirical). No phase flags. REQUIRES RETRAINING (step2 + step3).

 FIX 2 — Composition normalisation mismatch
   `_evaluate` fed un-normalised element fractions to the regressors while
   `decode_results` normalised them first, so NSGA-II optimised one
   representation and the results table reported another. Separately, the
   displayed alloy name dropped trace elements (<0.005) and renormalised, so
   even the reported alloy was not the alloy that was predicted.
   → All composition handling now goes through canonicalise_composition(),
     called once inside latent_to_alloys(). Constraints, model features,
     predictions and the displayed name refer to the identical vector.

 FIX 3 — Phase objective vs. displayed phase probability
   `get_obj` used classifiers.predict(mf) (binary, phase slots filled with
   predicted flags) while decode_results used predict_proba(base58) (phase
   slots zeroed) — two different quantities. Binary 0/1 also gives NSGA-II
   almost no signal to optimise against.
   → Both now use predict_proba(base54)[:, 1] from a single featurise() call.

 FIX 3b — post_filter() alloy-name parsing (bonus)
   Used substring matching, so element 'C' matched inside 'Co0.250' and 'N'
   inside 'Nb0.200'; the resulting parse raised ValueError and was silently
   swallowed, making the banned-element safety net ineffective.
   → Now parses with a regex that respects element-symbol boundaries.

 PIPELINE C — wired in
   A third pipeline is now selectable. Its regressors are trained on PREDICTED
   (out-of-fold) phase labels rather than the database's true ones, so training
   conditions match what this app can actually supply at run time, and it uses
   no imputed target values at all. See step4_retrain_models_C.py.
   Any combination of A / B / C can be run and compared side by side.

 R² TABLE — now live
   The performance table used to be hard-coded and went stale every retrain.
   It is now read from each pipeline's metrics.json, alongside phase-classifier
   metrics that don't mislead on rare phases (balanced accuracy, ROC-AUC,
   average precision, and the majority-class baseline).
═══════════════════════════════════════════════════════════════════════════════
"""

import io, os, re, warnings, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from joblib import load
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings('ignore')

st.set_page_config(page_title="MPEA Mech + Corrosion Design",
                   page_icon="⚗️", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
ELEMENTS = ['Ag','Al','B','C','Ca','Co','Cr','Cu','Fe','Ga','Ge','Hf',
            'Li','Mg','Mn','Mo','N','Nb','Nd','Ni','Pd','Re','Sc','Si',
            'Sn','Ta','Ti','V','W','Y','Zn','Zr']
MASSES  = [107.87,26.98,10.81,12.01,40.08,58.93,52.00,63.55,55.85,69.72,
           72.63,178.49,6.94,24.31,54.94,95.96,14.01,92.91,144.24,58.69,
           106.42,186.21,44.96,28.09,118.71,180.95,47.87,50.94,183.84,
           88.91,65.38,91.22]
VOLUMES = [10.27,10.00,4.39,5.29,26.20,6.67,7.23,7.11,7.09,11.80,
           13.63,13.44,13.02,14.00,7.35,9.38,13.54,10.83,20.59,6.59,
           8.56,8.86,15.00,12.06,16.29,10.85,10.64,8.32,9.47,19.88,
           9.16,14.02]
PROCESS_MAP = {
    'process_1': "As-cast / arc-melted",
    'process_2': "Arc-melted + artificial aging",
    'process_3': "Arc-melted + annealing",
    'process_4': "Powder metallurgy",
    'process_5': "Additive / laser / plasma / novel synthesis",
    'process_6': "Arc-melted + wrought processing",
    'process_7': "Cryogenic treatments",
}
ELECTROLYTES = ['NaCl','H2SO4','Seawater','HNO3','NaOH','HCl','KOH']
PHASE_LABELS = ['FCC','BCC','HCP','IM']

# ── Pipelines ─────────────────────────────────────────────────────────────────
PIPELINE_DIRS   = {'A': 'models_A', 'B': 'models_B', 'C': 'models_C'}
PIPELINE_LABELS = {
    'A': 'A — Separate models per subset',
    'B': 'B — Unified models on MissForest-imputed data',
    'C': 'C — Stacked out-of-fold phases, no imputation',
}
# Which feature matrix the mechanical regressors expect.
# A and C: 58-dim (mf). B: 66-dim (cf), because Pipeline B trains every
# regressor on the full imputed 66-column matrix.
MECH_USES_CORR_FEATURES = {'A': False, 'B': True, 'C': False}

# Property keys as written by step2/step3/step4 into metrics.json
METRIC_PROPS = ['Hardness', 'Yield Strength', 'Tensile', 'Elongation',
                'Ecorr', 'Epit', 'icorr']

# FIX 2 — single source of truth for the trace-element cut-off.
# An element below this molar fraction is treated as absent, EVERYWHERE:
# in the constraints, in the model features, and in the displayed alloy name.
TRACE_THRESHOLD = 0.005

OBJECTIVE_INFO = {
    'Tensile Strength' : ('maximize','MPa'),
    'Yield Strength'   : ('maximize','MPa'),
    'Elongation'       : ('maximize','%'),
    'Hardness'         : ('maximize','HV'),
    'Ecorr'            : ('maximize','mV vs SCE'),
    'Epit'             : ('maximize','mV vs SCE'),
    'icorr'            : ('minimize','µA/cm²'),
    'Density'          : ('minimize','g/cm³'),
    'FCC'              : ('maximize','probability'),
    'BCC'              : ('maximize','probability'),
    'HCP'              : ('maximize','probability'),
    'IM'               : ('maximize','probability'),
    'Aluminum Content' : ('maximize','molar ratio'),
}

# Maps objective name -> results dataframe column (for scatter plots)
PROP_KEY = {
    'Tensile Strength' : 'Tensile Strength (MPa)',
    'Yield Strength'   : 'Yield Strength (MPa)',
    'Elongation'       : 'Elongation (%)',
    'Hardness'         : 'Hardness (HV)',
    'Ecorr'            : 'Ecorr (mV vs SCE)',
    'Epit'             : 'Epit (mV vs SCE)',
    'icorr'            : 'icorr (µA/cm²)',
    'Density'          : 'Density (g/cm³)',
    'FCC'              : 'FCC probability',
    'BCC'              : 'BCC probability',
    'HCP'              : 'HCP probability',
    'IM'               : 'IM probability',
    'Aluminum Content' : 'Al molar fraction',
}

# ── Empirical parameter calculation ───────────────────────────────────────────
ATOMIC_RADII   = {'Ag':1.44,'Al':1.43,'B':0.87,'C':0.77,'Ca':1.97,'Co':1.25,'Cr':1.28,'Cu':1.28,'Fe':1.26,'Ga':1.22,'Ge':1.22,'Hf':1.59,'Li':1.52,'Mg':1.60,'Mn':1.26,'Mo':1.36,'N':0.75,'Nb':1.43,'Nd':1.82,'Ni':1.24,'Pd':1.37,'Re':1.37,'Sc':1.62,'Si':1.18,'Sn':1.40,'Ta':1.43,'Ti':1.47,'V':1.34,'W':1.37,'Y':1.80,'Zn':1.33,'Zr':1.60}
MELTING_TEMPS  = {'Ag':1235,'Al':933,'B':2349,'C':3823,'Ca':1115,'Co':1768,'Cr':2180,'Cu':1358,'Fe':1811,'Ga':303,'Ge':1211,'Hf':2506,'Li':454,'Mg':923,'Mn':1519,'Mo':2896,'N':63,'Nb':2750,'Nd':1297,'Ni':1728,'Pd':1828,'Re':3459,'Sc':1814,'Si':1687,'Sn':505,'Ta':3290,'Ti':1941,'V':2183,'W':3695,'Y':1799,'Zn':693,'Zr':2128}
ELECTRONEG_D   = {'Ag':1.93,'Al':1.61,'B':2.04,'C':2.55,'Ca':1.00,'Co':1.88,'Cr':1.66,'Cu':1.90,'Fe':1.83,'Ga':1.81,'Ge':2.01,'Hf':1.30,'Li':0.98,'Mg':1.31,'Mn':1.55,'Mo':2.16,'N':3.04,'Nb':1.60,'Nd':1.14,'Ni':1.91,'Pd':2.20,'Re':1.90,'Sc':1.36,'Si':1.90,'Sn':1.96,'Ta':1.50,'Ti':1.54,'V':1.63,'W':2.36,'Y':1.22,'Zn':1.65,'Zr':1.33}
VEC_D          = {'Ag':11,'Al':3,'B':3,'C':4,'Ca':2,'Co':9,'Cr':6,'Cu':11,'Fe':8,'Ga':3,'Ge':4,'Hf':4,'Li':1,'Mg':2,'Mn':7,'Mo':6,'N':5,'Nb':5,'Nd':4,'Ni':10,'Pd':10,'Re':7,'Sc':3,'Si':4,'Sn':4,'Ta':5,'Ti':4,'V':5,'W':6,'Y':3,'Zn':12,'Zr':4}
MOLAR_MASSES_D = {'Ag':107.87,'Al':26.98,'B':10.81,'C':12.01,'Ca':40.08,'Co':58.93,'Cr':52.00,'Cu':63.55,'Fe':55.85,'Ga':69.72,'Ge':72.63,'Hf':178.49,'Li':6.94,'Mg':24.31,'Mn':54.94,'Mo':95.96,'N':14.01,'Nb':92.91,'Nd':144.24,'Ni':58.69,'Pd':106.42,'Re':186.21,'Sc':44.96,'Si':28.09,'Sn':118.71,'Ta':180.95,'Ti':47.87,'V':50.94,'W':183.84,'Y':88.91,'Zn':65.38,'Zr':91.22}
MOLAR_VOLS_D   = {'Ag':10.27,'Al':10.00,'B':4.39,'C':5.29,'Ca':26.20,'Co':6.67,'Cr':7.23,'Cu':7.11,'Fe':7.09,'Ga':11.80,'Ge':13.63,'Hf':13.44,'Li':13.02,'Mg':14.00,'Mn':7.35,'Mo':9.38,'N':13.54,'Nb':10.83,'Nd':20.59,'Ni':6.59,'Pd':8.56,'Re':8.86,'Sc':15.00,'Si':12.06,'Sn':16.29,'Ta':10.85,'Ti':10.64,'V':8.32,'W':9.47,'Y':19.88,'Zn':9.16,'Zr':14.02}
LATTICE_D      = {'Ag':4.09,'Al':4.05,'B':5.06,'C':3.57,'Ca':5.58,'Co':2.51,'Cr':2.88,'Cu':3.62,'Fe':2.87,'Ga':4.52,'Ge':5.66,'Hf':3.20,'Li':3.51,'Mg':3.21,'Mn':8.91,'Mo':3.15,'N':4.04,'Nb':3.30,'Nd':3.66,'Ni':3.52,'Pd':3.89,'Re':2.76,'Sc':3.31,'Si':5.43,'Sn':5.83,'Ta':3.31,'Ti':2.95,'V':3.02,'W':3.16,'Y':3.65,'Zn':2.66,'Zr':3.23}
BULK_MODULI_D  = {'Ag':100,'Al':76,'B':320,'C':443,'Ca':17,'Co':180,'Cr':160,'Cu':140,'Fe':170,'Ga':59,'Ge':75,'Hf':110,'Li':11,'Mg':45,'Mn':120,'Mo':230,'N':0,'Nb':170,'Nd':32,'Ni':180,'Pd':180,'Re':370,'Sc':57,'Si':98,'Sn':58,'Ta':200,'Ti':110,'V':160,'W':310,'Y':41,'Zn':70,'Zr':94}
ENTHALPY_D     = {('Al','Co'):-19,('Al','Cr'):-10,('Al','Cu'):-1,('Al','Fe'):-11,('Al','Hf'):-45,('Al','Mg'):-2,('Al','Mn'):-19,('Al','Mo'):-22,('Al','Nb'):-18,('Al','Ni'):-22,('Al','Si'):-19,('Al','Ta'):-19,('Al','Ti'):-30,('Al','V'):-16,('Al','W'):-16,('Al','Zr'):-44,('Co','Cr'):-4,('Co','Cu'):6,('Co','Fe'):0,('Co','Mn'):0,('Co','Mo'):-5,('Co','Nb'):-25,('Co','Ni'):0,('Co','Ti'):-28,('Co','V'):-14,('Co','W'):-1,('Co','Zr'):-41,('Cr','Cu'):12,('Cr','Fe'):-1,('Cr','Mn'):2,('Cr','Mo'):0,('Cr','Nb'):-7,('Cr','Ni'):-7,('Cr','Si'):-37,('Cr','Ta'):-7,('Cr','Ti'):-7,('Cr','V'):-2,('Cr','W'):0,('Cr','Zr'):-12,('Cu','Fe'):13,('Cu','Mn'):4,('Cu','Mo'):19,('Cu','Ni'):4,('Cu','Ti'):-9,('Cu','Zr'):-23,('Fe','Mn'):0,('Fe','Mo'):-2,('Fe','Nb'):-16,('Fe','Ni'):-2,('Fe','Si'):-35,('Fe','Ta'):-15,('Fe','Ti'):-17,('Fe','V'):-7,('Fe','W'):-6,('Fe','Zr'):-25,('Mn','Mo'):0,('Mn','Ni'):-8,('Mn','Ti'):-8,('Mn','V'):-1,('Mo','Nb'):-6,('Mo','Ni'):-7,('Mo','Si'):-38,('Mo','Ta'):-5,('Mo','Ti'):-4,('Mo','V'):-5,('Mo','W'):0,('Mo','Zr'):-6,('Nb','Ni'):-30,('Nb','Si'):-56,('Nb','Ta'):0,('Nb','Ti'):-2,('Nb','V'):-2,('Nb','W'):-8,('Nb','Zr'):4,('Ni','Si'):-40,('Ni','Ta'):-24,('Ni','Ti'):-35,('Ni','V'):-18,('Ni','W'):-3,('Ni','Zr'):-49,('Si','Ta'):-45,('Si','Ti'):-66,('Si','V'):-48,('Si','W'):-37,('Si','Zr'):-84,('Ta','Ti'):-4,('Ta','V'):-1,('Ta','W'):-7,('Ti','V'):-2,('Ti','W'):-27,('Ti','Zr'):0,('V','W'):-8,('V','Zr'):-4,('W','Zr'):-27}
R_GAS = 8.314


def calc_empirical_vector(comp32):
    x = {ELEMENTS[i]: comp32[i] for i in range(32) if comp32[i] > 1e-6}
    if not x: return np.zeros(15)
    total = sum(x.values()); x = {e: v/total for e, v in x.items()}; elems = list(x.keys())
    a_mean  = sum(x[e]*LATTICE_D[e]    for e in elems)
    r_mean  = sum(x[e]*ATOMIC_RADII[e] for e in elems)
    delta   = 100*np.sqrt(sum(x[e]*(1-ATOMIC_RADII[e]/r_mean)**2 for e in elems))
    tm_mean = sum(x[e]*MELTING_TEMPS[e]  for e in elems)
    tm_std  = np.sqrt(sum(x[e]*(MELTING_TEMPS[e]-tm_mean)**2 for e in elems))
    entropy = -R_GAS*sum(xi*np.log(xi) for xi in x.values())
    enthalpy= sum(4*ENTHALPY_D.get((e1,e2),ENTHALPY_D.get((e2,e1),0))*x[e1]*x[e2]
                  for i,e1 in enumerate(elems) for e2 in elems[i+1:])
    enth_sq = sum((4*ENTHALPY_D.get((e1,e2),ENTHALPY_D.get((e2,e1),0))*x[e1]*x[e2])**2
                  for i,e1 in enumerate(elems) for e2 in elems[i+1:])
    enth_std= np.sqrt(enth_sq) if enth_sq > 0 else 0.0
    omega   = (tm_mean*entropy/(abs(enthalpy)*1000)) if enthalpy != 0 else 0.0
    xm      = sum(x[e]*ELECTRONEG_D[e] for e in elems)
    xs      = np.sqrt(sum(x[e]*(ELECTRONEG_D[e]-xm)**2 for e in elems))
    vm      = sum(x[e]*VEC_D[e] for e in elems)
    vs      = np.sqrt(sum(x[e]*(VEC_D[e]-vm)**2 for e in elems))
    km      = sum(x[e]*BULK_MODULI_D[e] for e in elems)
    ks      = np.sqrt(sum(x[e]*(BULK_MODULI_D[e]-km)**2 for e in elems))
    mm      = sum(x[e]*MOLAR_MASSES_D[e] for e in elems)
    vol     = sum(x[e]*MOLAR_VOLS_D[e]  for e in elems)
    dens    = mm/vol if vol > 0 else 0.0
    return np.array([a_mean,delta,tm_mean,tm_std,entropy,enthalpy,enth_std,omega,xm,xs,vm,vs,km,ks,dens])


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE BUILDERS
#  ─────────────────────────────────────────────────────────────────────────────
#  54-dim  base       = 32 element + 7 processing + 15 empirical
#                       → PHASE CLASSIFIERS         (FIX 1: no phase flags)
#  58-dim  mechanical = 54 base + 4 predicted phase flags
#                       → mechanical regressors (Pipeline A)
#  66-dim  corrosion  = 58 mech + 7 electrolyte one-hot + 1 concentration
#                       → corrosion regressors (both pipelines) and ALL
#                         regressors in Pipeline B
# ══════════════════════════════════════════════════════════════════════════════

def build_base_features(alloy39):
    """54-dim classifier input. FIX 1 — deliberately contains NO phase flags."""
    return np.concatenate([alloy39[:32], alloy39[32:39],
                           calc_empirical_vector(alloy39[:32])])


def build_mech_features(alloy39, phase4):
    """58-dim = 54 base + 4 phase flags."""
    return np.concatenate([build_base_features(alloy39), phase4])


def build_corr_features(alloy39, phase4, elec_onehot_7, conc_norm):
    """66-dim = 58 mech + 7 electrolyte + 1 concentration."""
    return np.concatenate([build_mech_features(alloy39, phase4),
                           elec_onehot_7, [conc_norm]])


# ══════════════════════════════════════════════════════════════════════════════
#  COMPOSITION CANONICALISATION  (FIX 2)
#  Called exactly once, in latent_to_alloys(). Everything downstream — the
#  NSGA-II constraints, the RF feature vectors, the density, and the alloy name
#  printed in the results table — reads the same canonical vector.
# ══════════════════════════════════════════════════════════════════════════════

def canonicalise_composition(comp32):
    """Normalise to sum 1 → drop trace elements → renormalise to sum 1."""
    c = np.atleast_2d(np.asarray(comp32, dtype=float)).copy()
    c = np.clip(c, 0.0, None)                       # generator can emit tiny negatives
    s = c.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    c /= s
    c[c < TRACE_THRESHOLD] = 0.0
    s = c.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return c / s


def latent_to_alloys(z, generator, comp_min, comp_max):
    """Latent vector(s) → canonical 39-dim design vector(s).

    THE single decode path, shared by _evaluate() and decode_results().
    Previously these two were separate and disagreed (FIX 2).
    """
    z = np.atleast_2d(np.asarray(z, dtype=float))
    with torch.no_grad():
        raw = generator(torch.tensor(z, dtype=torch.float32)).numpy()
    alloys39 = raw.copy()
    alloys39[:, :32] = canonicalise_composition(raw[:, :32] * comp_max + comp_min)
    return alloys39


def _proba_positive(clf, X):
    """P(class == 1). Guards against a classifier that saw only one class."""
    p = clf.predict_proba(X)
    if p.shape[1] == 1:
        return np.full(len(X), float(clf.classes_[0]))
    return p[:, 1]


def featurise(alloys39, classifiers, elec_onehot, conc_norm):
    """One call → every feature matrix and phase quantity anyone needs.

    FIX 3 — phase_proba computed here is used BOTH as the NSGA-II objective and
    as the '<phase> probability' column in the results table, so the two can no
    longer disagree. phase4 (the hard 0/1 flags fed to the regressors as input
    features) is derived from the same probabilities.
    """
    base54 = np.array([build_base_features(a) for a in alloys39])
    phase_proba = np.column_stack(
        [_proba_positive(classifiers[p], base54) for p in PHASE_LABELS])
    # Strictly '>' 0.5, not '>=': sklearn's .predict() is argmax(predict_proba),
    # which breaks a 50/50 tree vote toward class 0. Using '>=' would silently
    # disagree with .predict() on every exact tie.
    phase4 = (phase_proba > 0.5).astype(float)
    mf = np.hstack([base54, phase4])
    n  = len(alloys39)
    cf = np.hstack([mf,
                    np.tile(np.asarray(elec_onehot, dtype=float), (n, 1)),
                    np.full((n, 1), float(conc_norm))])
    return base54, phase_proba, phase4, mf, cf


def alloy_densities(comp32):
    """comp32 must already be canonical (sum = 1)."""
    ma, va = np.array(MASSES), np.array(VOLUMES)
    return (comp32 * ma).sum(1) / np.clip((comp32 * va).sum(1), 1e-9, None)


# ── Generator ─────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 39), nn.ReLU(),
            nn.Linear(39, 39), nn.ReLU(),
            nn.Linear(39, 39), nn.ReLU(),
        )
    def forward(self, z): return self.model(z)


# ── Cached loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(model_dir):
    gen = Generator()
    gen.load_state_dict(torch.load(f"{model_dir}/generator_net_MPEA.pt", map_location="cpu"))
    gen.eval()
    regressors = {
        'Tensile Strength': load(f"{model_dir}/tensile_regressor.joblib"),
        'Elongation':       load(f"{model_dir}/elongation_regressor.joblib"),
        'Yield Strength':   load(f"{model_dir}/yield_regressor.joblib"),
        'Hardness':         load(f"{model_dir}/hardness_regressor.joblib"),
        'Ecorr':            load(f"{model_dir}/ecorr_regressor.joblib"),
        'Epit':             load(f"{model_dir}/epit_regressor.joblib"),
        'icorr':            load(f"{model_dir}/icorr_regressor.joblib"),
    }
    classifiers = {p: load(f"{model_dir}/{p}_classifier.joblib") for p in PHASE_LABELS}

    # FIX 1 guard — refuse to run against stale 58-dim (leaky) classifiers.
    n_in = getattr(classifiers['FCC'], 'n_features_in_', None)
    if n_in is not None and n_in != 54:
        raise RuntimeError(
            f"{model_dir}/FCC_classifier.joblib expects {n_in} features, not 54. "
            "These are the old leakage-affected classifiers — re-run "
            "step2_retrain_models_A.py and step3_retrain_models_B.py.")
    return gen, regressors, classifiers


@st.cache_data
def load_dataset_bounds():
    df = pd.read_excel("MPEAs_Mech_Corr_DB_updated.xlsx")
    ELEM_COLS = ['Ag','Al','B','C','Ca','Co','Cr','Cu','Fe','Ga','Ge','Hf','Li','Mg','Mn','Mo','N','Nb','Nd','Ni','Pd','Re','Sc','Si','Sn','Ta','Ti','V','W','Y','Zn','Zr']
    PROCESS_COLS = ['process_1','process_2','process_3','process_4','process_5','process_6','process_7']
    comp = df[ELEM_COLS].to_numpy(dtype=float)
    return np.min(comp, axis=0), np.max(comp, axis=0), PROCESS_COLS


@st.cache_data
def load_metrics(model_dir):
    """Read metrics.json written by the training scripts. None if absent."""
    path = os.path.join(model_dir, 'metrics.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def available_pipelines():
    """Pipelines whose model directory actually exists on disk."""
    return [p for p in ('A', 'B', 'C') if os.path.isdir(PIPELINE_DIRS[p])]


# ── Optimisation problem ───────────────────────────────────────────────────────
class AlloyProblem(Problem):
    def __init__(self, objectives, generator, regressors, classifiers,
                 comp_min, comp_max, elec_onehot, conc_norm,
                 max_elements=10, banned_indices=None, required_indices=None,
                 pipeline='B'):
        # n_ieq_constr:
        #   1  (max elements)
        #   1  (max single-element fraction ≤ 60%)
        #   +1 per banned element
        #   +1 per required element (ALL must be present)
        n_constr = 2
        if banned_indices:   n_constr += len(banned_indices)
        if required_indices: n_constr += len(required_indices)
        super().__init__(n_var=10, n_obj=len(objectives), n_ieq_constr=n_constr,
                         xl=-3.0, xu=3.0)
        self.objectives       = objectives
        self.generator        = generator
        self.pipeline         = pipeline
        self.regressors       = regressors
        self.classifiers      = classifiers
        self.comp_min         = comp_min
        self.comp_max         = comp_max
        self.elec_onehot      = elec_onehot
        self.conc_norm        = conc_norm
        self.max_elements     = max_elements
        self.banned_indices   = banned_indices   or []
        self.required_indices = required_indices or []

    def _evaluate(self, x, out, *args, **kwargs):
        # FIX 2 — identical decode path as decode_results()
        alloys39 = latent_to_alloys(x, self.generator, self.comp_min, self.comp_max)
        comp32   = alloys39[:, :32]                 # already canonical, sums to 1

        base54, phase_proba, phase4, mf, cf = featurise(
            alloys39, self.classifiers, self.elec_onehot, self.conc_norm)
        densities = alloy_densities(comp32)

        def get_obj(name):
            # A and C: mechanical regressors trained on 58-dim mf
            # B:       every regressor trained on the 66-dim imputed matrix
            mech_feat = cf if MECH_USES_CORR_FEATURES.get(self.pipeline, False) else mf
            if name == 'Tensile Strength': return -self.regressors['Tensile Strength'].predict(mech_feat)
            if name == 'Yield Strength':   return -self.regressors['Yield Strength'].predict(mech_feat)
            if name == 'Elongation':       return -self.regressors['Elongation'].predict(mech_feat)
            if name == 'Hardness':         return -self.regressors['Hardness'].predict(mech_feat)
            if name == 'Ecorr':            return -self.regressors['Ecorr'].predict(cf)
            if name == 'Epit':             return -self.regressors['Epit'].predict(cf)
            if name == 'icorr':            return  self.regressors['icorr'].predict(cf)
            if name == 'Density':          return densities
            if name == 'Aluminum Content': return -comp32[:, 1]
            # FIX 3 — continuous probability, same array reported in the table
            if name in PHASE_LABELS:       return -phase_proba[:, PHASE_LABELS.index(name)]
            return np.zeros(len(alloys39))

        out['F'] = np.column_stack([get_obj(o) for o in self.objectives])

        # ── Constraints (all read the canonical comp32) ────────────────────────
        # Trace elements are already zeroed by canonicalise_composition(), so
        # "present" is exactly "> 0" — no second threshold anywhere.
        EPS = 1e-9

        # G1: number of elements ≤ max_elements
        n_elements = (comp32 > 0).sum(axis=1).astype(float)
        G = [n_elements - self.max_elements]

        # G2: banned elements must be absent
        for idx in self.banned_indices:
            G.append(comp32[:, idx] - EPS)

        # G3: every required element must be present
        for idx in self.required_indices:
            G.append(EPS - comp32[:, idx])

        # G4: no single element > 60% (prevents degenerate single-principal alloys)
        G.append(comp32.max(axis=1) - 0.60)

        out['G'] = np.column_stack(G)


def decode_results(res_X, generator, comp_min, comp_max, regressors,
                   classifiers, proc_names, elec_onehot, conc_norm,
                   pipeline='B'):
    # FIX 2 — same decode + canonicalisation the optimiser used.
    # (np.atleast_2d for the single-solution case is handled inside.)
    alloys39 = latent_to_alloys(res_X, generator, comp_min, comp_max)
    comp32   = alloys39[:, :32]

    base54, phase_proba, phase4, mf, cf = featurise(
        alloys39, classifiers, elec_onehot, conc_norm)

    # Alloy name is now a straight readout of comp32 — no separate
    # thresholding/renormalisation step, because that already happened once
    # in canonicalise_composition(). The alloy shown IS the alloy predicted.
    names, n_elements_list, al_fractions, comp_sums = [], [], [], []
    for comp in comp32:
        active = [(j, comp[j]) for j in range(32) if comp[j] > 0]
        names.append("".join(f"{ELEMENTS[j]}{v:.3f}" for j, v in active))
        n_elements_list.append(len(active))
        al_fractions.append(round(float(comp[1]), 4))
        comp_sums.append(round(float(comp.sum()), 4))    # sanity check — always 1.0

    proc_idx = np.argmax(alloys39[:, 32:39], axis=1)
    procs    = [PROCESS_MAP.get(proc_names[i], "Unknown") for i in proc_idx]

    densities  = alloy_densities(comp32)
    icorr_vals = np.clip(10 ** regressors['icorr'].predict(cf), 0, 1e6)
    mech_feat  = cf if MECH_USES_CORR_FEATURES.get(pipeline, False) else mf

    # Phase label: list every phase predicted present; if none, name the most
    # probable one. Both branches read the same phase_proba used as objective.
    phases = []
    for i in range(len(alloys39)):
        present = [PHASE_LABELS[j] for j in range(4) if phase4[i, j] > 0]
        if present:
            phases.append("+".join(present))
        else:
            phases.append(f"{PHASE_LABELS[int(np.argmax(phase_proba[i]))]} (dominant)")

    return pd.DataFrame({
        'Alloy Composition':      names,
        'N Elements':             n_elements_list,
        'Processing Method':      procs,
        'Predicted Phase':        phases,
        'Hardness (HV)':          np.round(regressors['Hardness'].predict(mech_feat),        2),
        'Tensile Strength (MPa)': np.round(regressors['Tensile Strength'].predict(mech_feat),2),
        'Yield Strength (MPa)':   np.round(regressors['Yield Strength'].predict(mech_feat),  2),
        'Elongation (%)':         np.round(regressors['Elongation'].predict(mech_feat),      2),
        'Ecorr (mV vs SCE)':      np.round(regressors['Ecorr'].predict(cf),                  2),
        'Epit (mV vs SCE)':       np.round(regressors['Epit'].predict(cf),                   2),
        'icorr (µA/cm²)':         np.round(icorr_vals,                                       4),
        'Density (g/cm³)':        np.round(densities,                                        3),
        'FCC probability':        np.round(phase_proba[:, 0],                                3),
        'BCC probability':        np.round(phase_proba[:, 1],                                3),
        'HCP probability':        np.round(phase_proba[:, 2],                                3),
        'IM probability':         np.round(phase_proba[:, 3],                                3),
        'Al molar fraction':      al_fractions,
        'Composition sum':        comp_sums,
    })


def run_optimisation(objectives, pop_size, n_gen, seed, generator,
                     regressors, classifiers, comp_min, comp_max,
                     proc_names, elec_onehot, conc_norm,
                     max_elements=10, banned_indices=None, required_indices=None,
                     pipeline='B'):
    problem   = AlloyProblem(objectives, generator, regressors, classifiers,
                             comp_min, comp_max, elec_onehot, conc_norm,
                             max_elements, banned_indices, required_indices,
                             pipeline=pipeline)
    algorithm = NSGA2(pop_size=pop_size, mutation=PM(prob=0.1, eta=20))
    res = minimize(problem, algorithm, get_termination("n_gen", n_gen),
                   save_history=False, seed=int(seed), verbose=False)

    if res.X is None:          # no feasible solutions
        return None

    return decode_results(res.X, generator, comp_min, comp_max,
                          regressors, classifiers, proc_names,
                          elec_onehot, conc_norm, pipeline=pipeline)


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("⚗️ MPEA Mechanical + Corrosion Generative Design")
st.markdown("""
Generates novel MPEAs optimised simultaneously for **mechanical** and **corrosion** properties
using the NSGAN framework. Corrosion models include **electrolyte type + concentration** as features,
matching Ghorbani et al. (2025) *npj Materials Degradation*.
""")

comp_min, comp_max, proc_names = load_dataset_bounds()

with st.sidebar:
    st.header("⚙️ Settings")

    _avail = available_pipelines() or ['C']
    _default = ['C'] if 'C' in _avail else _avail[:1]
    selected_pipelines = st.multiselect(
        "Pipeline(s)", _avail, default=_default,
        format_func=lambda p: PIPELINE_LABELS[p],
        help="Select one to run it, or several to compare side by side.")
    if not selected_pipelines:
        st.warning("Select at least one pipeline.")

    st.divider()
    st.subheader("🌊 Test Environment")
    selected_electrolyte = st.selectbox("Electrolyte", ELECTROLYTES, index=0,
        help="Electrolyte used in corrosion testing — included as a model feature")
    selected_conc = st.number_input("Concentration (M)", min_value=0.05,
        max_value=6.0, value=0.6, step=0.05,
        help="Electrolyte molar concentration (0.05–6 M)")

    elec_onehot = np.array([1.0 if e == selected_electrolyte else 0.0 for e in ELECTROLYTES])
    conc_norm   = selected_conc / 6.0

    st.divider()
    st.subheader("🎯 Objectives")
    selected_objectives = st.multiselect("Optimisation Objectives",
        list(OBJECTIVE_INFO.keys()),
        default=["Tensile Strength", "Elongation", "icorr"])

    if selected_objectives:
        st.dataframe(pd.DataFrame([
            {'Objective': o,
             'Direction': '↑ Max' if OBJECTIVE_INFO[o][0]=='maximize' else '↓ Min',
             'Unit': OBJECTIVE_INFO[o][1]}
            for o in selected_objectives
        ]), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("🧪 Alloy Constraints")
    max_elements = st.slider("Max number of elements", min_value=2, max_value=10, value=7,
        help="Enforced as an NSGA-II inequality constraint — all returned alloys satisfy this. "
             f"An element counts as present above {TRACE_THRESHOLD:.3f} molar fraction. "
             "Training data: 2–10 elements (mean = 5.3). Most reliable range: 4–7.")

    allowed_elements = st.multiselect(
        "Allowed elements (pool)",
        options=ELEMENTS,
        default=ELEMENTS,
        help="Only elements selected here can appear in optimised alloys. "
             "Elements not selected are banned as an NSGA-II constraint.")

    required_elements = st.multiselect(
        "Required elements",
        options=allowed_elements if allowed_elements else ELEMENTS,
        default=[],
        help=f"Optional. Every element selected here MUST appear (> {TRACE_THRESHOLD:.3f} mol "
             "fraction) in every optimised alloy — e.g. select Al and Fe to guarantee both are "
             "present. Leave empty to impose no requirement.")

    banned_indices   = [ELEMENTS.index(e) for e in ELEMENTS if e not in allowed_elements]
    required_indices = [ELEMENTS.index(e) for e in required_elements]

    if len(allowed_elements) == 0:
        st.error("⚠️ No elements allowed — add at least one element to the pool.")
    elif len(allowed_elements) < 2:
        st.warning("⚠️ Only 1 element allowed — most alloy models require at least 2 elements.")
    else:
        if banned_indices:
            st.caption(f"🚫 {len(banned_indices)} banned: {', '.join(e for e in ELEMENTS if e not in allowed_elements)}")
        if required_indices:
            st.caption(f"✅ ALL required (each must be present): {', '.join(required_elements)}")
        if not banned_indices and not required_indices:
            st.caption("No element constraints — all 32 elements allowed freely.")

    st.divider()
    pop_size = st.slider("Population Size", 10, 200, 50, 10)
    n_gen    = st.slider("Generations",     10, 500, 200, 10)
    seed_val = st.number_input("Random Seed", 0, 9999, 2)

    run_btn = st.button("🚀 Start Optimisation", type="primary",
                        use_container_width=True,
                        disabled=len(selected_objectives) < 2 or not selected_pipelines)
    if len(selected_objectives) < 2:
        st.warning("Select at least 2 objectives.")

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL PERFORMANCE — read live from each pipeline's metrics.json
#  Previously hard-coded, which meant the table silently went stale every time
#  the models were retrained. Now whatever the training script measured is
#  what the app displays.
# ══════════════════════════════════════════════════════════════════════════════
PIPELINE_COLOURS = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}

LEGACY_R2 = {                      # fallback if a metrics.json is missing
    'A': {'Hardness': 0.832, 'Yield Strength': 0.638, 'Tensile': 0.666,
          'Elongation': 0.440, 'Ecorr': 0.646, 'Epit': 0.761, 'icorr': 0.459},
    'B': {'Hardness': 0.832, 'Yield Strength': 0.639, 'Tensile': 0.668,
          'Elongation': 0.441, 'Ecorr': 0.629, 'Epit': 0.731, 'icorr': 0.393},
}

PROP_FEATURES = {
    'Hardness':       '58 = 32 element + 7 processing + 15 empirical + 4 phase',
    'Yield Strength': '58 = 32 element + 7 processing + 15 empirical + 4 phase',
    'Tensile':        '58 = 32 element + 7 processing + 15 empirical + 4 phase',
    'Elongation':     '58 = 32 element + 7 processing + 15 empirical + 4 phase',
    'Ecorr':          '66 = 58 + 7 electrolyte + 1 concentration',
    'Epit':           '66 = 58 + 7 electrolyte + 1 concentration',
    'icorr':          '66 = 58 + 7 electrolyte + 1 concentration',
}


def build_r2_table(pipes):
    rows = []
    for prop in METRIC_PROPS:
        row = {'Property': prop + (' (log₁₀)' if prop == 'icorr' else '')}
        n_seen = None
        for p in pipes:
            m = load_metrics(PIPELINE_DIRS[p])
            entry = (m or {}).get('regressors', {}).get(prop)
            if entry:
                row[f'{p} R²'] = round(entry['cv_r2_mean'], 3)
                n_seen = n_seen or entry.get('n')
            else:
                legacy = LEGACY_R2.get(p, {}).get(prop)
                row[f'{p} R²'] = legacy if legacy is not None else float('nan')
        row['n (real labels)'] = n_seen if n_seen else '—'
        row['Features'] = PROP_FEATURES[prop]
        rows.append(row)
    return pd.DataFrame(rows)


def phase_verdict(entry):
    """Judge a phase classifier on DISCRIMINATION, not accuracy.

    For a class as rare as HCP (~2% positive), accuracy is pinned to the
    majority baseline no matter how good the model is: predicting "no" almost
    always is the accuracy-optimal strategy. ROC-AUC and average precision are
    threshold-free and unaffected by the imbalance, so they decide the verdict.
    """
    auc = entry.get('roc_auc')
    if auc is None or auc != auc:            # None or NaN
        return '—'
    if auc >= 0.85: return '✅ strong'
    if auc >= 0.70: return '✅ useful'
    if auc >= 0.60: return '⚠️ weak'
    return '⚠️ no skill'


def build_phase_table(pipes):
    rows = []
    for p in pipes:
        m = load_metrics(PIPELINE_DIRS[p])
        pm = (m or {}).get('phase_classifiers')
        if not pm:
            continue
        for phase in PHASE_LABELS:
            e = pm.get(phase)
            if not e:
                continue
            rows.append({
                'Pipeline': p, 'Phase': phase,
                'Positives': f"{e.get('positive_rate', float('nan')):.1%}",
                'Accuracy': round(e.get('accuracy', float('nan')), 3),
                'Majority baseline': round(e.get('majority_baseline', float('nan')), 3),
                'Balanced acc.': round(e.get('balanced_accuracy', float('nan')), 3),
                'ROC-AUC': round(e.get('roc_auc', float('nan')), 3),
                'Avg precision': round(e.get('average_precision', float('nan')), 3),
                'AP vs chance': (
                    f"{e['average_precision'] / e['positive_rate']:.0f}x"
                    if e.get('positive_rate') else '—'),
                'Verdict': phase_verdict(e),
            })
    return pd.DataFrame(rows)


_shown = selected_pipelines or available_pipelines() or ['C']
r2_df    = build_r2_table(_shown)
phase_df = build_phase_table(_shown)

with st.expander("📊 Model performance summary", expanded=False):
    st.markdown("**Regressors — cross-validated R²**")
    st.dataframe(r2_df, hide_index=True, use_container_width=True)
    st.caption(
        "Read live from each pipeline's `metrics.json`, so these numbers always match "
        "the models currently on disk. Processing and electrolyte features are one-hot "
        "encoded; PBS and Hanks are excluded (n < 15). icorr is trained on log₁₀ and "
        "back-transformed for display.")
    if 'C' in _shown:
        st.caption(
            "**Why Pipeline C's numbers may look slightly lower than A's:** C trains its "
            "regressors on *predicted* phase labels — the same imperfect labels the app "
            "supplies when you optimise. A trains on the database's true phase labels, "
            "which the app never has. C's R² is therefore the one that reflects what you "
            "actually get on a new alloy.")

    if len(phase_df):
        st.markdown("**Phase classifiers — why accuracy alone misleads**")
        st.dataframe(phase_df, hide_index=True, use_container_width=True)
        st.caption(
            "**Accuracy is the wrong score for a rare phase, and is shown only for "
            "reference.** HCP appears in under 2% of alloys, so \"never predict HCP\" "
            "already scores ~0.98 — accuracy is pinned near the baseline however good "
            "the model is. The Verdict column therefore comes from ROC-AUC, not accuracy.")
        st.caption(
            "**ROC-AUC** — shown one alloy with the phase and one without, how often is the "
            "right one ranked higher? 0.50 = coin flip, 0.90 = strong. "
            "**AP vs chance** — how many times better than random guessing the model is at "
            "surfacing that phase; anything above ~5x is a genuinely informative model. "
            "**Balanced accuracy** — average of the hit rate on each class at the default "
            "0.5 threshold; 0.50 = no skill. A rare-class model often has excellent ROC-AUC "
            "but moderate balanced accuracy, because 0.5 is a conservative cut-off for a "
            "class that rare. That combination is fine: the optimiser uses the continuous "
            "probability, i.e. the ranking, not the 0/1 decision.")
        st.caption(
            "⚠️ Classifiers use `class_weight='balanced_subsample'`, which is what gives "
            "the rare phases usable spread for optimisation. The trade-off: the reported "
            "probabilities are **not calibrated** to the true base rate — treat them as a "
            "relative ranking, not as literal likelihoods.")
    st.caption(
        "All phase classifiers take 54 features (element + processing + empirical). "
        "The phase columns are deliberately excluded from their own inputs, so these are "
        "genuine held-out scores — not the ~100% seen when a phase flag predicted itself.")

# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════
ALLOY_TOKEN = re.compile(r'([A-Z][a-z]?)(\d+\.\d+)')

if run_btn and len(selected_objectives) >= 2 and selected_pipelines:
    if len(allowed_elements) == 0:
        st.error("No elements allowed — please keep at least one element in the allowed pool.")
        st.stop()

    def post_filter(df):
        """Safety net: drop alloys that still contain a banned element.
        pymoo can return least-infeasible candidates when nothing is feasible.

        FIX 3b — parses the alloy name with a regex on element-symbol boundaries.
        The previous `if el in name` test matched 'C' inside 'Co0.250' and 'N'
        inside 'Nb0.200', then failed to parse and silently kept the row.
        """
        if df is None or not banned_indices:
            return df
        banned_syms = {ELEMENTS[i] for i in banned_indices}
        keep = []
        for name in df['Alloy Composition']:
            comp = {sym: float(frac) for sym, frac in ALLOY_TOKEN.findall(name)}
            keep.append(not any(comp.get(s, 0.0) > 0 for s in banned_syms))
        filtered = df[keep].reset_index(drop=True)
        return filtered if len(filtered) else None

    progress = st.progress(0, "Starting optimisation…")
    results = {}

    for i, p in enumerate(selected_pipelines):
        pct = int(5 + 90 * i / max(1, len(selected_pipelines)))
        progress.progress(pct, f"Pipeline {p} — NSGA-II ({n_gen} generations)…")
        try:
            gen, reg, clf = load_pipeline(PIPELINE_DIRS[p])
            res = run_optimisation(selected_objectives, pop_size, n_gen, seed_val,
                                   gen, reg, clf, comp_min, comp_max, proc_names,
                                   elec_onehot, conc_norm, max_elements,
                                   banned_indices, required_indices, pipeline=p)
            res = post_filter(res)
            if res is None:
                st.warning(f"Pipeline {p}: no feasible solutions found. Try relaxing the "
                           "element constraints, raising max elements, or running more generations.")
            else:
                results[p] = res
        except Exception as e:
            st.error(f"Pipeline {p} failed: {e}")

    progress.progress(100, "Done!")
    progress.empty()
    st.session_state.update({'results': results,
                             'objectives': selected_objectives,
                             'electrolyte': selected_electrolyte, 'conc': selected_conc,
                             'max_elements': max_elements,
                             'allowed_elements': allowed_elements,
                             'required_elements': required_elements})

# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get('results'):
    results    = st.session_state['results']
    objectives = st.session_state.get('objectives', [])
    max_el     = st.session_state.get('max_elements', 10)
    order      = [p for p in ('A', 'B', 'C') if p in results]

    st.divider()
    allowed_el  = st.session_state.get('allowed_elements', ELEMENTS)
    required_el = st.session_state.get('required_elements', [])
    banned_el   = [e for e in ELEMENTS if e not in allowed_el]
    info_parts  = [f"🌊 **{st.session_state.get('electrolyte','')}** at {st.session_state.get('conc','')} M",
                   f"max **{max_el}** elements",
                   "max **60%** per element"]
    if banned_el:   info_parts.append(f"🚫 banned: {', '.join(banned_el)}")
    if required_el: info_parts.append(f"✅ required: {', '.join(required_el)}")
    st.info("  ·  ".join(info_parts))

    # ── Scatter plots ──────────────────────────────────────────────────────────
    st.subheader("📈 Pareto Fronts")

    def get_pairs(objectives, df):
        """Return (xcol, ycol, title) pairs for objectives that exist in df."""
        valid   = [o for o in objectives if PROP_KEY.get(o) in df.columns]
        mech_o  = [o for o in valid if o in ('Tensile Strength','Yield Strength','Elongation','Hardness')]
        corr_o  = [o for o in valid if o in ('Ecorr','Epit','icorr')]
        other_o = [o for o in valid if o not in mech_o and o not in corr_o]
        pairs = []
        if len(mech_o) >= 2: pairs.append((mech_o[0], mech_o[1], "Mechanical"))
        if mech_o and corr_o: pairs.append((mech_o[0], corr_o[0], "Mech vs Corrosion"))
        if len(corr_o) >= 2: pairs.append((corr_o[0], corr_o[1], "Corrosion"))
        if other_o and valid:
            base = mech_o[0] if mech_o else corr_o[0] if corr_o else valid[0]
            for o in other_o:
                if o != base: pairs.append((base, o, f"{base} vs {o}"))
        if not pairs and len(valid) >= 2:
            pairs.append((valid[0], valid[1], "Objectives"))
        return pairs

    pairs = get_pairs(objectives, results[order[0]])

    if not pairs:
        st.info("No plottable objective pairs found — select at least 2 objectives with matching result columns.")
    else:
        n_p = len(order)
        for x_obj, y_obj, title in pairs:
            xk, yk = PROP_KEY[x_obj], PROP_KEY[y_obj]
            fig, axes = plt.subplots(1, n_p, figsize=(12, 4),
                                     sharey=(n_p > 1), squeeze=False)
            for ax, p in zip(axes[0], order):
                res = results[p]
                if xk in res.columns and yk in res.columns:
                    ax.scatter(res[xk], res[yk], c=PIPELINE_COLOURS[p],
                               alpha=0.7, edgecolors='white', s=60)
                ax.set_xlabel(xk); ax.set_ylabel(yk)
                ax.set_title(f"{title} — Pipeline {p}"); ax.grid(True, ls='--', alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

    # ── Results tables ─────────────────────────────────────────────────────────
    st.divider()

    HIDE_COLS = ['N Elements','FCC probability','BCC probability','HCP probability',
                 'IM probability','Al molar fraction','Composition sum']

    def display_df(df):
        """Drop internal columns from display; keep them in the download."""
        return df.drop(columns=[c for c in HIDE_COLS if c in df.columns]).reset_index(drop=True)

    if len(order) > 1:
        for tab, p in zip(st.tabs([f"Pipeline {p}" for p in order]), order):
            with tab:
                st.caption(f"{len(results[p])} alloys · all satisfy ≤ {max_el} elements · "
                           f"{PIPELINE_LABELS[p]}")
                st.dataframe(display_df(results[p]), use_container_width=True)
    else:
        p = order[0]
        st.subheader(f"Pipeline {p} — {len(results[p])} alloys")
        st.caption(f"All alloys satisfy ≤ {max_el} elements constraint · {PIPELINE_LABELS[p]}")
        st.dataframe(display_df(results[p]), use_container_width=True)

    st.caption("ℹ️ Training database: 2–10 elements (mean 5.3). "
               "Predictions most reliable in the 4–7 element range. "
               "Listed compositions are exactly the compositions passed to the models.")

    # ── Download (includes all columns) ───────────────────────────────────────
    st.divider()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        for p in order:
            results[p].to_excel(w, sheet_name=f'Pipeline_{p}', index=False)
        r2_df.to_excel(w, sheet_name='Model_R2', index=False)
        if len(phase_df):
            phase_df.to_excel(w, sheet_name='Phase_metrics', index=False)
    buf.seek(0)
    st.download_button("⬇️ Download Excel (all results)", data=buf,
                       file_name="MPEA_mech_corr_optimised.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
