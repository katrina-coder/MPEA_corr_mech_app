"""
app.py  —  MPEA Mechanical + Corrosion Generative Design Tool
─────────────────────────────────────────────────────────────
NSGAN framework extended to mechanical AND corrosion properties.
Corrosion models include electrolyte type + concentration features
matching Ghorbani et al. (2025) npj Materials Degradation.
"""

import io, os, warnings, json
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
    'process_5': "Novel synthesis (ball milling etc.)",
    'process_6': "Arc-melted + wrought processing",
    'process_7': "Cryogenic treatments",
}
ELECTROLYTES   = ['NaCl','H2SO4','Seawater','HNO3','NaOH','HCl','KOH']
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

def build_mech_features(alloy39, phase4):
    """58-dim feature vector for mechanical models = 54 + 4 phase."""
    return np.concatenate([alloy39[:32], alloy39[32:], calc_empirical_vector(alloy39[:32]), phase4])

def build_corr_features(alloy39, phase4, elec_onehot_7, conc_norm):
    """66-dim feature vector for corrosion models = 58 + 7 elec + 1 conc."""
    mf = build_mech_features(alloy39, phase4)
    return np.concatenate([mf, elec_onehot_7, [conc_norm]])

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
    classifiers = {p: load(f"{model_dir}/{p}_classifier.joblib") for p in ['FCC','BCC','HCP','IM']}
    return gen, regressors, classifiers

@st.cache_data
def load_dataset_bounds():
    df = pd.read_excel("MPEAs_Mech_Corr_DB_updated.xlsx")
    ELEM_COLS    = ['Ag','Al','B','C','Ca','Co','Cr','Cu','Fe','Ga','Ge','Hf','Li','Mg','Mn','Mo','N','Nb','Nd','Ni','Pd','Re','Sc','Si','Sn','Ta','Ti','V','W','Y','Zn','Zr']
    PROCESS_COLS = ['process_1','process_2','process_3','process_4','process_5','process_6','process_7']
    comp = df[ELEM_COLS].to_numpy(dtype=float)
    return np.min(comp, axis=0), np.max(comp, axis=0), PROCESS_COLS

# ── Optimisation problem ───────────────────────────────────────────────────────
class AlloyProblem(Problem):
    def __init__(self, objectives, generator, regressors, classifiers,
                 comp_min, comp_max, elec_onehot, conc_norm, max_elements=10):
        super().__init__(n_var=10, n_obj=len(objectives), n_ieq_constr=1, xl=-3.0, xu=3.0)
        self.objectives   = objectives
        self.generator    = generator
        self.regressors   = regressors
        self.classifiers  = classifiers
        self.comp_min     = comp_min
        self.comp_max     = comp_max
        self.elec_onehot  = elec_onehot
        self.conc_norm    = conc_norm
        self.max_elements = max_elements

    def _evaluate(self, x, out, *args, **kwargs):
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            raw = self.generator(x_t).numpy()
        # Generator outputs 39-dim; comp_min/max cover only the 32 element columns
        alloys39 = raw.copy()
        alloys39[:, :32] = raw[:, :32] * self.comp_max + self.comp_min

        # Build 58-dim vector with zero phase placeholders, predict phases, then use properly
        zeros4 = np.zeros(4)
        base58 = np.array([np.concatenate([a[:32], a[32:], calc_empirical_vector(a[:32]), zeros4]) for a in alloys39])
        phase4 = np.column_stack([self.classifiers[p].predict(base58).astype(float) for p in ['FCC','BCC','HCP','IM']])
        mf = np.array([build_mech_features(alloys39[i], phase4[i]) for i in range(len(alloys39))])
        cf = np.array([build_corr_features(alloys39[i], phase4[i], self.elec_onehot, self.conc_norm) for i in range(len(alloys39))])

        masses_a = np.array(MASSES); volumes_a = np.array(VOLUMES)
        comp32_n = alloys39[:, :32].copy()
        rs = comp32_n.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        comp32_n = comp32_n / rs
        densities = (comp32_n * masses_a).sum(1) / (comp32_n * volumes_a).sum(1)

        def get_obj(name):
            if name == 'Tensile Strength': return -self.regressors['Tensile Strength'].predict(mf)
            if name == 'Yield Strength':   return -self.regressors['Yield Strength'].predict(mf)
            if name == 'Elongation':       return -self.regressors['Elongation'].predict(mf)
            if name == 'Hardness':         return -self.regressors['Hardness'].predict(mf)
            if name == 'Ecorr':            return -self.regressors['Ecorr'].predict(cf)
            if name == 'Epit':             return -self.regressors['Epit'].predict(cf)
            if name == 'icorr':            return  self.regressors['icorr'].predict(cf)
            if name == 'Density':          return densities
            if name == 'Aluminum Content': return -alloys39[:, 1]
            if name in ('FCC','BCC','HCP','IM'):
                return -self.classifiers[name].predict(mf).astype(float)
            return np.zeros(len(alloys39))

        out['F'] = np.column_stack([get_obj(o) for o in self.objectives])

        # Inequality constraint: n_elements - max_elements <= 0  (G <= 0 is feasible in pymoo)
        n_elements = (alloys39[:, :32] > 0.005).sum(axis=1).astype(float)
        out['G'] = (n_elements - self.max_elements).reshape(-1, 1)


def decode_results(res_X, generator, comp_min, comp_max, regressors,
                   classifiers, proc_names, elec_onehot, conc_norm):
    # FIX BUG 4: ensure res_X is always 2D for batch generator input
    res_X = np.atleast_2d(res_X)

    x_t = torch.tensor(res_X, dtype=torch.float32)
    with torch.no_grad():
        raw = generator(x_t).numpy()
    alloys39 = raw.copy()
    alloys39[:, :32] = raw[:, :32] * comp_max + comp_min
    rs = alloys39[:, :32].sum(1, keepdims=True); rs[rs == 0] = 1
    alloys39[:, :32] /= rs

    zeros4 = np.zeros(4)
    base58 = np.array([np.concatenate([a[:32], a[32:], calc_empirical_vector(a[:32]), zeros4]) for a in alloys39])
    phase4 = np.column_stack([classifiers[p].predict(base58).astype(float) for p in ['FCC','BCC','HCP','IM']])
    mf = np.array([build_mech_features(alloys39[i], phase4[i]) for i in range(len(alloys39))])
    cf = np.array([build_corr_features(alloys39[i], phase4[i], elec_onehot, conc_norm) for i in range(len(alloys39))])

    names = []
    n_elements_list = []
    al_fractions = []
    for comp in alloys39[:, :32]:
        parts = [f"{ELEMENTS[j]}{comp[j]:.3f}" for j in range(32) if comp[j] > 0.005]
        names.append("".join(parts))
        n_elements_list.append(len(parts))
        al_fractions.append(round(comp[1], 4))   # Al is index 1

    proc_idx = np.argmax(alloys39[:, 32:], axis=1)
    procs = [PROCESS_MAP.get(proc_names[i], "Unknown") for i in proc_idx]

    ma  = np.array(MASSES); va = np.array(VOLUMES)
    c32 = alloys39[:, :32].copy()
    c32 /= c32.sum(1, keepdims=True).clip(1e-9)
    densities = (c32 * ma).sum(1) / (c32 * va).sum(1)

    icorr_vals = np.clip(10 ** regressors['icorr'].predict(cf), 0, 1e6)
    phase_lbls = ['FCC','BCC','HCP','IM']

    # Get probabilities for display and scatter (predict_proba[:,1] = probability of presence)
    phase_proba = np.column_stack([classifiers[p].predict_proba(base58)[:, 1] for p in phase_lbls])
    fcc_prob = phase_proba[:, 0]; bcc_prob = phase_proba[:, 1]
    hcp_prob = phase_proba[:, 2]; im_prob  = phase_proba[:, 3]

    # Phase label: list all phases predicted present (binary); if none, use highest-probability phase
    phases = []
    for i in range(len(alloys39)):
        present = [phase_lbls[j] for j in range(4) if phase4[i, j] > 0]
        if present:
            phases.append("+".join(present))
        else:
            # Fallback: pick the phase with highest predicted probability
            best = int(np.argmax(phase_proba[i]))
            phases.append(f"{phase_lbls[best]} (dominant)")

    return pd.DataFrame({
        'Alloy Composition':      names,
        'N Elements':             n_elements_list,
        'Processing Method':      procs,
        'Predicted Phase':        phases,
        'Hardness (HV)':          np.round(regressors['Hardness'].predict(mf),        2),
        'Tensile Strength (MPa)': np.round(regressors['Tensile Strength'].predict(mf),2),
        'Yield Strength (MPa)':   np.round(regressors['Yield Strength'].predict(mf),  2),
        'Elongation (%)':         np.round(regressors['Elongation'].predict(mf),      2),
        'Ecorr (mV vs SCE)':      np.round(regressors['Ecorr'].predict(cf),           2),
        'Epit (mV vs SCE)':       np.round(regressors['Epit'].predict(cf),            2),
        'icorr (µA/cm²)':         np.round(icorr_vals,                                4),
        'Density (g/cm³)':        np.round(densities,                                 3),
        'FCC probability':        np.round(fcc_prob,                                  3),
        'BCC probability':        np.round(bcc_prob,                                  3),
        'HCP probability':        np.round(hcp_prob,                                  3),
        'IM probability':         np.round(im_prob,                                   3),
        'Al molar fraction':      al_fractions,
    })


def run_optimisation(objectives, pop_size, n_gen, seed, generator,
                     regressors, classifiers, comp_min, comp_max,
                     proc_names, elec_onehot, conc_norm, max_elements=10):
    problem   = AlloyProblem(objectives, generator, regressors, classifiers,
                              comp_min, comp_max, elec_onehot, conc_norm, max_elements)
    algorithm = NSGA2(pop_size=pop_size, mutation=PM(prob=0.1, eta=20))
    res = minimize(problem, algorithm, get_termination("n_gen", n_gen),
                   save_history=False, seed=int(seed), verbose=False)

    # FIX BUG 1: res.X is None when no feasible solutions found
    if res.X is None:
        return None

    return decode_results(res.X, generator, comp_min, comp_max,
                          regressors, classifiers, proc_names, elec_onehot, conc_norm)


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

    pipeline_choice = st.radio("Pipeline",
        ["A — Separate models", "B — Imputed unified models", "A vs B — Compare both"],
        index=0)

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
             "Training data: 2–10 elements (mean = 5.3). Most reliable range: 4–7.")

    st.divider()
    pop_size = st.slider("Population Size", 10, 200, 50, 10)
    n_gen    = st.slider("Generations",     10, 500, 200, 10)
    seed_val = st.number_input("Random Seed", 0, 9999, 2)

    run_btn = st.button("🚀 Start Optimisation", type="primary",
                        use_container_width=True,
                        disabled=len(selected_objectives) < 2)
    if len(selected_objectives) < 2:
        st.warning("Select at least 2 objectives.")

# ── Model performance table ────────────────────────────────────────────────────
with st.expander("📊 Model R² performance summary", expanded=False):
    r2_data = {
        'Property':             ['Hardness','Yield Strength','Tensile','Elongation','Ecorr','Epit','icorr (log₁₀)'],
        'Pipeline A R²':        [0.802, 0.574, 0.662, 0.525, 0.646, 0.775, 0.451],
        'Pipeline B R²':        [0.921, 0.756, 0.708, 0.617, 0.742, 0.856, 0.598],
        'Features (corrosion)': ['—','—','—','—','58+7elec','58+7elec','58+7elec'],
        'Note':                 ['','','','','Electrolyte type matters','Electrolyte type matters','Inherently noisy'],
    }
    st.dataframe(pd.DataFrame(r2_data), hide_index=True, use_container_width=True)
    st.caption("Corrosion models: 58 features (32 elem + 7 proc + 15 empirical + 4 phase) "
               "+ 7 electrolyte one-hot + concentration. PBS and Hanks excluded (n<15). "
               "Matches Ghorbani et al. (2025) npj Materials Degradation.")

# ── Run ────────────────────────────────────────────────────────────────────────
if run_btn and len(selected_objectives) >= 2:
    run_A = "A" in pipeline_choice
    run_B = "B" in pipeline_choice
    progress = st.progress(0, "Starting optimisation…")
    result_A = result_B = None

    # FIX BUG 6: wrap each pipeline in try/except for user-friendly error messages
    if run_A:
        try:
            gen_A, reg_A, clf_A = load_pipeline("models_A")
            progress.progress(10, f"Pipeline A — NSGA-II ({n_gen} generations)…")
            result_A = run_optimisation(selected_objectives, pop_size, n_gen, seed_val,
                                        gen_A, reg_A, clf_A, comp_min, comp_max, proc_names,
                                        elec_onehot, conc_norm, max_elements)
            if result_A is None:
                st.warning("Pipeline A: No feasible solutions found. Try increasing max elements or generations.")
        except Exception as e:
            st.error(f"Pipeline A failed: {e}")

    if run_B:
        try:
            gen_B, reg_B, clf_B = load_pipeline("models_B")
            progress.progress(55 if run_A else 10, f"Pipeline B — NSGA-II ({n_gen} generations)…")
            result_B = run_optimisation(selected_objectives, pop_size, n_gen, seed_val,
                                        gen_B, reg_B, clf_B, comp_min, comp_max, proc_names,
                                        elec_onehot, conc_norm, max_elements)
            if result_B is None:
                st.warning("Pipeline B: No feasible solutions found. Try increasing max elements or generations.")
        except Exception as e:
            st.error(f"Pipeline B failed: {e}")

    progress.progress(100, "Done!")
    progress.empty()
    st.session_state.update({'result_A': result_A, 'result_B': result_B,
                              'objectives': selected_objectives,
                              'electrolyte': selected_electrolyte, 'conc': selected_conc,
                              'max_elements': max_elements})

# ── Display results ────────────────────────────────────────────────────────────
if 'result_A' in st.session_state or 'result_B' in st.session_state:
    result_A   = st.session_state.get('result_A')
    result_B   = st.session_state.get('result_B')
    objectives = st.session_state.get('objectives', [])
    max_el     = st.session_state.get('max_elements', 10)

    if result_A is None and result_B is None:
        st.stop()

    st.divider()
    st.info(f"🌊 Results for **{st.session_state.get('electrolyte','')}** "
            f"at {st.session_state.get('conc','')} M  ·  max {max_el} elements enforced")

    # ── Scatter plots ──────────────────────────────────────────────────────────
    st.subheader("📈 Pareto Fronts")

    # Build all valid axis pairs from selected objectives that have columns in results
    def get_pairs(objectives, df):
        """Return (xcol, ycol, title) pairs for objectives that exist in df."""
        valid = [o for o in objectives if PROP_KEY.get(o) in df.columns]
        mech_o = [o for o in valid if o in ('Tensile Strength','Yield Strength','Elongation','Hardness')]
        corr_o = [o for o in valid if o in ('Ecorr','Epit','icorr')]
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

    ref_df = result_A if result_A is not None else result_B
    pairs  = get_pairs(objectives, ref_df)
    both   = result_A is not None and result_B is not None

    if not pairs:
        st.info("No plottable objective pairs found — select at least 2 objectives with matching result columns.")
    else:
        # Build the list of (result, label, colour) for pipelines that actually ran
        pipeline_results = []
        if result_A is not None: pipeline_results.append((result_A, "Pipeline A", "#1f77b4"))
        if result_B is not None: pipeline_results.append((result_B, "Pipeline B", "#ff7f0e"))
        n_pipelines = len(pipeline_results)

        for x_obj, y_obj, title in pairs:
            xk, yk = PROP_KEY[x_obj], PROP_KEY[y_obj]
            fig, axes = plt.subplots(1, n_pipelines, figsize=(6 * n_pipelines, 4),
                                     sharey=(n_pipelines > 1), squeeze=False)
            axes = axes[0]  # unwrap the row
            for ax, (res, lbl, col) in zip(axes, pipeline_results):
                if xk in res.columns and yk in res.columns:
                    ax.scatter(res[xk], res[yk], c=col, alpha=0.7, edgecolors='white', s=60)
                ax.set_xlabel(xk); ax.set_ylabel(yk)
                ax.set_title(f"{title} — {lbl}"); ax.grid(True, ls='--', alpha=0.4)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Results tables ─────────────────────────────────────────────────────────
    st.divider()

    # Columns to hide from the display table (used internally / as scatter data)
    HIDE_COLS = ['N Elements','FCC probability','BCC probability','HCP probability',
                 'IM probability','Al molar fraction']

    def display_df(df):
        """Drop internal columns from display; keep them in the download."""
        return df.drop(columns=[c for c in HIDE_COLS if c in df.columns]).reset_index(drop=True)

    if both:
        t1, t2 = st.tabs(["Pipeline A", "Pipeline B"])
        with t1:
            st.caption(f"{len(result_A)} alloys · all satisfy ≤ {max_el} elements constraint")
            st.dataframe(display_df(result_A), use_container_width=True)
        with t2:
            st.caption(f"{len(result_B)} alloys · all satisfy ≤ {max_el} elements constraint")
            st.dataframe(display_df(result_B), use_container_width=True)
    elif result_A is not None:
        st.subheader(f"Pipeline A — {len(result_A)} alloys")
        st.caption(f"All alloys satisfy ≤ {max_el} elements constraint")
        st.dataframe(display_df(result_A), use_container_width=True)
    elif result_B is not None:
        st.subheader(f"Pipeline B — {len(result_B)} alloys")
        st.caption(f"All alloys satisfy ≤ {max_el} elements constraint")
        st.dataframe(display_df(result_B), use_container_width=True)

    st.caption("ℹ️ Training database: 2–10 elements (mean 5.3). "
               "Predictions most reliable in the 4–7 element range.")

    # ── Download (includes all columns) ───────────────────────────────────────
    st.divider()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        if result_A is not None: result_A.to_excel(w, sheet_name='Pipeline_A', index=False)
        if result_B is not None: result_B.to_excel(w, sheet_name='Pipeline_B', index=False)
        pd.DataFrame(r2_data).to_excel(w, sheet_name='Model_R2', index=False)
    buf.seek(0)
    st.download_button("⬇️ Download Excel (all results)", data=buf,
                       file_name="MPEA_mech_corr_optimised.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
