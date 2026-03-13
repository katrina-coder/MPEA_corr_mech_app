"""
step1_calculate_empirical_params.py
─────────────────────────────────────
Calculates all empirical parameters for corrosion rows (which have zeros)
and saves the completed database as  MPEAs_Mech_Corr_DB_updated.xlsx

Empirical parameters calculated:
  - a         : mean lattice parameter (Å)
  - delta      : atomic size mismatch (%)
  - Tm         : mean melting temperature (K)
  - std of Tm  : std dev of melting temperature
  - entropy    : mixing entropy ΔS_mix (J/mol·K)
  - enthalpy   : mixing enthalpy ΔH_mix (kJ/mol)
  - std of enthalpy
  - omega      : Ω = Tm·ΔSmix / |ΔHmix|
  - X          : mean Pauling electronegativity
  - std of X   : std dev of electronegativity
  - VEC        : valence electron concentration
  - std of vec : std dev of VEC
  - K          : mean bulk modulus (GPa)
  - std of K   : std dev of bulk modulus
  - density    : theoretical density (g/cm³)

Usage:
    python3 step1_calculate_empirical_params.py
"""

import pandas as pd
import numpy as np

# ── Elemental data (same 32 elements as original MPEA dataset) ────────────────
# Order: Ag, Al, B, C, Ca, Co, Cr, Cu, Fe, Ga, Ge, Hf, Li, Mg, Mn, Mo,
#        N,  Nb, Nd, Ni, Pd, Re, Sc, Si, Sn, Ta, Ti, V,  W,  Y,  Zn, Zr

ELEMENTS = ['Ag','Al','B','C','Ca','Co','Cr','Cu','Fe','Ga','Ge','Hf',
            'Li','Mg','Mn','Mo','N','Nb','Nd','Ni','Pd','Re','Sc','Si',
            'Sn','Ta','Ti','V','W','Y','Zn','Zr']

# Atomic radii (Å) — Goldschmidt/Slater radii
ATOMIC_RADII = {
    'Ag':1.44,'Al':1.43,'B':0.87,'C':0.77,'Ca':1.97,'Co':1.25,'Cr':1.28,
    'Cu':1.28,'Fe':1.26,'Ga':1.22,'Ge':1.22,'Hf':1.59,'Li':1.52,'Mg':1.60,
    'Mn':1.26,'Mo':1.36,'N':0.75,'Nb':1.43,'Nd':1.82,'Ni':1.24,'Pd':1.37,
    'Re':1.37,'Sc':1.62,'Si':1.18,'Sn':1.40,'Ta':1.43,'Ti':1.47,'V':1.34,
    'W':1.37,'Y':1.80,'Zn':1.33,'Zr':1.60
}

# Melting temperatures (K)
MELTING_TEMPS = {
    'Ag':1235,'Al':933,'B':2349,'C':3823,'Ca':1115,'Co':1768,'Cr':2180,
    'Cu':1358,'Fe':1811,'Ga':303,'Ge':1211,'Hf':2506,'Li':454,'Mg':923,
    'Mn':1519,'Mo':2896,'N':63,'Nb':2750,'Nd':1297,'Ni':1728,'Pd':1828,
    'Re':3459,'Sc':1814,'Si':1687,'Sn':505,'Ta':3290,'Ti':1941,'V':2183,
    'W':3695,'Y':1799,'Zn':693,'Zr':2128
}

# Pauling electronegativities
ELECTRONEG = {
    'Ag':1.93,'Al':1.61,'B':2.04,'C':2.55,'Ca':1.00,'Co':1.88,'Cr':1.66,
    'Cu':1.90,'Fe':1.83,'Ga':1.81,'Ge':2.01,'Hf':1.30,'Li':0.98,'Mg':1.31,
    'Mn':1.55,'Mo':2.16,'N':3.04,'Nb':1.60,'Nd':1.14,'Ni':1.91,'Pd':2.20,
    'Re':1.90,'Sc':1.36,'Si':1.90,'Sn':1.96,'Ta':1.50,'Ti':1.54,'V':1.63,
    'W':2.36,'Y':1.22,'Zn':1.65,'Zr':1.33
}

# Valence electron counts
VEC_VALS = {
    'Ag':11,'Al':3,'B':3,'C':4,'Ca':2,'Co':9,'Cr':6,'Cu':11,'Fe':8,'Ga':3,
    'Ge':4,'Hf':4,'Li':1,'Mg':2,'Mn':7,'Mo':6,'N':5,'Nb':5,'Nd':4,'Ni':10,
    'Pd':10,'Re':7,'Sc':3,'Si':4,'Sn':4,'Ta':5,'Ti':4,'V':5,'W':6,'Y':3,
    'Zn':12,'Zr':4
}

# Molar masses (g/mol)
MOLAR_MASSES = {
    'Ag':107.87,'Al':26.98,'B':10.81,'C':12.01,'Ca':40.08,'Co':58.93,
    'Cr':52.00,'Cu':63.55,'Fe':55.85,'Ga':69.72,'Ge':72.63,'Hf':178.49,
    'Li':6.94,'Mg':24.31,'Mn':54.94,'Mo':95.96,'N':14.01,'Nb':92.91,
    'Nd':144.24,'Ni':58.69,'Pd':106.42,'Re':186.21,'Sc':44.96,'Si':28.09,
    'Sn':118.71,'Ta':180.95,'Ti':47.87,'V':50.94,'W':183.84,'Y':88.91,
    'Zn':65.38,'Zr':91.22
}

# Molar volumes (cm³/mol)
MOLAR_VOLUMES = {
    'Ag':10.27,'Al':10.00,'B':4.39,'C':5.29,'Ca':26.20,'Co':6.67,'Cr':7.23,
    'Cu':7.11,'Fe':7.09,'Ga':11.80,'Ge':13.63,'Hf':13.44,'Li':13.02,'Mg':14.00,
    'Mn':7.35,'Mo':9.38,'N':13.54,'Nb':10.83,'Nd':20.59,'Ni':6.59,'Pd':8.56,
    'Re':8.86,'Sc':15.00,'Si':12.06,'Sn':16.29,'Ta':10.85,'Ti':10.64,'V':8.32,
    'W':9.47,'Y':19.88,'Zn':9.16,'Zr':14.02
}

# Lattice parameters (Å) — approximate for pure metals
LATTICE_PARAMS = {
    'Ag':4.09,'Al':4.05,'B':5.06,'C':3.57,'Ca':5.58,'Co':2.51,'Cr':2.88,
    'Cu':3.62,'Fe':2.87,'Ga':4.52,'Ge':5.66,'Hf':3.20,'Li':3.51,'Mg':3.21,
    'Mn':8.91,'Mo':3.15,'N':4.04,'Nb':3.30,'Nd':3.66,'Ni':3.52,'Pd':3.89,
    'Re':2.76,'Sc':3.31,'Si':5.43,'Sn':5.83,'Ta':3.31,'Ti':2.95,'V':3.02,
    'W':3.16,'Y':3.65,'Zn':2.66,'Zr':3.23
}

# Bulk moduli (GPa)
BULK_MODULI = {
    'Ag':100,'Al':76,'B':320,'C':443,'Ca':17,'Co':180,'Cr':160,'Cu':140,
    'Fe':170,'Ga':59,'Ge':75,'Hf':110,'Li':11,'Mg':45,'Mn':120,'Mo':230,
    'N':0,'Nb':170,'Nd':32,'Ni':180,'Pd':180,'Re':370,'Sc':57,'Si':98,
    'Sn':58,'Ta':200,'Ti':110,'V':160,'W':310,'Y':41,'Zn':70,'Zr':94
}

# Binary mixing enthalpy parameters (Ω_ij in kJ/mol) — Takeuchi & Inoue
# Using a representative subset; remaining pairs default to 0
ENTHALPY_PARAMS = {
    ('Al','Co'):-19,('Al','Cr'):-10,('Al','Cu'):-1,('Al','Fe'):-11,
    ('Al','Hf'):-45,('Al','Mg'):-2,('Al','Mn'):-19,('Al','Mo'):-22,
    ('Al','Nb'):-18,('Al','Ni'):-22,('Al','Si'):-19,('Al','Ta'):-19,
    ('Al','Ti'):-30,('Al','V'):-16,('Al','W'):-16,('Al','Zr'):-44,
    ('Co','Cr'):-4,('Co','Cu'):6,('Co','Fe'):0,('Co','Mn'):0,
    ('Co','Mo'):-5,('Co','Nb'):-25,('Co','Ni'):0,('Co','Ti'):-28,
    ('Co','V'):-14,('Co','W'):-1,('Co','Zr'):-41,
    ('Cr','Cu'):12,('Cr','Fe'):-1,('Cr','Mn'):2,('Cr','Mo'):0,
    ('Cr','Nb'):-7,('Cr','Ni'):-7,('Cr','Si'):-37,('Cr','Ta'):-7,
    ('Cr','Ti'):-7,('Cr','V'):-2,('Cr','W'):0,('Cr','Zr'):-12,
    ('Cu','Fe'):13,('Cu','Mn'):4,('Cu','Mo'):19,('Cu','Ni'):4,
    ('Cu','Ti'):-9,('Cu','Zr'):-23,
    ('Fe','Mn'):0,('Fe','Mo'):-2,('Fe','Nb'):-16,('Fe','Ni'):-2,
    ('Fe','Si'):-35,('Fe','Ta'):-15,('Fe','Ti'):-17,('Fe','V'):-7,
    ('Fe','W'):-6,('Fe','Zr'):-25,
    ('Mn','Mo'):0,('Mn','Ni'):-8,('Mn','Ti'):-8,('Mn','V'):-1,
    ('Mo','Nb'):-6,('Mo','Ni'):-7,('Mo','Si'):-38,('Mo','Ta'):-5,
    ('Mo','Ti'):-4,('Mo','V'):-5,('Mo','W'):0,('Mo','Zr'):-6,
    ('Nb','Ni'):-30,('Nb','Si'):-56,('Nb','Ta'):0,('Nb','Ti'):-2,
    ('Nb','V'):-2,('Nb','W'):-8,('Nb','Zr'):4,
    ('Ni','Si'):-40,('Ni','Ta'):-24,('Ni','Ti'):-35,('Ni','V'):-18,
    ('Ni','W'):-3,('Ni','Zr'):-49,
    ('Si','Ta'):-45,('Si','Ti'):-66,('Si','V'):-48,('Si','W'):-37,
    ('Si','Zr'):-84,('Ta','Ti'):-4,('Ta','V'):-1,('Ta','W'):-7,
    ('Ti','V'):-2,('Ti','W'):-27,('Ti','Zr'):0,
    ('V','W'):-8,('V','Zr'):-4,('W','Zr'):-27,
}

R_GAS = 8.314  # J/(mol·K)


def get_enthalpy_pair(e1, e2):
    """Look up binary mixing enthalpy, symmetric."""
    key = (e1, e2) if (e1, e2) in ENTHALPY_PARAMS else (e2, e1)
    return ENTHALPY_PARAMS.get(key, 0.0)


def calc_empirical_params(row):
    """Calculate all 15 empirical parameters for one alloy row."""
    # Extract molar fractions for non-zero elements
    fracs = {}
    for e in ELEMENTS:
        val = float(row.get(e, 0.0))
        if val > 0:
            fracs[e] = val

    if not fracs:
        return {k: 0.0 for k in ['a','delta','Tm','std of Tm','entropy',
                                   'enthalpy','std of enthalpy','omega',
                                   'X','std of X','VEC','std of vec',
                                   'K','std of K','density']}

    # Normalise fractions
    total = sum(fracs.values())
    x = {e: v/total for e, v in fracs.items()}
    elems = list(x.keys())

    # Mean lattice parameter
    a_mean = sum(x[e] * LATTICE_PARAMS[e] for e in elems)

    # Atomic size mismatch δ (%)
    r_mean = sum(x[e] * ATOMIC_RADII[e] for e in elems)
    delta = 100 * np.sqrt(sum(x[e] * (1 - ATOMIC_RADII[e]/r_mean)**2 for e in elems))

    # Melting temperature Tm (K) and its std
    tm_mean = sum(x[e] * MELTING_TEMPS[e] for e in elems)
    tm_std  = np.sqrt(sum(x[e] * (MELTING_TEMPS[e] - tm_mean)**2 for e in elems))

    # Mixing entropy ΔS_mix (J/mol·K)  = -R Σ xi ln(xi)
    entropy = -R_GAS * sum(xi * np.log(xi) for xi in x.values())

    # Mixing enthalpy ΔH_mix (kJ/mol) = Σ_ij 4 Ω_ij xi xj
    enthalpy = 0.0
    enthalpy_sq = 0.0
    for i, e1 in enumerate(elems):
        for e2 in elems[i+1:]:
            pair_h = get_enthalpy_pair(e1, e2)
            contrib = 4 * pair_h * x[e1] * x[e2]
            enthalpy     += contrib
            enthalpy_sq  += contrib**2
    enthalpy_std = np.sqrt(enthalpy_sq) if enthalpy_sq > 0 else 0.0

    # Omega Ω = Tm * ΔSmix / |ΔHmix|  (dimensionless)
    omega = (tm_mean * entropy / (abs(enthalpy) * 1000)) if enthalpy != 0 else 0.0

    # Electronegativity mean and std
    x_mean = sum(x[e] * ELECTRONEG[e] for e in elems)
    x_std  = np.sqrt(sum(x[e] * (ELECTRONEG[e] - x_mean)**2 for e in elems))

    # Valence electron concentration
    vec_mean = sum(x[e] * VEC_VALS[e] for e in elems)
    vec_std  = np.sqrt(sum(x[e] * (VEC_VALS[e] - vec_mean)**2 for e in elems))

    # Bulk modulus mean and std
    k_mean = sum(x[e] * BULK_MODULI[e] for e in elems)
    k_std  = np.sqrt(sum(x[e] * (BULK_MODULI[e] - k_mean)**2 for e in elems))

    # Theoretical density (g/cm³)
    mass_mix   = sum(x[e] * MOLAR_MASSES[e]  for e in elems)
    volume_mix = sum(x[e] * MOLAR_VOLUMES[e] for e in elems)
    density = mass_mix / volume_mix if volume_mix > 0 else 0.0

    return {
        'a':              round(a_mean, 6),
        'delta':          round(delta, 6),
        'Tm':             round(tm_mean, 4),
        'std of Tm':      round(tm_std, 4),
        'entropy':        round(entropy, 6),
        'enthalpy':       round(enthalpy, 6),
        'std of enthalpy':round(enthalpy_std, 6),
        'omega':          round(omega, 6),
        'X':              round(x_mean, 6),
        'std of X':       round(x_std, 6),
        'VEC':            round(vec_mean, 6),
        'std of vec':     round(vec_std, 6),
        'K':              round(k_mean, 4),
        'std of K':       round(k_std, 4),
        'density':        round(density, 6),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading database...")
    df = pd.read_excel('MPEAs_Mech_Corr_DB.xlsx')
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    EMP_COLS = ['a','delta','Tm','std of Tm','entropy','enthalpy',
                'std of enthalpy','omega','X','std of X','VEC',
                'std of vec','K','std of K','density']

    corr_mask = df['OG property'] == 'corrosion'
    corr_idx  = df[corr_mask].index
    print(f"\nCalculating empirical parameters for {len(corr_idx)} corrosion rows...")

    results = []
    for i, idx in enumerate(corr_idx):
        params = calc_empirical_params(df.loc[idx])
        results.append(params)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(corr_idx)} rows...")

    params_df = pd.DataFrame(results, index=corr_idx)

    # Write calculated values back into the dataframe
    for col in EMP_COLS:
        df.loc[corr_idx, col] = params_df[col]

    # Validation
    print("\n=== Validation — sample of calculated values (first 3 corrosion rows) ===")
    print(df.loc[corr_idx[:3], ['Composition'] + EMP_COLS].to_string())

    print("\n=== Cross-check vs mechanical rows (means should be broadly similar) ===")
    for col in ['delta', 'Tm', 'entropy', 'VEC', 'density']:
        mech_mean = df[df['OG property']=='mechanical'][col].mean()
        corr_mean = df[df['OG property']=='corrosion'][col].mean()
        print(f"  {col:15s}:  mechanical={mech_mean:.4f}  corrosion={corr_mean:.4f}")

    # Save updated database
    out_path = 'MPEAs_Mech_Corr_DB_updated.xlsx'
    df.to_excel(out_path, index=False, engine='openpyxl')
    print(f"\n✅  Updated database saved to: {out_path}")
    print(f"    Shape: {df.shape}")
    print(f"    Corrosion rows now have full empirical parameters.")
