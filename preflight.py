#!/usr/bin/env python3
"""
preflight.py — check everything is in place BEFORE you spend time training.

You only ever need to copy TWO things into this folder:

    MPEAs_Mech_Corr_DB.xlsx    (or the already-processed _updated version)
    generator_net_MPEA.pt      -> into models_C/ (and models_A/, models_B/ if used)

Everything else, including step0 and step1 which turn the raw database into the
_updated one, is already here.

    python3 preflight.py
"""
import importlib
import os
import sys

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
OK, BAD, WARN = f"{GREEN}  ok {RESET}", f"{RED} FAIL{RESET}", f"{YELLOW} warn{RESET}"

problems, warnings_ = [], []


def check(label, passed, detail="", fatal=True):
    print(f"[{OK if passed else (BAD if fatal else WARN)}] {label}"
          + (f"\n        {DIM}{detail}{RESET}" if detail else ""))
    if not passed:
        (problems if fatal else warnings_).append(label)
    return passed


def note(text):
    print(f"        {DIM}{text}{RESET}")


print("\n" + "=" * 70)
print("  MPEA app — preflight check")
print("=" * 70 + "\n")

# ── 1. Python environment ────────────────────────────────────────────────────
print("1. Python environment")
v = sys.version_info
check(f"Python {v.major}.{v.minor}.{v.micro}", v >= (3, 9),
      "Python 3.9 or newer required")

PACKAGES = ['streamlit', 'pandas', 'numpy', 'sklearn', 'torch',
            'pymoo', 'matplotlib', 'openpyxl', 'joblib']
missing = []
for pkg in PACKAGES:
    try:
        m = importlib.import_module(pkg)
        print(f"[{OK}] {pkg:<12} {getattr(m, '__version__', '')}")
    except ImportError:
        print(f"[{BAD}] {pkg:<12} not installed")
        missing.append(pkg)
if missing:
    problems.append("missing packages")
    print(f"\n        {DIM}Fix:  pip install -r requirements.txt{RESET}")

# Later sections need only a subset — a missing torch shouldn't stop us telling
# you whether your database schema is right.
can_read_data   = not {'pandas', 'openpyxl'} & set(missing)
can_load_models = not {'joblib', 'sklearn'} & set(missing)

# ── 2. Data ──────────────────────────────────────────────────────────────────
print("\n2. Database")

has_updated = os.path.exists("MPEAs_Mech_Corr_DB_updated.xlsx")
has_raw     = os.path.exists("MPEAs_Mech_Corr_DB.xlsx")
has_step0   = os.path.exists("step0_harmonise_processing.py")
has_step1   = os.path.exists("step1_calculate_empirical_params.py")

# Two valid starting points. Missing the _updated file is NOT a problem if you
# have the raw one — step1 and step0 generate it, and they ship with this folder.
if has_updated:
    MODE = 'ready'
    check("MPEAs_Mech_Corr_DB_updated.xlsx", True, "Ready to train.")
    if has_raw:
        note("(raw MPEAs_Mech_Corr_DB.xlsx also present — not needed, harmless)")
elif has_raw:
    MODE = 'needs_prep'
    check("MPEAs_Mech_Corr_DB.xlsx (raw)", True,
          "The _updated file will be generated from this — that is expected.")
    ok0 = check("  step1_calculate_empirical_params.py", has_step1)
    ok1 = check("  step0_harmonise_processing.py", has_step0)
    if ok0 and ok1:
        note("Run:  python3 step1_calculate_empirical_params.py")
        note("then: python3 step0_harmonise_processing.py")
        note("(setup_and_run.sh does both automatically)")
else:
    MODE = 'missing'
    check("database", False,
          "Copy ONE of these into this folder:\n"
          "          MPEAs_Mech_Corr_DB.xlsx           (raw — will be processed for you)\n"
          "          MPEAs_Mech_Corr_DB_updated.xlsx   (already processed)")

# ── 3. Generator ─────────────────────────────────────────────────────────────
print("\n3. GAN generator")
gen_found = [d for d in ('models_A', 'models_B', 'models_C')
             if os.path.exists(f"{d}/generator_net_MPEA.pt")]
check("generator_net_MPEA.pt", len(gen_found) > 0,
      f"Found in: {', '.join(gen_found)}" if gen_found else
      "Copy it into each models_* folder you plan to use:\n"
      "          cp /path/to/generator_net_MPEA.pt models_C/")
if gen_found and len(gen_found) < 3:
    note(f"Not in: {', '.join(d for d in ('models_A','models_B','models_C') if d not in gen_found)} "
         "— setup_and_run.sh copies it across for any pipeline you train.")

# ── 4. Schema ────────────────────────────────────────────────────────────────
print("\n4. Database schema")
if MODE != 'missing' and can_read_data:
    import pandas as pd

    path = ("MPEAs_Mech_Corr_DB_updated.xlsx" if MODE == 'ready'
            else "MPEAs_Mech_Corr_DB.xlsx")
    df = pd.read_excel(path)
    note(f"checking {path} — {len(df)} rows x {df.shape[1]} columns")

    ELEM = ['Ag','Al','B','C','Ca','Co','Cr','Cu','Fe','Ga','Ge','Hf','Li','Mg',
            'Mn','Mo','N','Nb','Nd','Ni','Pd','Re','Sc','Si','Sn','Ta','Ti','V',
            'W','Y','Zn','Zr']
    PROC = [f'process_{i}' for i in range(1, 8)]
    EMP  = ['a','delta','Tm','std of Tm','entropy','enthalpy','std of enthalpy',
            'omega','X','std of X','VEC','std of vec','K','std of K','density']
    PHASE   = ['FCC','BCC','HCP','IM']
    META    = ['OG property','Electrolyte','Concentration in M']
    TARGETS = ['Hardness (HVN)','Yield Strength (MPa)',
               'Ultimate Tensile Strength (MPa)','Elongation (%)',
               'Corrosion potential (mV vs SCE)','Pitting potential (mV vs SCE)',
               'Corrosion current density (microA/cm2)']

    for label, cols in [('32 element columns', ELEM), ('7 processing columns', PROC),
                        ('15 empirical columns', EMP), ('4 phase columns', PHASE),
                        ('metadata columns', META), ('7 target columns', TARGETS)]:
        absent = [c for c in cols if c not in df.columns]
        check(label, not absent, f"missing: {absent}" if absent else "")

    if 'OG property' in df.columns:
        counts = df['OG property'].value_counts().to_dict()
        check("'OG property' has mechanical + corrosion rows",
              {'mechanical', 'corrosion'} <= set(counts), f"{counts}")

    if MODE == 'needs_prep':
        # step0 needs this free-text column to build the one-hots
        check("'Processing_corr' column (needed by step0)",
              'Processing_corr' in df.columns,
              "" if 'Processing_corr' in df.columns else
              "step0_harmonise_processing.py cannot run without it")
        note("empirical parameters not checked yet — step1 fills them in")
    elif 'delta' in df.columns and 'OG property' in df.columns:
        corr_rows = df[df['OG property'] == 'corrosion']
        # coerce: these columns can carry stray text in a hand-built workbook
        dl = pd.to_numeric(corr_rows['delta'], errors='coerce')
        blank = int((dl.isna() | (dl == 0)).sum())
        check("corrosion rows have empirical parameters",
              blank < max(1, 0.1 * len(corr_rows)),
              f"{blank}/{len(corr_rows)} corrosion rows have no delta — "
              "re-run step1_calculate_empirical_params.py" if blank else "")
        proc_sum = df.loc[df['OG property'] == 'corrosion', PROC].apply(
            pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
        check("corrosion rows have a processing flag",
              (proc_sum == 1).mean() > 0.95,
              f"only {(proc_sum == 1).mean():.0%} have exactly one — "
              "run step0_harmonise_processing.py" if (proc_sum == 1).mean() <= 0.95 else "")

    if 'Electrolyte' in df.columns:
        supported = ['NaCl','H2SO4','Seawater','HNO3','NaOH','HCl','KOH']
        # This column mixes types: strings for corrosion rows, numeric 0 for
        # mechanical ones. Normalise to str before comparing or sorting --
        # sorted() cannot order 'NaCl' against 0.
        PLACEHOLDERS = {'', '0', '0.0', 'nan', 'none', 'na', 'n/a', '-', 'missing'}
        labels = [str(v).strip() for v in df['Electrolyte'].dropna().tolist()]
        found  = {s for s in labels if s.lower() not in PLACEHOLDERS}
        known, extra = sorted(found & set(supported)), sorted(found - set(supported))
        check("electrolytes recognised", bool(known),
              (f"supported: {known}" + (f"  |  ignored: {extra}" if extra else ""))
              if known else
              "no supported electrolyte found -- corrosion models would have no rows")

        if 'OG property' in df.columns:
            cmask = df['OG property'] == 'corrosion'
            usable = int(df.loc[cmask, 'Electrolyte'].astype(str).str.strip()
                         .isin(supported).sum())
            total = int(cmask.sum())
            check("corrosion rows with a supported electrolyte", usable > 0,
                  f"{usable}/{total} usable"
                  + ("  (PBS/Hanks and blanks excluded by design)"
                     if usable < total else ""))
elif MODE == 'missing':
    note("skipped — no database present yet")
else:
    note("skipped — needs pandas + openpyxl")

# ── 5. Existing models ───────────────────────────────────────────────────────
print("\n5. Trained models (if any)")
EXPECTED = {'FCC_classifier': 54, 'BCC_classifier': 54, 'HCP_classifier': 54,
            'IM_classifier': 54, 'hardness_regressor': 58, 'yield_regressor': 58,
            'tensile_regressor': 58, 'elongation_regressor': 58,
            'ecorr_regressor': 66, 'epit_regressor': 66, 'icorr_regressor': 66}
any_models = False
for d in ('models_A', 'models_B', 'models_C'):
    files = [f for f in EXPECTED if os.path.exists(f"{d}/{f}.joblib")]
    if not files:
        note(f"{d}: not trained yet")
        continue
    any_models = True
    if not can_load_models:
        note(f"{d}: {len(files)} model files present, needs joblib + sklearn to inspect")
        continue
    from joblib import load
    bad = []
    for f in files:
        n = getattr(load(f"{d}/{f}.joblib"), 'n_features_in_', None)
        exp = EXPECTED[f]
        # Pipeline B trains every regressor on the 66-dim imputed matrix
        if d == 'models_B' and f.endswith('_regressor'):
            exp = 66
        if n is not None and n != exp:
            bad.append(f"{f}: {n} (expected {exp})")
    check(f"{d} feature widths", not bad,
          "; ".join(bad) + "  -> retrain, these are the old leaky models" if bad else
          f"{len(files)}/{len(EXPECTED)} models present")
    check(f"{d}/metrics.json", os.path.exists(f"{d}/metrics.json"),
          "app.py falls back to built-in numbers without it", fatal=False)

if not any_models:
    note("Nothing trained yet — expected on a fresh copy.")

# ── Verdict ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
if problems:
    print(f"{RED}  NOT READY — {len(problems)} blocking issue(s){RESET}")
    for p in problems:
        print(f"    - {p}")
    print("\n  Fix the above, then run preflight.py again.")
    sys.exit(1)

print(f"{GREEN}  READY{RESET}")
if warnings_:
    print(f"{YELLOW}  {len(warnings_)} non-blocking warning(s): {', '.join(warnings_)}{RESET}")

if MODE == 'needs_prep':
    print("""
  Next — process the database, then train:
      python3 step1_calculate_empirical_params.py     # ~5 s
      python3 step0_harmonise_processing.py           # ~10 s
      python3 step4_retrain_models_C.py               # ~5 min
      streamlit run app.py

  Or just:  ./setup_and_run.sh
""")
else:
    print("""
  Next:
      python3 step4_retrain_models_C.py     # ~5 min, the recommended pipeline
      python3 step2_retrain_models_A.py     # ~2 min, optional, for comparison
      streamlit run app.py

  Or just:  ./setup_and_run.sh
""")
print("=" * 70 + "\n")
