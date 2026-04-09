# MPEA Mechanical + Corrosion Design Tool

NSGAN framework extended to mechanical AND corrosion properties,
with Pipeline A (separate models) vs Pipeline B (unified imputed models) comparison.

---

## Folder Structure

```
mpea_corr_app/
├── app.py                              ← Streamlit app
├── requirements.txt                    ← Python dependencies
├── step0_harmonise_processing.py       ← Run SECOND (after step1)
├── step1_calculate_empirical_params.py ← Run FIRST
├── step2_retrain_models_A.py           ← Run THIRD
├── step3_retrain_models_B.py           ← Run FOURTH
├── MPEAs_Mech_Corr_DB.xlsx            ← Original database (do not modify)
├── MPEAs_Mech_Corr_DB_updated.xlsx    ← Created by step1 + step0 (with empirical params + processing)
├── imputed_dataset_B.xlsx             ← Created by step3 (full imputed dataset for inspection)
├── models_A/                           ← Created by step2 (Pipeline A models)
│   ├── generator_net_MPEA.pt           ← Copy from original NSGAN repo
│   ├── hardness_regressor.joblib
│   ├── yield_regressor.joblib
│   ├── tensile_regressor.joblib
│   ├── elongation_regressor.joblib
│   ├── ecorr_regressor.joblib
│   ├── epit_regressor.joblib
│   ├── icorr_regressor.joblib          ← predicts log₁₀(icorr); app back-transforms via 10^pred
│   ├── FCC_classifier.joblib
│   ├── BCC_classifier.joblib
│   ├── HCP_classifier.joblib
│   ├── IM_classifier.joblib
│   └── feature_config.json
└── models_B/                           ← Created by step3 (Pipeline B models)
    ├── (same .joblib files as models_A)
    ├── imputer.joblib                  ← Saved MissForest imputer
    └── r2_report.txt                  ← Leakage-free R² summary
```

---

## Setup — Run in This Order

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Calculate empirical parameters (runs in ~5 seconds)
```bash
python3 step1_calculate_empirical_params.py
```
Reads `MPEAs_Mech_Corr_DB.xlsx`, calculates δ, Tm, ΔS_mix, ΔH_mix, Ω, VEC,
density etc. for the 619 corrosion rows, saves `MPEAs_Mech_Corr_DB_updated.xlsx`.

### 3. Harmonise processing metadata (runs in ~10 seconds)
```bash
python3 step0_harmonise_processing.py
```
Maps the 193 free-text `Processing_corr` strings in corrosion rows to the
same 7 canonical one-hot processing columns used by the mechanical rows.
Updates `MPEAs_Mech_Corr_DB_updated.xlsx` in place.

**Must run AFTER step1** (step1 creates the file that step0 reads).

### 4. Train Pipeline A models (~1–5 minutes)
```bash
python3 step2_retrain_models_A.py
```
Trains separate RF regressors on mechanical and corrosion subsets independently.
Feature vector: 58-dim (mechanical) / 66-dim (corrosion).
Saves models to `models_A/`.

### 5. Train Pipeline B models (~60 minutes)
```bash
python3 step3_retrain_models_B.py
```
Runs MissForest imputation (nested K-fold CV for leakage-free R²), trains
unified models on all 2,323 rows, saves models to `models_B/`.
Also saves `imputed_dataset_B.xlsx` with observed vs imputed flags.

### 6. Copy generator to both model folders
```bash
cp /path/to/generator_net_MPEA.pt models_A/
cp /path/to/generator_net_MPEA.pt models_B/
```

### 7. Run the app
```bash
streamlit run app.py
```

---

## Feature Vectors

| Layer | Features | Dim |
|---|---|---|
| Element fractions | Ag, Al, B, ... Zr | 32 |
| Processing (one-hot) | process_1 ... process_7 | 7 |
| Empirical parameters | a, δ, Tm, σTm, ΔSmix, ΔHmix, σΔH, Ω, χ̄, σχ, VEC, σVEC, K̄, σK, ρ | 15 |
| Phase flags | FCC, BCC, HCP, IM | 4 |
| **Mechanical feature total** | | **58** |
| Electrolyte (one-hot) | NaCl, H₂SO₄, Seawater, HNO₃, NaOH, HCl, KOH | 7 |
| Concentration | normalised by 6 M | 1 |
| **Corrosion feature total** | | **66** |

---

## Pipeline Comparison

| | Pipeline A | Pipeline B |
|---|---|---|
| **Approach** | Separate models per subset | Unified models on MissForest-imputed data |
| **Imputation** | None | MissForest (IterativeImputer + RF, max_iter=5) |
| **Training rows** | 334–951 per model | 2,323 (all rows) |
| **R² evaluation** | 5-fold CV (same as Pipeline B) | 5-fold CV, nested imputation (leakage-free) |
| **Best for** | Publication-quality benchmarking | Multi-objective optimisation |

**icorr note:** The icorr model is trained on log₁₀(icorr) values.
The app back-transforms predictions via `10 ** pred` to display µA/cm².
Do NOT change this without updating `app.py`.

---

## Optimisation Objectives

| Objective | Direction | Unit |
|---|---|---|
| Tensile Strength | Maximise ↑ | MPa |
| Yield Strength | Maximise ↑ | MPa |
| Elongation | Maximise ↑ | % |
| Hardness | Maximise ↑ | HV |
| Ecorr | Maximise ↑ | mV vs SCE |
| Epit | Maximise ↑ | mV vs SCE |
| icorr | **Minimise ↓** | µA/cm² |
| Density | **Minimise ↓** | g/cm³ |
| FCC / BCC / HCP / IM | Maximise ↑ | probability |
| Aluminum Content | Maximise ↑ | molar ratio |

---

## Processing Categories

| Code | Category | Description |
|---|---|---|
| process_1 | As-cast / arc-melted | Default; arc melting, vacuum arc, induction, copper mold casting |
| process_2 | Artificial aging | Precipitation hardening after casting |
| process_3 | Annealing | Post-cast heat treatment, homogenisation |
| process_4 | Powder metallurgy | Sintering, ball milling, SPS |
| process_5 | Additive / laser / plasma | SLM, LPBF, EBM, laser cladding, plasma spray, PVD |
| process_6 | Wrought | Rolling, forging, cold work, hot pressing |
| process_7 | Cryogenic | Cryogenic treatment |
