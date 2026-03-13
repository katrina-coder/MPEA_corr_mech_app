# MPEA Mechanical + Corrosion Design Tool

NSGAN framework extended to mechanical AND corrosion properties,
with Pipeline A vs B comparison.

---

## Folder Structure

```
mpea_corr_app/
├── app.py                              ← Streamlit app
├── requirements.txt                    ← Dependencies
├── step1_calculate_empirical_params.py ← Run ONCE first
├── step2_retrain_models_A.py           ← Run ONCE second
├── step3_retrain_models_B.py           ← Run ONCE third
├── MPEAs_Mech_Corr_DB.xlsx            ← Your original database
├── MPEAs_Mech_Corr_DB_updated.xlsx    ← Created by step1 (with empirical params)
├── models_A/                           ← Created by step2 (Pipeline A models)
│   ├── generator_net_MPEA.pt
│   ├── hardness_regressor.joblib
│   ├── yield_regressor.joblib
│   ├── tensile_regressor.joblib
│   ├── elongation_regressor.joblib
│   ├── ecorr_regressor.joblib
│   ├── epit_regressor.joblib
│   ├── icorr_regressor.joblib          ← predicts log10(icorr)
│   ├── FCC_classifier.joblib
│   ├── BCC_classifier.joblib
│   ├── HCP_classifier.joblib
│   └── IM_classifier.joblib
└── models_B/                           ← Created by step3 (Pipeline B models)
    └── (same files as models_A)
```

---

## Setup — Run in This Order

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Calculate empirical parameters (~5 seconds)
```bash
python3 step1_calculate_empirical_params.py
```
Calculates δ, Tm, ΔS_mix, ΔH_mix, Ω, VEC, density etc. for the 619
corrosion rows, saves `MPEAs_Mech_Corr_DB_updated.xlsx`

### 3. Train Pipeline A models (~1 minute)
```bash
python3 step2_retrain_models_A.py
```
Trains separate RF models for mechanical and corrosion subsets.
Saves models to `models_A/`

### 4. Train Pipeline B models (~3–5 minutes)
```bash
python3 step3_retrain_models_B.py
```
Runs MissForest imputation on the combined dataset,
then trains unified models on all 2323 rows.
Saves models to `models_B/`

### 5. Copy generator to both model folders
The GAN generator file from the original MPEA app is shared:
```bash
cp /path/to/original/mpea_app/models/generator_net_MPEA.pt models_A/
cp /path/to/original/mpea_app/models/generator_net_MPEA.pt models_B/
```

### 6. Run the app
```bash
streamlit run app.py
```

---

## Pipeline Comparison

| | Pipeline A | Pipeline B |
|---|---|---|
| **Approach** | Separate models per subset | Unified models on imputed data |
| **Hardness R²** | 0.894 | 0.915 |
| **Tensile R²** | 0.676 | 0.790 |
| **Ecorr R²** | 0.435 | 0.665 |
| **Epit R²** | 0.774 | 0.872 |
| **icorr R²** | 0.133 | 0.081 |
| **Training samples** | 334–951 per model | 2323 (all) |
| **Imputation noise** | None | Yes |
| **Best for** | Trustworthy predictions | Cross-property relationships |

**Note:** icorr is inherently hard to predict from composition alone
(R² < 0.15 in both pipelines). Ecorr and Epit are better behaved.
The icorr model predicts log₁₀(icorr); the app back-transforms to µA/cm².

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
