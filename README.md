# MPEA Mechanical + Corrosion Design Tool

NSGAN framework extended to mechanical **and** corrosion properties, with three
selectable model pipelines.

---

## Pipelines

| | A | B | C *(recommended)* |
|---|---|---|---|
| **Approach** | Separate models per subset | Unified models on MissForest-imputed data | Stacked, no imputation |
| **Imputed targets** | None | Yes — all 2323 rows | None |
| **Phase features fed to regressors** | Database ground truth | Database ground truth | **Predicted (out-of-fold)** |
| **Phase classifier input** | 54-dim | 54-dim | 54-dim |
| **Missing molarity** | 0 M | 0 M | **Per-electrolyte median** |
| **R² reflects real deployment?** | Optimistic | Optimistic | **Yes** |
| **Best for** | Subset-level benchmarking | Cross-property relationships | Actually generating alloys |

### Why C is the default

Pipelines A and B train their strength/corrosion models using the *true* phase
labels from the database. The app has no true phase at run time — it only has a
classifier's guess. So A and B are trained under conditions they never meet in
production, and their reported R² is correspondingly optimistic.

Pipeline C trains on **out-of-fold predicted** phase labels: every row's phase
comes from a classifier that never saw that row. Training conditions now match
deployment conditions, so its R² is what you actually get on a new alloy. If C's
numbers look slightly lower than A's, that gap *is* the optimism being removed.

---

## Setup — run in this order

```bash
pip install -r requirements.txt

python3 step1_calculate_empirical_params.py   # ~5 s   → *_updated.xlsx
python3 step0_harmonise_processing.py         # ~10 s  (must run AFTER step1)

python3 step2_retrain_models_A.py             # ~2 min → models_A/
python3 step3_retrain_models_B.py             # ~60 min → models_B/
python3 step4_retrain_models_C.py             # ~5 min → models_C/

cp /path/to/generator_net_MPEA.pt models_A/
cp /path/to/generator_net_MPEA.pt models_B/
cp /path/to/generator_net_MPEA.pt models_C/

streamlit run app.py
```

You only need the pipelines you intend to use — the app shows whichever
`models_*/` directories exist.

### Optional

```bash
python3 step5_export_imputed_csvs.py      # fill missing targets, flag which are synthetic
python3 step6_ablation_C.py               # measure each Pipeline C design choice
python3 verify_fixes.py                   # offline consistency checks (no torch needed)
```

---

## Feature vectors

| Layer | Features | Dim |
|---|---|---|
| Element fractions | Ag, Al, B … Zr | 32 |
| Processing (one-hot) | process_1 … process_7 | 7 |
| Empirical parameters | a, δ, Tm, σTm, ΔSmix, ΔHmix, σΔH, Ω, χ̄, σχ, VEC, σVEC, K̄, σK, ρ | 15 |
| **Phase classifier input** | | **54** |
| Phase flags | FCC, BCC, HCP, IM | 4 |
| **Mechanical regressor input** | | **58** |
| Electrolyte (one-hot) | NaCl, H₂SO₄, Seawater, HNO₃, NaOH, HCl, KOH | 7 |
| Concentration | normalised by 6 M | 1 |
| **Corrosion regressor input** | | **66** |

**The phase columns are deliberately excluded from the phase classifiers' own
inputs.** They used to be included, which meant the FCC classifier had `FCC` as
a feature and scored ~100% by reading off the answer. `app.py`'s
`build_base_features()` emits the 54-dim vector; keep it in sync with the
training scripts.

Pipeline B is the exception on routing: every one of its regressors trains on
the full 66-dim imputed matrix, so the app feeds B's mechanical models 66 columns.

---

## Reading the model metrics

The app's performance table is read live from each pipeline's `metrics.json`,
so it can't go stale after a retrain.

### Regressors
Cross-validated R². Pipeline B additionally uses nested CV so the imputer is fit
on training folds only.

### Phase classifiers — do not quote raw accuracy

HCP is present in under 2% of alloys. A model that *never* predicts HCP scores
~98% accuracy while being completely useless. Report these instead:

| Metric | What it means | No-skill value |
|---|---|---|
| **Balanced accuracy** | Average performance on both classes | 0.50 |
| **ROC-AUC** | Shown one alloy with the phase and one without, how often is the right one ranked higher | 0.50 |
| **Average precision** | Best metric when positives are rare | = the positives rate |
| Majority baseline | Score for always guessing the common class | — |

The app flags any classifier that fails to beat its majority baseline.

Classifiers use `class_weight='balanced_subsample'`. This is what gives rare
phases enough spread to be usable as optimisation objectives — without it,
P(HCP) ≈ 0.02 for every alloy and "maximise HCP" has nothing to climb. The
trade-off is that **the probabilities are not calibrated** to the true base
rate; treat them as a relative ranking.

---

## Optimisation objectives

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
| FCC / BCC / HCP / IM | Maximise ↑ | probability (uncalibrated) |
| Aluminum Content | Maximise ↑ | molar ratio |

Constraints available: max element count, allowed-element pool, required
elements, and a fixed 60% cap on any single element.

---

## Processing categories

| Code | Category |
|---|---|
| process_1 | As-cast / arc-melted |
| process_2 | Artificial aging |
| process_3 | Annealing |
| process_4 | Powder metallurgy |
| process_5 | Additive / laser / plasma |
| process_6 | Wrought |
| process_7 | Cryogenic |

---

## Conventions worth not breaking

- **icorr** is trained on log₁₀(icorr); `app.py` back-transforms with `10 ** pred`.
- **Composition canonicalisation** happens exactly once, in
  `latent_to_alloys()`: scale → normalise → drop elements below
  `TRACE_THRESHOLD` (0.005) → renormalise. Constraints, model features,
  density and the displayed alloy name all read that single vector, so the
  alloy shown in the results table is the alloy that was predicted.
- **Phase objectives** use `predict_proba()[:, 1]`, the same array reported in
  the `<phase> probability` columns.
- **Electrolytes**: PBS and Hanks are excluded (n < 15).

---

## Known limitations

- icorr is intrinsically hard to predict from composition; treat its values as
  order-of-magnitude guidance.
- Training data spans 2–10 elements (mean 5.3); predictions are most reliable
  in the 4–7 element range.
- Phase probabilities are uncalibrated (see above).
- Pipeline C reports one number per property from a single CV split. For
  publication, repeat with several random seeds and report the spread.
