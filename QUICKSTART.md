# Quickstart — running everything locally

All local, all through the terminal. No GitHub involved.

You copy in **two files**. Everything else is here, including `step1` and
`step0`, which turn your raw database into the processed one.

---

## Step 1 — Open the folder in Terminal

```bash
cd ~/Downloads/mpea_app_v2        # wherever you unzipped it
ls
```

You should see `app.py`, `preflight.py`, `setup_and_run.sh`, the `step*.py`
scripts, and empty `models_A/`, `models_B/`, `models_C/` folders.

---

## Step 2 — Copy in your two files

Adjust the paths to wherever your existing repo lives:

```bash
cp ~/path/to/MPEA_corr_mech_app/MPEAs_Mech_Corr_DB.xlsx .
cp ~/path/to/MPEA_corr_mech_app/models_A/generator_net_MPEA.pt models_C/
```

That's it. **Do not** copy `MPEAs_Mech_Corr_DB_updated.xlsx` — it gets built
from the raw file in the next step. (If you'd rather reuse your existing
`_updated` file, copy that instead and the build step is skipped.)

If you want to run Pipeline A or B for comparison, the generator goes in those
folders too — but `setup_and_run.sh` copies it across automatically for any
pipeline it trains, so you can skip that.

---

## Step 3 — Run it

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

Seven stages, all automatic:

| | | |
|---|---|---|
| 1 | Python check | instant |
| 2 | Create `.venv` | seconds |
| 3 | Install dependencies | few min (torch is large) |
| 4 | Preflight — stops with a clear message if anything's missing | instant |
| 5 | **`step1` then `step0`** → builds `MPEAs_Mech_Corr_DB_updated.xlsx` | ~15 s |
| 6 | Train Pipeline C | ~5 min |
| 7 | Consistency check, then launch Streamlit | — |

Then open <http://localhost:8501>. `Ctrl+C` stops it.

### Variants

```bash
./setup_and_run.sh --check-only   # verify setup, build nothing
./setup_and_run.sh --with-a       # also train Pipeline A (+2 min) to compare
./setup_and_run.sh --all          # A, B and C — B alone takes ~60 min
./setup_and_run.sh --skip-train   # relaunch the app without retraining
./setup_and_run.sh --force-prep   # rebuild the _updated database from scratch
```

Re-running is always safe. It reuses the venv, skips the dependency install,
and won't rebuild the database unless you ask.

---

## Prefer to type each command yourself?

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 preflight.py                          # confirm setup

python3 step1_calculate_empirical_params.py   # raw -> _updated  (~5 s)
python3 step0_harmonise_processing.py         # fills process_1..7 (~10 s)

python3 step4_retrain_models_C.py             # Pipeline C, ~5 min
python3 step2_retrain_models_A.py             # optional, ~2 min
python3 step3_retrain_models_B.py             # optional, ~60 min

streamlit run app.py
```

**Order matters for the first two:** `step1` creates the `_updated` file that
`step0` then edits in place. Running `step0` first will fail with a missing-file
error.

---

## What to look at once it's running

**Sidebar** — the Pipeline selector now takes any combination of A, B and C.
Start with just C, then add A and compare the Pareto fronts side by side.

**"📊 Model performance summary"** — the interesting part:

- The R² table is read live from each pipeline's `metrics.json`, so it can't
  drift out of date after a retrain.
- **Expect Pipeline C's R² to be a bit lower than A's.** That is the fix
  working, not a regression. A trains on the database's true phase labels,
  which the app never has at run time; C trains on predicted ones. The gap
  between them is roughly how much A was over-reporting.
- The phase table shows accuracy *next to the majority baseline*. Watch HCP —
  if accuracy sits below baseline, the app flags it ⚠️. Judge it on balanced
  accuracy and ROC-AUC instead.

**Run an optimisation** with, say, Tensile Strength + Elongation + icorr. In the
results table the composition shown is now exactly the composition fed to the
models — `Composition sum` should read 1.0 on every row.

---

## Optional: settle the hyperparameter question

```bash
source .venv/bin/activate
python3 step6_ablation_C.py --quick        # 3 folds, faster
python3 step6_ablation_C.py                # full 5 folds, 10–25 min
```

Changes one design decision at a time and reports what each is worth, including
estimated NSGA-II run time — the number that actually matters for the
100-vs-300 trees question. Results land in `ablation_results.csv`.

---

## Troubleshooting

**"NOT READY" from preflight** — it names the specific missing item. Usually the
database or the generator hasn't been copied in.

**`FileNotFoundError: MPEAs_Mech_Corr_DB_updated.xlsx`** when running `step0`
directly — run `step1_calculate_empirical_params.py` first; it creates that file.

**`FCC_classifier.joblib expects 58 features, not 54`** — you're pointing at old
pre-fix models. Retrain that pipeline.

**"No feasible solutions found"** — element constraints are too tight. Widen the
allowed pool, raise max elements, or run more generations.

**`command not found: streamlit`** — the virtual environment isn't active:
`source .venv/bin/activate`.

**Torch install slow or failing on Apple Silicon** — install it alone first:
`pip install torch`, then `pip install -r requirements.txt`.

**Port already in use** — `streamlit run app.py --server.port 8502`

**Start completely over** — `rm -rf .venv MPEAs_Mech_Corr_DB_updated.xlsx models_*/*.joblib models_*/*.json`
then re-run `./setup_and_run.sh`. Your two copied-in files are untouched.
