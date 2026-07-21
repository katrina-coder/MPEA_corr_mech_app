#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  setup_and_run.sh — one command from a fresh copy to a running app
#
#    ./setup_and_run.sh                 venv + deps + data prep + C + app
#    ./setup_and_run.sh --with-a        also train Pipeline A  (+2 min)
#    ./setup_and_run.sh --with-b        also train Pipeline B  (+60 min!)
#    ./setup_and_run.sh --all           A, B and C
#    ./setup_and_run.sh --skip-train    just launch the app with existing models
#    ./setup_and_run.sh --check-only    stop after preflight
#    ./setup_and_run.sh --force-prep    re-run step1 + step0 even if _updated exists
#
#  You only need to copy TWO things into this folder yourself:
#      MPEAs_Mech_Corr_DB.xlsx        (the raw database)
#      generator_net_MPEA.pt          (into models_C/)
#  step1 + step0 turn the raw database into MPEAs_Mech_Corr_DB_updated.xlsx
#  automatically — this script runs them for you if that file is missing.
#
#  Safe to re-run. Skips the venv, the dependency install and the data prep
#  if they're already done.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"

TRAIN_A=0; TRAIN_B=0; TRAIN_C=1; SKIP_TRAIN=0; CHECK_ONLY=0; FORCE_PREP=0
for arg in "$@"; do
  case "$arg" in
    --with-a)     TRAIN_A=1 ;;
    --with-b)     TRAIN_B=1 ;;
    --all)        TRAIN_A=1; TRAIN_B=1 ;;
    --skip-train) SKIP_TRAIN=1 ;;
    --check-only) CHECK_ONLY=1 ;;
    --force-prep) FORCE_PREP=1 ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg  (try --help)"; exit 1 ;;
  esac
done

B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
step() { echo; echo "${B}▶ $1${N}"; }
die()  { echo "${R}✖ $1${N}"; exit 1; }

RAW="MPEAs_Mech_Corr_DB.xlsx"
UPD="MPEAs_Mech_Corr_DB_updated.xlsx"

# ── 1. Python ───────────────────────────────────────────────────────────────
step "1/7  Python"
PY=$(command -v python3 || true)
[ -n "$PY" ] || die "python3 not found. Install Python 3.9+ and re-run."
echo "     $($PY --version)  at  $PY"

# ── 2. Virtual environment ──────────────────────────────────────────────────
step "2/7  Virtual environment"
if [ ! -d ".venv" ]; then
  echo "     creating .venv ..."
  "$PY" -m venv .venv
else
  echo "     .venv already exists — reusing"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "     active: $(command -v python)"

# ── 3. Dependencies ─────────────────────────────────────────────────────────
step "3/7  Dependencies"
if [ -f ".venv/.deps_installed" ] && [ requirements.txt -ot .venv/.deps_installed ]; then
  echo "     already installed — skipping (delete .venv/.deps_installed to force)"
else
  python -m pip install --quiet --upgrade pip
  echo "     installing (torch is large; first run may take several minutes) ..."
  python -m pip install --quiet -r requirements.txt
  touch .venv/.deps_installed
  echo "${G}     done${N}"
fi

# ── 4. Preflight ────────────────────────────────────────────────────────────
step "4/7  Preflight checks"
if ! python preflight.py; then
  echo
  echo "${Y}Preflight failed. The two files you must copy in yourself are:${N}"
  echo "  1)  $RAW                    -> this folder"
  echo "  2)  generator_net_MPEA.pt   -> models_C/"
  echo
  echo "Roughly:"
  echo "  cp ~/path/to/MPEA_corr_mech_app/$RAW ."
  echo "  cp ~/path/to/MPEA_corr_mech_app/models_A/generator_net_MPEA.pt models_C/"
  exit 1
fi
[ "$CHECK_ONLY" -eq 1 ] && { echo "${G}Check-only mode — stopping here.${N}"; exit 0; }

# ── 5. Prepare the database ─────────────────────────────────────────────────
step "5/7  Database preparation"
if [ -f "$UPD" ] && [ "$FORCE_PREP" -eq 0 ]; then
  echo "     $UPD already exists — skipping"
  echo "     ${Y}(use --force-prep to rebuild it from $RAW)${N}"
elif [ -f "$RAW" ]; then
  echo "     building $UPD from $RAW"
  echo
  echo "${B}     step1 — empirical parameters (~5 s)${N}"
  python step1_calculate_empirical_params.py
  echo
  echo "${B}     step0 — harmonise processing routes (~10 s)${N}"
  python step0_harmonise_processing.py
  echo "${G}     $UPD ready${N}"
else
  die "Neither $UPD nor $RAW found. Copy one into this folder."
fi

# ── 6. Training ─────────────────────────────────────────────────────────────
step "6/7  Training"
if [ "$SKIP_TRAIN" -eq 1 ]; then
  echo "     --skip-train given — using whatever models are already on disk"
else
  if [ "$TRAIN_C" -eq 1 ]; then
    echo; echo "${B}     Pipeline C  (~5 min)${N}"
    python step4_retrain_models_C.py
  fi
  if [ "$TRAIN_A" -eq 1 ]; then
    echo; echo "${B}     Pipeline A  (~2 min)${N}"
    python step2_retrain_models_A.py
  fi
  if [ "$TRAIN_B" -eq 1 ]; then
    echo; echo "${B}     Pipeline B  (~60 min — this one is slow)${N}"
    python step3_retrain_models_B.py
  fi
fi

# Make sure every trained pipeline has the generator alongside it
for d in models_A models_B models_C; do
  if compgen -G "$d/*_regressor.joblib" > /dev/null && [ ! -f "$d/generator_net_MPEA.pt" ]; then
    for src in models_C models_A models_B; do
      if [ -f "$src/generator_net_MPEA.pt" ]; then
        cp "$src/generator_net_MPEA.pt" "$d/"
        echo "     copied generator into $d/"
        break
      fi
    done
  fi
done

# ── 7. Launch ───────────────────────────────────────────────────────────────
step "7/7  Consistency check, then launch"
# Non-fatal by design: this is a self-diagnostic, not a gate. A problem in the
# checker itself must never stop the app from starting.
if VERIFY_OUT=$(python verify_fixes.py 2>&1); then
  echo "$VERIFY_OUT" | tail -4
else
  echo "${Y}     self-check did not complete — this does NOT affect the app.${N}"
  echo "${Y}     last lines:${N}"
  echo "$VERIFY_OUT" | tail -6 | sed 's/^/       /'
fi
echo
echo "${G}Starting Streamlit — it should open at http://localhost:8501${N}"
echo "${Y}Press Ctrl+C to stop.${N}"
echo
exec streamlit run app.py
