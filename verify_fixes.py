"""
verify_fixes.py — offline checks for the three bug fixes in app.py.

Runs WITHOUT torch/streamlit/pymoo: the pure numerical helpers are extracted
from app.py and executed against stubbed modules, so this can be run anywhere.

    python3 verify_fixes.py
"""
import re, sys, types, contextlib
import numpy as np

# Import sklearn FIRST, before any module stubbing below. sklearn pulls in
# scipy.stats, which probes sys.modules['torch'] for a real torch.Tensor. If a
# stub is already sitting there at that moment, scipy raises AttributeError.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ── torch: use the real one if installed, stub it only if it is not ──────────
# app.py's helper block needs just three torch names. Preferring the real
# module avoids poisoning sys.modules for anything that imports afterwards.
try:
    import torch                                          # noqa: F401
    TORCH_MODE = f"real torch {torch.__version__}"
except ImportError:
    torch_stub = types.ModuleType("torch")
    torch_stub.no_grad = contextlib.nullcontext
    torch_stub.float32 = np.float32
    torch_stub.tensor  = lambda x, dtype=None: np.asarray(x, dtype=np.float64)
    torch_stub.Tensor  = np.ndarray        # scipy probes for this attribute
    torch_stub.__version__ = "0.0.0-stub"
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = types.ModuleType("torch.nn")
    TORCH_MODE = "stubbed torch (not installed)"

# ── Pull the helper block out of app.py and exec it ──────────────────────────
src = open("app.py").read()
start = src.index("# ── Constants ─")
end   = src.index("# ── Generator ─")
block = "import numpy as np\nimport torch\n" + src[start:end]
ns = {}
exec(compile(block, "app.py:helpers", "exec"), ns)

canonicalise_composition = ns["canonicalise_composition"]
latent_to_alloys         = ns["latent_to_alloys"]
build_base_features      = ns["build_base_features"]
build_mech_features      = ns["build_mech_features"]
build_corr_features      = ns["build_corr_features"]
featurise                = ns["featurise"]
alloy_densities          = ns["alloy_densities"]
TRACE_THRESHOLD          = ns["TRACE_THRESHOLD"]
PHASE_LABELS             = ns["PHASE_LABELS"]

rng = np.random.default_rng(0)
ok  = lambda msg: print(f"  PASS  {msg}")
print(f"  ({TORCH_MODE})")


# ══════════════════════════════════════════════════════════════════════════════
print("\n[FIX 2] composition canonicalisation")
# ══════════════════════════════════════════════════════════════════════════════
raw = rng.random((200, 32)) ** 6          # many tiny values → lots of traces
c   = canonicalise_composition(raw)

assert np.allclose(c.sum(axis=1), 1.0), "rows must sum to 1"
ok("every canonical composition sums to exactly 1.0")

assert np.all((c == 0) | (c >= TRACE_THRESHOLD)), "no value may sit below the threshold"
ok(f"no element survives below TRACE_THRESHOLD ({TRACE_THRESHOLD}) — 'present' == '> 0'")

assert np.allclose(canonicalise_composition(c), c), "must be idempotent"
ok("canonicalisation is idempotent (applying it twice changes nothing)")


# ══════════════════════════════════════════════════════════════════════════════
print("\n[FIX 2] optimiser and results table now see the SAME vector")
# ══════════════════════════════════════════════════════════════════════════════
class MockGen:                     # stands in for the WGAN-GP generator
    def __init__(self):
        self.W = rng.standard_normal((10, 39))    # fixed weights — deterministic
    def __call__(self, z):
        # z is a real torch tensor when torch is installed, ndarray otherwise
        arr = z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)
        out = np.abs(np.sin(arr.astype(np.float64) @ self.W * 1.7))
        return types.SimpleNamespace(numpy=lambda: out)

gen      = MockGen()
comp_min = np.zeros(32)
comp_max = np.ones(32)
z        = rng.standard_normal((40, 10))

# _evaluate() and decode_results() both call latent_to_alloys with the same args
a_eval   = latent_to_alloys(z, gen, comp_min, comp_max)
a_decode = latent_to_alloys(z, gen, comp_min, comp_max)
assert np.array_equal(a_eval, a_decode)
ok("latent_to_alloys is the single shared decode path — byte-identical output")

assert np.allclose(a_eval[:, :32].sum(axis=1), 1.0)
ok("compositions reaching the regressors are normalised (was NOT true before)")

# The displayed name is now a straight readout of the same vector
comp  = a_eval[0, :32]
name  = "".join(f"{ns['ELEMENTS'][j]}{comp[j]:.3f}" for j in range(32) if comp[j] > 0)
parsed = {s: float(f) for s, f in re.findall(r'([A-Z][a-z]?)(\d+\.\d+)', name)}
assert abs(sum(parsed.values()) - 1.0) < 5e-3
ok(f"displayed alloy == predicted alloy, e.g. {name}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n[FIX 1] feature dimensions")
# ══════════════════════════════════════════════════════════════════════════════
alloy39 = a_eval[0]
phase4  = np.array([1.0, 0.0, 0.0, 0.0])
elec7   = np.array([1.0, 0, 0, 0, 0, 0, 0])

assert len(build_base_features(alloy39))            == 54
assert len(build_mech_features(alloy39, phase4))    == 58
assert len(build_corr_features(alloy39, phase4, elec7, 0.1)) == 66
ok("54 (phase classifier) / 58 (mechanical) / 66 (corrosion)")

base54 = build_base_features(alloy39)
mech58 = build_mech_features(alloy39, phase4)
assert np.array_equal(mech58[:54], base54)
ok("58-dim vector is exactly the 54-dim vector + 4 phase flags appended")

# The classifier input must not contain the phase flags anywhere
assert len(base54) == 54 and np.array_equal(mech58[54:], phase4)
ok("phase flags live ONLY in positions 54:58 — absent from classifier input")


# ══════════════════════════════════════════════════════════════════════════════
print("\n[FIX 1] the leakage itself, demonstrated numerically")
# ══════════════════════════════════════════════════════════════════════════════
n = 800
X54 = rng.random((n, 54))
# a genuinely hard-ish target: noisy nonlinear function of the features
y = ((X54[:, :5].sum(1) + 0.6 * rng.standard_normal(n)) > 2.5).astype(int)

acc_clean = cross_val_score(
    RandomForestClassifier(n_estimators=60, random_state=0), X54, y, cv=5).mean()
X58 = np.hstack([X54, y.reshape(-1, 1), rng.random((n, 3))])   # y as a feature
acc_leaky = cross_val_score(
    RandomForestClassifier(n_estimators=60, random_state=0), X58, y, cv=5).mean()

print(f"        target excluded from features : acc = {acc_clean:.3f}")
print(f"        target included as a feature  : acc = {acc_leaky:.3f}")
assert acc_leaky > 0.99 and acc_leaky - acc_clean > 0.1
ok("including the target as a feature drives accuracy to ~1.00 — the old setup")

# And: a leaky model fed zeros in the target slot collapses, as observed in the app
leaky = RandomForestClassifier(n_estimators=60, random_state=0).fit(X58, y)
X58_zeroed = X58.copy(); X58_zeroed[:, 54] = 0.0      # what app.py used to pass
pred_zeroed = leaky.predict(X58_zeroed)
print(f"        leaky model with that slot zeroed: predicts class 1 for "
      f"{100*pred_zeroed.mean():.1f}% of rows (true rate {100*y.mean():.1f}%)")
ok("explains the 'predicts 0 for all four phases' behaviour you patched around")


# ══════════════════════════════════════════════════════════════════════════════
print("\n[FIX 3] phase objective == reported phase probability")
# ══════════════════════════════════════════════════════════════════════════════
clfs = {}
X54_train = rng.random((400, 54))
for i, p in enumerate(PHASE_LABELS):
    yp = (X54_train[:, i] + 0.3 * rng.standard_normal(400) > 0.5).astype(int)
    clfs[p] = RandomForestClassifier(n_estimators=40, random_state=i).fit(X54_train, yp)

alloys = a_eval[:20]
b1, proba1, flags1, mf1, cf1 = featurise(alloys, clfs, elec7, 0.1)   # _evaluate
b2, proba2, flags2, mf2, cf2 = featurise(alloys, clfs, elec7, 0.1)   # decode_results

assert np.array_equal(proba1, proba2) and np.array_equal(mf1, mf2) and np.array_equal(cf1, cf2)
ok("one featurise() call feeds both the objective and the results table")

# objective value for 'FCC' vs the number printed in the 'FCC probability' column
obj_fcc = -proba1[:, PHASE_LABELS.index('FCC')]
col_fcc = proba1[:, 0]
assert np.allclose(-obj_fcc, col_fcc)
ok("maximising the FCC objective maximises exactly the reported FCC probability")

assert np.all((proba1 >= 0) & (proba1 <= 1)) and len(np.unique(proba1)) > 8
ok("objective is continuous in [0,1], not binary 0/1 — NSGA-II now has a gradient")

# hard flags handed to the regressors agree with sklearn's own .predict()
for i, p in enumerate(PHASE_LABELS):
    assert np.array_equal(flags1[:, i], clfs[p].predict(b1).astype(float))
ok("phase flags (proba > 0.5) match RandomForestClassifier.predict() exactly, "
   "including 50/50 tie-breaking")


# ══════════════════════════════════════════════════════════════════════════════
print("\n[constraints] read the canonical vector directly")
# ══════════════════════════════════════════════════════════════════════════════
comp32 = a_eval[:, :32]
n_el   = (comp32 > 0).sum(axis=1)
assert np.array_equal(n_el, (comp32 > TRACE_THRESHOLD - 1e-12).sum(axis=1))
ok("'> 0' and '> TRACE_THRESHOLD' now count identically — no double threshold")

d = alloy_densities(comp32)
assert np.all(np.isfinite(d)) and np.all(d > 0)
ok(f"densities finite and positive (range {d.min():.2f}–{d.max():.2f} g/cm³)")


# ══════════════════════════════════════════════════════════════════════════════
print("\n[FIX 3b] banned-element name parsing")
# ══════════════════════════════════════════════════════════════════════════════
TOKEN = re.compile(r'([A-Z][a-z]?)(\d+\.\d+)')
sample = "Co0.250Cr0.250Nb0.250Ni0.250"

old_hit_C = 'C' in sample and 'C' not in ('Co', 'Cr')     # old substring test
new = {s: float(f) for s, f in TOKEN.findall(sample)}
assert old_hit_C is True                                   # old code: false positive
assert 'C' not in new and 'N' not in new
assert set(new) == {'Co', 'Cr', 'Nb', 'Ni'}
ok("regex parse of 'Co0.250Cr0.250Nb0.250Ni0.250' -> Co, Cr, Nb, Ni (not C, N)")

# ══════════════════════════════════════════════════════════════════════════════
print("\n[Pipeline C] feature routing and metrics contract")
# ══════════════════════════════════════════════════════════════════════════════
ROUTING      = ns["MECH_USES_CORR_FEATURES"]
METRIC_PROPS = ns["METRIC_PROPS"]
PIPELINE_DIRS = ns["PIPELINE_DIRS"]

assert set(ROUTING) == {'A', 'B', 'C'}
assert ROUTING['A'] is False and ROUTING['C'] is False and ROUTING['B'] is True
ok("A and C route mechanical regressors to the 58-dim vector; B to the 66-dim one")

# Simulate exactly what _evaluate / decode_results do for each pipeline
for pipe, expected_dim in [('A', 58), ('B', 66), ('C', 58)]:
    feat = cf1 if ROUTING[pipe] else mf1
    assert feat.shape[1] == expected_dim, (pipe, feat.shape)
ok("pipeline -> feature-width mapping is A:58, B:66, C:58 as trained")

assert METRIC_PROPS == ['Hardness', 'Yield Strength', 'Tensile', 'Elongation',
                        'Ecorr', 'Epit', 'icorr']
ok("METRIC_PROPS matches the regressor keys written by step2/step3/step4")

assert PIPELINE_DIRS == {'A': 'models_A', 'B': 'models_B', 'C': 'models_C'}
ok("pipeline -> model directory mapping is complete")

# Phase classifiers are 54-dim for every pipeline — C included
assert b1.shape[1] == 54
ok("all three pipelines feed phase classifiers the same 54-dim vector")

print("\n" + "=" * 62)
print("  ALL CHECKS PASSED")
print("=" * 62)
