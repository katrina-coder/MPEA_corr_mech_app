"""
step0_harmonise_processing.py  —  Harmonise Processing_corr free-text column
─────────────────────────────────────────────────────────────────────────────
The corrosion rows (619 rows) in MPEAs_Mech_Corr_DB_updated.xlsx have their
processing route recorded as a free-text column 'Processing_corr' with 193
unique strings (e.g. "arc melting+annealed at 1473k", "laser cladding 3 mm/s",
"milled for 5h", "vacuum arc melting").

The mechanical rows (1704 rows) encode the same information as 7 binary
one-hot columns (process_1 .. process_7), which are the features used by
the ML models.

This script maps the corrosion free-text to the same 7 one-hot columns
using priority-ordered keyword matching — a deterministic, domain-guided
rule set validated by manual inspection of the most frequent assignments.

The 7 canonical categories are:
  process_1 : As-cast / arc-melted          (default)
  process_2 : Arc-melted + artificial aging
  process_3 : Arc-melted + annealing / heat treatment
  process_4 : Powder metallurgy / sintering
  process_5 : Novel / additive (laser, EBM, SLM, LPBF, plasma spray, PVD)
  process_6 : Wrought processing (rolling, forging, hot pressing)
  process_7 : Cryogenic treatments

Priority order matters: more specific rules fire first.
For example "arc melting+annealed at 1473k" contains both 'arc' (→ process_1)
and 'anneal' (→ process_3). Annealing fires first because it describes the
final microstructural state more precisely than the melting step.

Usage:
    python step0_harmonise_processing.py

Input / Output:
    MPEAs_Mech_Corr_DB_updated.xlsx  (overwritten in place)
"""

import pandas as pd
import numpy as np

DB_PATH = 'MPEAs_Mech_Corr_DB_updated.xlsx'

# ── Keyword mapping function ──────────────────────────────────────────────────
def map_processing(val):
    """Map a free-text processing string to a canonical category index 1..7."""
    if val is None or str(val).strip() in ['0', '', 'missing', 'nan']:
        return 1  # unknown → default to as-cast (most common)
    v = str(val).strip().lower()

    # process_7: cryogenic treatments
    if any(k in v for k in ['cryo']):
        return 7

    # process_6: wrought (rolling, forging, cold work, hot pressing)
    if any(k in v for k in ['roll', 'forg', 'wrough', 'cold work',
                              'hip', 'vacuum hot', 'vhps']):
        return 6

    # process_4: powder metallurgy (sintering, ball milling, powder)
    if any(k in v for k in ['powder', 'sinter', 'ball mill',
                              'milled for', 'milling']):
        return 4

    # process_5: novel / additive / laser / plasma / PVD / EBM / SLM
    if any(k in v for k in ['laser', 'slm', 'lpbf', 'ebm', 'sebm', 'lmd',
                              'plasma', 'pvd', 'spray', 'cladding', 'clad',
                              'pta', 'gta', 'electroly', 'electrochem',
                              'passiv', 'directional', 'lsp']):
        return 5

    # process_3: annealing / heat treatment / aging at temperature
    # Note: fires before process_1 so "arc melting + anneal" → annealing
    if any(k in v for k in ['anneal', 'homo', 'aging', 'heat treat', 'aged',
                              '\u00b0c', '\u25e6c', 're-melt', 'remelted',
                              'remelting', 'equilibrat']):
        return 3

    # process_2: artificial aging specifically (precipitation hardening)
    if any(k in v for k in ['artificial aging', 'age harden',
                              'precipitation harden']):
        return 2

    # process_1: as-cast / arc-melted / induction / furnace / vacuum arc
    if any(k in v for k in ['cast', 'arc', 'vacuum', 'induction', 'melt',
                              'melted', 'melting', 'furnace', 'copper mold',
                              'suction', 'as-', 'solidif', 'lm', 'am',
                              'amf', 'ac']):
        return 1

    return 1  # default: as-cast


# ── Load database ─────────────────────────────────────────────────────────────
print(f"Loading {DB_PATH}...")
df = pd.read_excel(DB_PATH)
print(f"  {len(df)} rows × {df.shape[1]} columns")

corr_mask = df['OG property'] == 'corrosion'
print(f"\n  Mechanical rows : {(df['OG property']=='mechanical').sum()}")
print(f"  Corrosion rows  : {corr_mask.sum()}")

# ── Show before state ─────────────────────────────────────────────────────────
proc_cols = ['process_1','process_2','process_3','process_4',
             'process_5','process_6','process_7']
print(f"\n=== process_1..7 for corrosion rows BEFORE mapping ===")
print(df.loc[corr_mask, proc_cols].sum().to_string())

# ── Apply mapping ─────────────────────────────────────────────────────────────
print(f"\nApplying keyword-based mapping to 'Processing_corr' column...")
mapped = df.loc[corr_mask, 'Processing_corr'].apply(map_processing)

# Show distribution of mapped categories
print(f"\n  Mapping distribution:")
PROC_NAMES = {
    1: 'As-cast / arc-melted',
    2: 'Arc-melted + aging',
    3: 'Arc-melted + annealing',
    4: 'Powder metallurgy',
    5: 'Novel / additive',
    6: 'Wrought processing',
    7: 'Cryogenic',
}
for k, count in mapped.value_counts().sort_index().items():
    print(f"    process_{k} ({PROC_NAMES[k]}): {count} rows")

# Write into process_1..7 one-hot columns for corrosion rows
for proc_idx in range(1, 8):
    df.loc[corr_mask, f'process_{proc_idx}'] = (
        (mapped == proc_idx).astype(int).values
    )

# ── Validate ──────────────────────────────────────────────────────────────────
print(f"\n=== process_1..7 for corrosion rows AFTER mapping ===")
print(df.loc[corr_mask, proc_cols].sum().to_string())

# Every corrosion row should now have exactly one process column = 1
row_sums = df.loc[corr_mask, proc_cols].sum(axis=1)
assert (row_sums == 1).all(), f"ERROR: some rows have != 1 process flag: {row_sums.value_counts()}"
print(f"\n✓ All {corr_mask.sum()} corrosion rows have exactly one process flag")

# ── Show sample mappings ──────────────────────────────────────────────────────
print(f"\n=== Sample mappings (first 20 non-zero Processing_corr values) ===")
sample = df.loc[corr_mask & (df['Processing_corr'] != 0)].head(20)
for _, row in sample.iterrows():
    assigned = int(mapped.loc[row.name])
    print(f"  '{row['Processing_corr']}'")
    print(f"      → process_{assigned} ({PROC_NAMES[assigned]})")

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving updated database to {DB_PATH}...")
df.to_excel(DB_PATH, index=False)
print(f"✓ Saved — process_1..7 now filled for all {corr_mask.sum()} corrosion rows")
print(f"\n✅ Harmonisation complete — ready for step2_retrain_models_A.py and step3_retrain_models_B.py")
