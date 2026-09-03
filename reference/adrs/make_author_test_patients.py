"""Author test set for the ADRS (run from the repo root: PYTHONPATH=src python3 reference/adrs/make_author_test_patients.py).

fixtures/author_test_patients.csv: 20 patients to send to the authors (inputs only, seed 2024), 10 women and 10 men
aged 18-79, four in each ACC/AHA 2017 blood-pressure stage of the CAD model (crisis included), the full ranges of
vitamin D, eGFR and TC/HDL ratio. Hypertension = on treatment or stage 2 / crisis BP.
fixtures/author_test_expected.csv: our 10-year risks (%) for the same ids, kept back for the comparison, with the
alternatives the authors' answer has to choose between (REPORT.md section 5):
  *_gfr113: ASCVD and CAD with the GFR mean 113.12 of the 2026 preprint instead of 111.4;
  stroke_noint: stroke with the Data S1 second assignment (no hypertension x sex term, mean 6.37 kept).
"""

import importlib
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from cvd_risk.adrs import adrs

mod = importlib.import_module("cvd_risk.adrs.adrs")  # the package name is shadowed by the function

FIX = Path(__file__).parent / "fixtures"
N, SEED = 20, 2024
# (SBP range, DBP range) per stage, inclusive; stage drawn as id order cycles 0..4
STAGES = [((100, 119), (60, 79)), ((120, 129), (60, 79)), ((130, 139), (80, 89)), ((140, 179), (90, 109)),
          ((181, 200), (100, 125))]


def main():
    rng = np.random.default_rng(SEED)
    st = np.arange(N) % 5
    sbp = [int(rng.integers(STAGES[s][0][0], STAGES[s][0][1] + 1)) for s in st]
    dbp = [int(rng.integers(STAGES[s][1][0], STAGES[s][1][1] + 1)) for s in st]
    tc = np.round(rng.uniform(3.5, 7.5, N), 1)
    hdl = np.round(rng.uniform(0.8, 1.8, N), 2)
    treated = (rng.random(N) < 0.35).astype(int)
    p = pd.DataFrame(dict(
        age=rng.integers(18, 80, N), sex=np.tile(["female", "male"], N // 2), smoker=(rng.random(N) < 0.35).astype(int),
        sbp=sbp, dbp=dbp, total_chol_mmol=tc, hdl_chol_mmol=hdl, tc_hdl=np.round(tc / hdl, 2),
        diabetes=(rng.random(N) < 0.4).astype(int), bp_treated=treated,
        hypertension=((treated == 1) | (st >= 3)).astype(int),
        vitamin_d=rng.integers(8, 61, N), egfr=rng.integers(60, 141, N)))
    p.insert(0, "id", [f"AD{i:02d}" for i in range(1, N + 1)])
    p.to_csv(FIX / "author_test_patients.csv", index=False)

    args = [p[c] for c in ("age", "sex", "smoker", "sbp", "dbp", "tc_hdl", "diabetes", "hypertension", "bp_treated",
                           "vitamin_d", "egfr")]
    e = p[["id"]].copy()
    for o in ("ascvd", "cad", "stroke"):
        e[f"{o}_10y_pct"] = np.round(100 * adrs(*args, outcome=o), 4)
    with mock.patch.object(mod, "GFR_MEAN", 113.12):
        for o in ("ascvd", "cad"):
            e[f"{o}_10y_pct_gfr113"] = np.round(100 * adrs(*args, outcome=o), 4)
    with mock.patch.dict(mod.STROKE, hypertension_x_male=0.0):
        e["stroke_10y_pct_noint"] = np.round(100 * adrs(*args, outcome="stroke"), 4)
    assert e.notna().all().all() and set(st) == set(range(5)) and p.sex.value_counts().eq(10).all()
    e.to_csv(FIX / "author_test_expected.csv", index=False)
    d = lambda a, b: (e[b] / e[a] - 1).abs()  # noqa: E731
    print(e.to_string(index=False))
    print(f"relative change, GFR mean 113.12: ascvd max {d('ascvd_10y_pct', 'ascvd_10y_pct_gfr113').max():.3f}, "
          f"cad max {d('cad_10y_pct', 'cad_10y_pct_gfr113').max():.3f}; stroke without term: "
          f"max {d('stroke_10y_pct', 'stroke_10y_pct_noint').max():.3f}, "
          f"n changed {(d('stroke_10y_pct', 'stroke_10y_pct_noint') > 1e-9).sum()}")


if __name__ == "__main__":
    main()
