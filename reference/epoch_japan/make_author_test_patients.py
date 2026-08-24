"""Author test set for EPOCH-JAPAN (run from the repo root: PYTHONPATH=src python3 reference/epoch_japan/make_author_test_patients.py).

fixtures/author_test_patients.csv: 20 patients to send to the authors (inputs only): the two worked examples of
Supplementary Table 2, then 18 seeded patients aged 40-79 (seed 2021), integer SBP and lipids in mg/dL.
fixtures/author_test_expected.csv: our 10-year risks (%) for the same ids, kept back for the comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.epoch_japan import epoch_japan

FIX = Path(__file__).parent / "fixtures"
N, SEED = 18, 2021


def main():
    rng = np.random.default_rng(SEED)
    p = pd.DataFrame(dict(
        age=rng.integers(40, 80, N), sex=rng.choice(["male", "female"], N), sbp=rng.integers(100, 191, N),
        total_chol=rng.integers(140, 301, N), hdl_chol=rng.integers(30, 96, N), smoker=rng.integers(0, 2, N),
        diabetes=(rng.random(N) < 0.3).astype(int), proteinuria=(rng.random(N) < 0.25).astype(int)))
    ex = pd.DataFrame([  # Supplementary Table 2: example 1 (CHD), example 2 (stroke); lipids of example 2 not given
        dict(age=70, sex="female", sbp=135, total_chol=185, hdl_chol=58, smoker=0, diabetes=1, proteinuria=0),
        dict(age=70, sex="male", sbp=140, total_chol=200, hdl_chol=50, smoker=1, diabetes=0, proteinuria=0)])
    p = pd.concat([ex, p], ignore_index=True)
    p.insert(0, "id", [f"EJ{i:02d}" for i in range(1, len(p) + 1)])
    p.to_csv(FIX / "author_test_patients.csv", index=False)
    e = p[["id"]].copy()
    for o in ("chd", "stroke", "cvd"):
        e[f"{o}_10y_pct"] = np.round(100 * epoch_japan(p.age, p.sex, p.sbp, p.smoker, p.diabetes, p.proteinuria,
                                                         total_chol=p.total_chol, hdl_chol=p.hdl_chol, outcome=o), 4)
    assert e.notna().all().all() and e.loc[0, "chd_10y_pct"].round(2) == 1.13 and e.loc[1, "stroke_10y_pct"].round(2) == 2.82
    e.to_csv(FIX / "author_test_expected.csv", index=False)


if __name__ == "__main__":
    main()
