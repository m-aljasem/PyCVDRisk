"""Author test set for the Suita score 2014 (run from the repo root:
PYTHONPATH=src python3 reference/suita/make_author_test_patients.py).

fixtures/author_test_patients.csv: 20 patients to send to the authors (inputs only). S01-S06 target the TC-sheet
entries in question (HDL >= 60, CKD stage 4-5, age 35-45, HDL 35, TC 280); S07-S20 are seeded (2014), aged 35-79,
integer values in mg/dL, LDL by Friedewald from a drawn triglyceride.
fixtures/author_test_expected.csv: our points and printed 10-year CHD probability on both sheets, kept back.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.suita import suita, suita_risk

FIX = Path(__file__).parent / "fixtures"
COLS = ["id", "age", "sex", "smoker", "diabetes", "sbp", "dbp", "total_chol", "ldl_chol", "hdl_chol", "egfr"]
TARGETED = [  # S01 HDL >= 60; S02 CKD 4-5; S03 age 40 (TC band 36-45); S04 age 35; S05 HDL 35; S06 TC 280
    ("S01", 58, "female", 0, 0, 128, 78, 230, 140, 72, 75),
    ("S02", 66, "male", 1, 1, 150, 88, 210, 128, 44, 24),
    ("S03", 40, "male", 1, 0, 136, 86, 245, 165, 48, 92),
    ("S04", 35, "male", 0, 0, 118, 76, 190, 115, 55, 105),
    ("S05", 52, "male", 1, 0, 142, 92, 220, 150, 35, 68),
    ("S06", 62, "female", 0, 1, 165, 95, 280, 190, 58, 52),
]


def main():
    rng = np.random.default_rng(2014)
    rows = list(TARGETED)
    for i in range(7, 21):
        age = int(rng.integers(35, 80))
        sbp = int(rng.integers(100, 181)); dbp = int(min(sbp - 25, rng.integers(60, 106)))
        tc = int(rng.integers(140, 301)); hdl = int(rng.integers(30, 91)); tg = int(rng.integers(50, 251))
        ldl = max(40, tc - hdl - round(tg / 5))
        egfr = int(rng.choice([rng.integers(15, 30), rng.integers(30, 60), rng.integers(60, 120)], p=[.1, .25, .65]))
        rows.append((f"S{i:02d}", age, rng.choice(["male", "female"]), int(rng.random() < .3), int(rng.random() < .15),
                     sbp, dbp, tc, ldl, hdl, egfr))
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(FIX / "author_test_patients.csv", index=False)
    a = [df[c].to_numpy() for c in ("age", "sex", "smoker", "diabetes", "sbp", "dbp", "hdl_chol", "egfr")]
    tc, ldl = suita(*a, tc=df.total_chol.to_numpy()), suita(*a, ldl=df.ldl_chol.to_numpy())
    pd.DataFrame({"id": df.id, "tc_points": tc, "tc_risk_pct": suita_risk(tc, "tc"),
                  "ldl_points": ldl, "ldl_risk_pct": suita_risk(ldl, "ldl")}).to_csv(
        FIX / "author_test_expected.csv", index=False, float_format="%.0f")


if __name__ == "__main__":
    main()
