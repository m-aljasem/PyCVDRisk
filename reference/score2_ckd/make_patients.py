"""Seeded synthetic patients for verifying SCORE2-CKD. Run from repo root:
python reference/score2_ckd/make_patients.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

N = 20_000


def main():
    rng = np.random.default_rng(20261101)
    whole = np.tile(np.arange(40, 90), 40).astype(float)  # every whole age 40-89 (SCORE2 and -OP)
    d = pd.DataFrame({
        "age": np.concatenate([whole, rng.uniform(40, 89.99, N - len(whole)).round(2)]),
        "sex": rng.choice(["male", "female"], N),
        "smoker": rng.integers(0, 2, N),
        "sbp": rng.uniform(90, 200, N).round(0),
        "total_chol": rng.uniform(3, 9, N).round(1),
        "hdl_chol": rng.uniform(0.7, 2.5, N).round(2),
        "diabetes": rng.integers(0, 2, N),
        "egfr": rng.uniform(15, 130, N).round(0),   # covers all three spline segments
        "acr": np.exp(rng.uniform(np.log(1), np.log(3000), N)).round(1),
        "dipstick": rng.choice(["negative", "trace", "1+", "2+", "3+"], N),
        "region": rng.choice(["low", "moderate", "high", "very_high"], N),
    })
    d.to_csv(Path(__file__).parent / "fixtures" / "patients.csv", index_label="id")


if __name__ == "__main__":
    main()
