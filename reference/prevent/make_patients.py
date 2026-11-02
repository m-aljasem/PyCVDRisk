"""Seeded synthetic patients for verifying PREVENT (inputs in the package's units).
Ranges follow the PREVENT validity ranges as enforced by preventr.
Run from repo root: python reference/prevent/make_patients.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

N = 5_000


def main():
    rng = np.random.default_rng(20261101)
    whole = np.tile(np.arange(30, 80), 10).astype(float)  # every whole age 30-79
    d = pd.DataFrame({
        "age": np.concatenate([whole, rng.uniform(30, 79.99, N - len(whole)).round(2)]),
        "sex": rng.choice(["male", "female"], N),
        "sbp": rng.uniform(90, 180, N).round(0),
        "bp_treated": rng.integers(0, 2, N),
        "total_chol": rng.uniform(3.4, 8.2, N).round(2),   # mmol/L (~130-320 mg/dL)
        "hdl_chol": rng.uniform(0.6, 2.5, N).round(2),     # mmol/L (~20-100 mg/dL)
        "statin": rng.integers(0, 2, N),
        "diabetes": rng.integers(0, 2, N),
        "smoker": rng.integers(0, 2, N),
        "egfr": rng.uniform(15, 140, N).round(0),
        "bmi": rng.uniform(18.5, 39.9, N).round(1),
        "hba1c": rng.uniform(4.5, 15, N).round(1),          # %
        "uacr": np.exp(rng.uniform(np.log(0.1), np.log(25000), N)).round(1),  # mg/g
        "sdi": rng.integers(1, 11, N).astype(float),        # SDI decile
    })
    for col in ("uacr", "hba1c", "sdi"):  # ~10% missing: exercises the "missing" indicator terms
        d.loc[rng.random(N) < 0.1, col] = np.nan
    d.to_csv(Path(__file__).parent / "fixtures" / "patients.csv", index_label="id")


if __name__ == "__main__":
    main()
