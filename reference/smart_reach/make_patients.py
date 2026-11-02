"""Seeded synthetic patients for verifying SMART-REACH against the authors' calculator.
Run from repo root: python reference/smart_reach/make_patients.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.smart_reach import REGION_NAMES

N = 600


def main():
    rng = np.random.default_rng(20261101)
    loc = rng.integers(0, 2, (N, 3))
    loc[loc.sum(1) == 0, rng.integers(0, 3)] = 1  # at least one disease location
    d = pd.DataFrame({
        "age": np.concatenate([np.arange(45, 90), rng.integers(45, 90, N - 45)]),
        "sex": rng.choice(["male", "female"], N),
        "smoker": rng.integers(0, 2, N), "diabetes": rng.integers(0, 2, N),
        "sbp": rng.uniform(90, 200, N).round(0), "total_chol": rng.uniform(2.5, 8.5, N).round(1),
        "creatinine": rng.uniform(40, 250, N).round(0),
        "cad": loc[:, 0], "cerebrovascular": loc[:, 1], "pad": loc[:, 2],
        "af": rng.integers(0, 2, N), "heart_failure": rng.integers(0, 2, N),
        "region": rng.choice(REGION_NAMES, N),
    })
    d.to_csv(Path(__file__).parent / "fixtures" / "patients.csv", index_label="id")


if __name__ == "__main__":
    main()
