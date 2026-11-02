"""Seeded synthetic patients for verifying Framingham 2008. Run from repo root:
python reference/framingham/make_patients.py   (cholesterol in mg/dL, as CVrisk takes it)
"""

from pathlib import Path

import numpy as np
import pandas as pd

N = 20_000


def main():
    rng = np.random.default_rng(20261101)
    whole = np.tile(np.arange(30, 75), 100).astype(float)
    d = pd.DataFrame({
        "age": np.concatenate([whole, rng.uniform(30, 74.99, N - len(whole)).round(2)]),
        "sex": rng.choice(["male", "female"], N),
        "sbp": rng.uniform(90, 200, N).round(0),
        "bp_treated": rng.integers(0, 2, N),
        "smoker": rng.integers(0, 2, N),
        "diabetes": rng.integers(0, 2, N),
        "total_chol_mgdl": rng.uniform(100, 405, N).round(0),
        "hdl_chol_mgdl": rng.uniform(10, 100, N).round(0),
        "bmi": rng.uniform(15, 50, N).round(1),
    })
    d.to_csv(Path(__file__).parent / "fixtures" / "patients.csv", index_label="id")


if __name__ == "__main__":
    main()
