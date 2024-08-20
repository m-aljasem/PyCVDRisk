"""Seeded synthetic patients for verifying the PCE. Run from repo root:
python reference/pce/make_patients.py   (cholesterol in mg/dL, as the R references take it)
"""

from pathlib import Path

import numpy as np
import pandas as pd

N = 20_000


def main():
    rng = np.random.default_rng(20261101)
    whole = np.tile(np.arange(40, 80), 100).astype(float)
    d = pd.DataFrame({
        "age": np.concatenate([whole, rng.uniform(40, 79.99, N - len(whole)).round(2)]),
        "sex": rng.choice(["male", "female"], N),
        "race": rng.choice(["white", "african_american"], N),
        # within CVrisk's accepted ranges (TC 130-320, HDL 20-100, SBP 90-200 mg/dL / mmHg)
        "total_chol_mgdl": rng.uniform(130, 320, N).round(0),
        "hdl_chol_mgdl": rng.uniform(20, 100, N).round(0),
        "sbp": rng.uniform(90, 200, N).round(0),
        "bp_treated": rng.integers(0, 2, N),
        "smoker": rng.integers(0, 2, N),
        "diabetes": rng.integers(0, 2, N),
    })
    d.to_csv(Path(__file__).parent / "fixtures" / "patients.csv", index_label="id")


if __name__ == "__main__":
    main()
