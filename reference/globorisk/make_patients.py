"""Seeded synthetic patients for verifying Globorisk (all countries, years, versions).
Run from repo root: python reference/globorisk/make_patients.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "fixtures"
N = 30_000


def main():
    OUT.mkdir(exist_ok=True)
    isos = pd.read_csv(Path(__file__).parent / "data" / "baseline_rates.csv",
                       usecols=["iso"], keep_default_na=False).iso.unique()
    rng = np.random.default_rng(20261101)
    whole = np.tile(np.arange(40, 81), 10).astype(float)  # every whole age 40-80
    age = np.concatenate([whole, np.minimum(rng.uniform(40, 81, N - len(whole)).round(2), 80.99)])
    d = pd.DataFrame({
        "age": age,
        "sex": rng.choice(["male", "female"], N),
        "sbp": rng.uniform(90, 200, N).round(1),
        "smoker": rng.integers(0, 2, N),
        "country": rng.choice(isos, N),
        "year": rng.integers(2000, 2021, N),
        "total_chol": rng.uniform(3.0, 9.0, N).round(2),
        "diabetes": rng.integers(0, 2, N),
        "bmi": rng.uniform(16, 45, N).round(1),
        # mostly 10-year (the published horizon), plus every other allowed horizon
        "horizon": np.where(rng.random(N) < 0.7, 10, rng.integers(1, 10, N)),
    })
    d.to_csv(OUT / "patients.csv", index_label="id")


if __name__ == "__main__":
    main()
