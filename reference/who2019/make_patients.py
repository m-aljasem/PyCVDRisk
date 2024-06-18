"""Seeded synthetic patients for verifying the 2019 WHO models.
Run from repo root: python reference/who2019/make_patients.py
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

N = 30_000
SRC = Path(__file__).parents[2] / "src" / "cvd_risk" / "who2019"


def main():
    rng = np.random.default_rng(20261101)
    with (SRC / "countries.csv").open() as f:
        countries = [r["iso3"] for r in csv.DictReader(f)]
    whole = np.tile(np.arange(40, 81), 50).astype(float)  # every whole age 40-80
    d = pd.DataFrame({
        "age": np.concatenate([whole, rng.uniform(40, 80.99, N - len(whole)).round(2)]),
        "sex": rng.choice(["male", "female"], N),
        "sbp": rng.uniform(90, 200, N).round(0),
        "smoker": rng.integers(0, 2, N),
        "total_chol": rng.uniform(2.5, 10, N).round(2),
        "diabetes": rng.integers(0, 2, N),
        "bmi": rng.uniform(16, 45, N).round(1),
        "country": rng.choice(countries, N),  # every country, so every region too
        "year": np.where(rng.random(N) < 0.5, 2017, rng.uniform(1990, 2017, N).round(2)),
    })
    d.to_csv(Path(__file__).parent / "fixtures" / "patients.csv", index_label="id")


if __name__ == "__main__":
    main()
