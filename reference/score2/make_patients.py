"""Generate the synthetic patients used to verify the SCORE2 family.

Seeded and deterministic. Covers every sex x region x smoker x diabetes
combination, every integer age in each variant's range (including both ends),
and continuous risk factors over and beyond the usual clinical range.
Run: python make_patients.py  (writes fixtures/<variant>_patients.csv)
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "fixtures"
REGIONS = ["low", "moderate", "high", "very_high"]
N = 20_000


def base(rng, ages):
    n = len(ages)
    return pd.DataFrame({
        "age": ages,
        "sex": rng.choice(["male", "female"], n),
        "smoker": rng.integers(0, 2, n),
        "sbp": rng.uniform(90, 200, n).round(1),
        "total_chol": rng.uniform(3.0, 9.0, n).round(2),
        "hdl_chol": rng.uniform(0.7, 2.5, n).round(2),
        "diabetes": rng.integers(0, 2, n),
        "region": rng.choice(REGIONS, n),
    })


def ages(rng, lo, hi):
    whole = np.tile(np.arange(lo, hi + 1), 4).astype(float)  # every integer age, both ends
    return np.concatenate([whole, rng.uniform(lo, hi, N - len(whole)).round(1)])


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261101)
    base(rng, ages(rng, 40, 69)).to_csv(OUT / "score2_patients.csv", index_label="id")
    base(rng, ages(rng, 70, 89)).to_csv(OUT / "score2_op_patients.csv", index_label="id")
    base(rng, ages(rng, 40, 69)).to_csv(OUT / "score2_asia_pacific_patients.csv", index_label="id")
    d = base(rng, ages(rng, 40, 69))
    d["age_at_diabetes_diagnosis"] = np.where(
        d.diabetes == 1, rng.uniform(18, d.age).round(0), np.nan)
    d["hba1c"] = rng.uniform(20, 120, len(d)).round(1)
    d["egfr"] = rng.uniform(15, 150, len(d)).round(0)
    d.to_csv(OUT / "score2_diabetes_patients.csv", index_label="id")


if __name__ == "__main__":
    main()
