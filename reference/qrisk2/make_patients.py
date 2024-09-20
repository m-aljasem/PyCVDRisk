"""Seeded synthetic patients for verifying QRISK2-2014. Inputs run a little past ClinRisk's
validation ranges (and hit their exact bounds) so the NaN rules are checked too.
Run from repo root: python reference/qrisk2/make_patients.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.qrisk2 import RANGES
from cvd_risk.qrisk3 import DIABETES, ETHNICITY, SMOKING

N = 30_000
FLAGS = ("af", "rheumatoid_arthritis", "ckd", "bp_treated", "family_history")


def draw(rng, lo, hi, digits):
    pad = 0.05 * (hi - lo)
    x = rng.uniform(lo - pad, hi + pad, N).round(digits)
    x[rng.random(N) < 0.02] = lo
    x[rng.random(N) < 0.02] = hi
    return x


def main():
    rng = np.random.default_rng(20261101)
    whole = np.tile(np.arange(23, 87), 100).astype(float)  # every whole age 23-86, both sexes
    d = pd.DataFrame({
        "age": np.concatenate([whole, rng.uniform(25, 84.99, N - len(whole)).round(2)]),
        "sex": np.resize(["male", "female"], N),
        "ethnicity": rng.choice(ETHNICITY, N),
        "smoking": rng.choice(SMOKING, N),
        "diabetes": rng.choice(DIABETES, N),
        "bmi": draw(rng, *RANGES["bmi"], 1),
        "chol_hdl_ratio": draw(rng, *RANGES["chol_hdl_ratio"], 1),
        "sbp": draw(rng, *RANGES["sbp"], 0),
        "townsend": draw(rng, *RANGES["townsend"], 2),
        **{f: rng.integers(0, 2, N) for f in FLAGS},
    })
    d.to_csv(Path(__file__).parent / "fixtures" / "patients.csv", index_label="id")


if __name__ == "__main__":
    main()
