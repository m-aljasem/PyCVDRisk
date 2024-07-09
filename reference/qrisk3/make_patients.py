"""Seeded synthetic patients for verifying QRISK3. Run from repo root:
python reference/qrisk3/make_patients.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.qrisk3 import DIABETES, ETHNICITY, SMOKING

N = 30_000
FLAGS = ("af", "atypical_antipsychotics", "corticosteroids", "erectile_dysfunction", "migraine",
         "rheumatoid_arthritis", "ckd", "severe_mental_illness", "sle", "bp_treated", "family_history")


def main():
    rng = np.random.default_rng(20261101)
    whole = np.tile(np.arange(25, 85), 100).astype(float)  # every whole age, both sexes
    d = pd.DataFrame({
        "age": np.concatenate([whole, rng.uniform(25, 84.99, N - len(whole)).round(2)]),
        "sex": np.resize(["male", "female"], N),
        "ethnicity": rng.choice(ETHNICITY, N),
        "smoking": rng.choice(SMOKING, N),
        "diabetes": rng.choice(DIABETES, N),
        "bmi": rng.uniform(15, 50, N).round(1),
        "chol_hdl_ratio": rng.uniform(1, 12, N).round(1),
        "sbp": rng.uniform(70, 210, N).round(0),
        "sbp_sd": rng.uniform(0, 30, N).round(1),
        "townsend": rng.uniform(-7, 11, N).round(2),
        **{f: rng.integers(0, 2, N) for f in FLAGS},
    })
    d.to_csv(Path(__file__).parent / "fixtures" / "patients.csv", index_label="id")


if __name__ == "__main__":
    main()
