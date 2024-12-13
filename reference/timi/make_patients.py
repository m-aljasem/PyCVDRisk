"""Write reference/timi/fixtures/patients.csv in our form and RiskScorescvd's (five risk-factor
flags, previous PCI/CABG for known stenosis, hs-troponin I against limits 34/16 ng/L for markers).
Run from repo root."""

from pathlib import Path

import numpy as np

FX = Path(__file__).parent / "fixtures"
FLAGS = ("diabetes", "smoker", "hypertension", "hyperlipidaemia", "family_history")


def main(n=5000, seed=2000):
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], n)
    age = np.where(rng.random(n) < 0.3, rng.choice([64, 65], n), rng.integers(25, 96, n))
    count = rng.integers(0, 6, n)
    pci, cabg = (rng.random(n) < 0.2).astype(int), (rng.random(n) < 0.1).astype(int)
    url = np.where(sex == "male", 34, 16)
    hstni = np.maximum(0, np.where(rng.random(n) < 0.3, url + rng.choice([-1, 0], n), rng.integers(0, 80, n)))
    cols = {"id": np.arange(n), "sex": sex, "age": age, "risk_factors": count,
            **{f: (count > i).astype(int) for i, f in enumerate(FLAGS)},
            "previous_pci": pci, "previous_cabg": cabg, "coronary_stenosis": pci | cabg,
            "st_deviation": (rng.random(n) < 0.3).astype(int), "episodes": rng.integers(0, 5, n),
            "aspirin": (rng.random(n) < 0.5).astype(int), "hstni": hstni,
            "elevated_markers": (hstni >= url).astype(int)}
    FX.mkdir(exist_ok=True)
    with (FX / "patients.csv").open("w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(n):
            f.write(",".join(str(cols[c][i]) for c in cols) + "\n")
    print(f"wrote {n} patients")


if __name__ == "__main__":
    main()
