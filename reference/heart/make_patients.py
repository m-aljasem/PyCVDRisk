"""Write reference/heart/fixtures/patients.csv: random HEART inputs, in both our form (grades,
risk-factor count, troponin ratio) and RiskScorescvd's (symptom count, ECG flags, five risk-factor
flags, hs-troponin I in ng/L with its sex-specific limits 34/16). Run from repo root."""

from pathlib import Path

import numpy as np

FX = Path(__file__).parent / "fixtures"
URL = {"male": 34, "female": 16}  # RiskScorescvd's hs-troponin I limits (ng/L)
FLAGS = ("diabetes", "smoker", "hypertension", "hyperlipidaemia", "family_history")


def main(n=5000, seed=2008):
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], n)
    history = rng.integers(0, 3, n)
    symptoms = np.select([history == 0, history == 1], [rng.integers(0, 2, n), rng.integers(2, 5, n)],
                         rng.integers(5, 7, n))
    ecg = rng.integers(0, 3, n)
    age = np.where(rng.random(n) < 0.2, rng.choice([44, 45, 64, 65], n), rng.integers(18, 96, n))
    count = rng.integers(0, 6, n)
    ath = (rng.random(n) < 0.2).astype(int)
    url = np.array([URL[s] for s in sex])
    hstni = np.where(rng.random(n) < 0.3, rng.choice([-1, 0, 1, 3, 3], n) * url + rng.choice([-1, 0, 1], n),
                     rng.integers(0, 400, n))
    hstni = np.maximum(hstni, 0)
    cols = {"id": np.arange(n), "sex": sex, "history": history, "symptoms": symptoms, "ecg": ecg,
            "ecg_normal": (ecg == 0).astype(int), "abn_repolarisation": (ecg == 1).astype(int),
            "ecg_st_depression": (ecg == 2).astype(int), "age": age, "risk_factors": count,
            **{f: (count > i).astype(int) for i, f in enumerate(FLAGS)},
            "atherosclerosis": ath, "hstni": hstni, "troponin_ratio": [repr(float(h / u)) for h, u in zip(hstni, url)]}
    FX.mkdir(exist_ok=True)
    with (FX / "patients.csv").open("w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(n):
            f.write(",".join(str(cols[c][i]) for c in cols) + "\n")
    print(f"wrote {n} patients")


if __name__ == "__main__":
    main()
