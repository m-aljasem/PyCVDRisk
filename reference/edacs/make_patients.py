"""Write reference/edacs/fixtures/patients.csv. RiskScorescvd has no known-CAD input, so known_cad is 0
here; the CAD rule is covered by the rule table in report.py. Run from repo root."""

from pathlib import Path

import numpy as np

FX = Path(__file__).parent / "fixtures"
FLAGS = ("diabetes", "smoker", "hypertension", "hyperlipidaemia", "family_history")
EDGES = [18, 45, 46, 50, 51, 55, 56, 60, 61, 65, 66, 70, 71, 75, 76, 80, 81, 85, 86]


def main(n=5000, seed=2014):
    rng = np.random.default_rng(seed)
    age = np.where(rng.random(n) < 0.4, rng.choice(EDGES, n), rng.integers(18, 100, n))
    count = rng.integers(0, 6, n)
    flag = lambda p: (rng.random(n) < p).astype(int)
    cols = {"id": np.arange(n), "sex": rng.choice(["male", "female"], n), "age": age, "known_cad": np.zeros(n, int),
            "risk_factors": count, **{f: (count > i).astype(int) for i, f in enumerate(FLAGS)},
            "diaphoresis": flag(0.5), "radiates": flag(0.35), "pleuritic": flag(0.2), "palpation": flag(0.1)}
    FX.mkdir(exist_ok=True)
    with (FX / "patients.csv").open("w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(n):
            f.write(",".join(str(cols[c][i]) for c in cols) + "\n")
    print(f"wrote {n} patients")


if __name__ == "__main__":
    main()
