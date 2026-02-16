"""Seeded synthetic patients for the Jcvrisk comparison: fixtures/patients.csv (run from the repo root).

Integer blood pressure and lipids so that category edges (e.g. SBP 139, DBP 84 and 89) occur often.
"""

import csv
from pathlib import Path

import numpy as np

N, SEED = 3000, 0
OUT = Path(__file__).parent / "fixtures/patients.csv"


def main():
    r = np.random.default_rng(SEED)
    hdl = r.integers(25, 100, N)
    cols = dict(age=r.integers(40, 90, N), male=r.integers(0, 2, N), sbp=r.integers(95, 200, N),
                dbp=r.integers(55, 120, N), ht_medication=r.integers(0, 2, N), t2dm=r.integers(0, 2, N),
                hdl=hdl, tc=hdl + r.integers(60, 230, N), ldl=r.integers(60, 220, N),
                urineprotein=r.integers(0, 2, N), smoking=r.integers(0, 2, N), exercise=r.integers(0, 2, N),
                af=r.integers(0, 2, N), lvh=r.integers(0, 2, N), bmi=np.round(r.uniform(16, 34, N), 1),
                egfr=r.integers(30, 120, N))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", *cols])
        w.writerows([i, *(c[i] for c in cols.values())] for i in range(N))


if __name__ == "__main__":
    main()
