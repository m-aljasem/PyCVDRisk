"""Pack the extracted Globorisk tables into src/cvd_risk/globorisk/data.npz.

Input: reference/globorisk/data/*.csv (made by extract_data.R from the authors'
package). Output arrays:
  isos                      country ISO3 codes, sorted
  rates[outcome, iso, sex, year-2000, age-40]   baseline hazard; outcome 0 = fatal +
                                                non-fatal CVD, 1 = fatal CVD; NaN = absent
  means[iso, sex, agec-1, var]  population means; var = sbp/10, tc, dm, smk, bmi/5
Run from repo root: python reference/globorisk/build_package_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("reference/globorisk/data")
OUT = Path("src/cvd_risk/globorisk/data.npz")
YEARS, AGES = np.arange(2000, 2030), np.arange(40, 90)
MEANS = ["mean_sbp", "mean_tc", "mean_dm", "mean_smk", "mean_bmi"]


def main():
    r = pd.read_csv(SRC / "baseline_rates.csv", keep_default_na=False, na_values=[""])
    m = pd.read_csv(SRC / "mean_risk_factors.csv", keep_default_na=False, na_values=[""])
    isos = np.array(sorted(set(r.iso) | set(m.iso)))
    idx = {c: i for i, c in enumerate(isos)}

    rates = np.full((2, len(isos), 2, len(YEARS), len(AGES)), np.nan)
    rates[(r.type == "F").to_numpy(int), r.iso.map(idx), r.sex, r.year - 2000, r.age - 40] = r.rate
    means = np.full((len(isos), 2, 9, len(MEANS)), np.nan)
    means[m.iso.map(idx), m.sex, m.agec - 1] = m[MEANS].to_numpy()

    np.savez_compressed(OUT, isos=isos, rates=rates, means=means)
    print(OUT, f"{OUT.stat().st_size / 1e6:.1f} MB", len(isos), "countries")


if __name__ == "__main__":
    main()
