"""Build reference/globorisk/REPORT.md. Run from repo root: python reference/globorisk/report.py"""

from __future__ import annotations

import copy
import datetime as dt
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.globorisk import coefficients as C

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "tests" / "globorisk"))
from test_globorisk import run  # noqa: E402  (same call path as the test suite)

mod = importlib.import_module("cvd_risk.globorisk.globorisk")
TABLES = {"lab": "LAB", "office": "OFFICE", "fatal": "FATAL"}
REL_NUDGE = 1e-6  # stored values carry ~17 digits; the papers print 3-4


def main():
    p = pd.read_csv(HERE / "fixtures" / "patients.csv", keep_default_na=False, na_values=[""])
    r = pd.read_csv(HERE / "fixtures" / "reference.csv", keep_default_na=False, na_values=[""])
    rows, muts = [], []
    for version, table in TABLES.items():
        ref = r[version].to_numpy()
        t0 = time.perf_counter()
        got = run(p, version)
        secs = time.perf_counter() - t0
        ok = ~np.isnan(ref)
        same_nan = bool((np.isnan(got) == np.isnan(ref)).all())
        rel = np.abs(got[ok] - ref[ok]) / ref[ok]
        rows.append((version, len(p), int((~ok).sum()), same_nan, rel.max(), secs))
        original = getattr(C, table)
        for key, val in original.items():
            mutated = copy.deepcopy(original)
            mutated[key] = val * (1 + REL_NUDGE)
            setattr(C, table, mutated)
            try:
                caught = not np.allclose(run(p, version)[ok], ref[ok], rtol=1e-12, atol=1e-15)
            finally:
                setattr(C, table, original)
            muts.append((version, key, caught))

    m = pd.DataFrame(muts, columns=["version", "coef", "caught"])
    lines = [
        "# Globorisk — verification report", "",
        f"Generated {dt.date.today()} by `reference/globorisk/report.py`.", "",
        "## 1. Agreement with the reference", "",
        "Reference: the authors' own R package `globorisk` (evidence level A).", "",
        "```", (HERE / "fixtures" / "VERSION").read_text().strip(), "```", "",
        "30,000 seeded synthetic patients: 183 countries, baseline years 2000-2020, every",
        "whole age 40-80 plus fractional ages, horizons 1-10 years (70% at 10).", "",
        "| Version | Patients | NaN (both) | NaN pattern identical | Max rel diff | Time (s) |",
        "|---|---|---|---|---|---|",
        *[f"| {v} | {n:,} | {na} | {'yes' if s else '**NO**'} | {d:.1e} | {t:.3f} |" for v, n, na, s, d, t in rows],
        "",
        "All NaN rows are South Sudan (SSD): it has baseline rates but no population",
        "mean risk factors in the source data, so neither implementation can score it.", "",
        "## 2. Coefficient mutation check", "",
        f"Each coefficient multiplied by (1 + {REL_NUDGE:g}); caught if the reference comparison fails.", "",
        "| Version | Coefficients | Caught |", "|---|---|---|",
        *[f"| {v} | {len(g)} | {int(g.caught.sum())} |" for v, g in m.groupby("version", sort=False)],
        "",
        "## 3. Data provenance", "",
        "Coefficients, country baseline hazards and population means come from the",
        "authors' package data (MIT). `extract_data.R` proves the package's 8.9 M",
        "per-follow-up-year hazards reduce exactly (zero conflicts) to 1.06 M rates per",
        "(outcome, country, sex, calendar year, attained age); `build_package_data.py`",
        "packs them into `src/cvd_risk/globorisk/data.npz` (1.3 MB).", "",
        "## 4. Findings", "",
        "1. **Smoking lowers risk at the oldest ages.** Lab model smoking log-HR is",
        "   1.8467 - 0.02205 x attained age, negative after ~83.7 years, so a 79-year-old",
        "   smoker's 10-year risk can be below a non-smoker's. Same in the reference.",
        "2. **Reference contradicts its own warning for age > 80.** It warns these ages",
        "   \"will be returned as NAs\" but returns values when the horizon is short.",
        "   This package follows the documented rule (NaN at age >= 81).",
        "3. **Possible bug in the reference's optional LAC update (office version):** the",
        "   term `main_sbpsexc` (an SBP x sex interaction by name) is multiplied by",
        "   sex x smoking instead of sex x SBP. Not used by default (`updated_lac = FALSE`);",
        "   the LAC update is not implemented here. To confirm with the authors.", "",
        "## 5. Open items", "",
        "- Check rounded coefficients against the papers' printed tables.",
        "- Reproduce published country risk charts (Ueda 2017) cell by cell.",
        "- LAC-recalibrated equations (`updated_lac`) not implemented.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:24]))
    print(m.groupby("version").caught.agg(["size", "sum"]))


if __name__ == "__main__":
    main()
