"""Build reference/pce/REPORT.md. Run from repo root: python reference/pce/report.py"""

from __future__ import annotations

import copy
import datetime as dt
import importlib
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.prevent.prevent import MG_PER_MMOL

mod = importlib.import_module("cvd_risk.pce.pce")
HERE = Path(__file__).parent
FX = HERE / "fixtures"


def last_digit(v: float) -> float:
    return float(Decimal(10) ** Decimal(repr(v)).as_tuple().exponent)


def main():
    p = pd.read_csv(FX / "patients.csv")
    r = pd.read_csv(FX / "reference.csv")
    run = lambda q: mod.pce(q.age, q.sex, q.race, q.total_chol_mgdl / MG_PER_MMOL,  # noqa: E731
                            q.hdl_chol_mgdl / MG_PER_MMOL, q.sbp, q.bp_treated, q.smoker, q.diabetes)
    na = r.cvrisk.isna().to_numpy()
    ok = lambda g: np.allclose(g[~na], r.cvrisk[~na], rtol=1e-12, atol=1e-15)  # noqa: E731
    t0 = time.perf_counter()
    got = run(p)
    secs = time.perf_counter() - t0
    d_cv = float(np.max(np.abs(got[~na] / r.cvrisk[~na] - 1)))
    d_pc = np.abs(got / r.pooledcohort - 1)

    original, caught = mod.MODELS, []
    for g, coefs in original.items():
        for k, v in coefs.items():
            m = copy.deepcopy(original)
            m[g][k] = v + last_digit(v)
            mod.MODELS = m
            try:
                caught.append(not ok(run(p)))
            finally:
                mod.MODELS = original

    big = pd.concat([p] * 50, ignore_index=True)
    t0 = time.perf_counter()
    run(big)
    secs_1m = (time.perf_counter() - t0) * 1e6 / len(big)

    lines = [
        "# Pooled Cohort Equations (2013 ACC/AHA) — verification report", "",
        f"Generated {dt.date.today()} by `reference/pce/report.py`.", "",
        "## 1. Source", "",
        "Goff et al., 2013 ACC/AHA Guideline on the Assessment of Cardiovascular Risk,",
        "Appendix 7 Table A (author manuscript PMC4700825, saved in `source/`).",
        "`check_coefficients.py` parses Table A from that file: all 46 values (coefficients,",
        "means, baseline survivals) equal `coefficients.py`.", "",
        "**Worked example (Table A/B).** For the guideline's example patient the package gives",
        "2.1%, 3.0%, 5.4% and 6.1% (White women, African American women, White men, African",
        "American men); the guideline prints 2.1, 3.0, **5.3** and 6.1. For White men the exact",
        "individual sum is 60.6995 but the table prints 60.69 (truncated), and Table B computes",
        "1 - 0.9144^exp(60.69 - 61.18) = 5.33%. With the exact sum the risk is 5.38%. The",
        "difference is the guideline's rounding of an intermediate value, not the equation.", "",
        "## 2. Agreement with the references", "",
        "```", (FX / "VERSION").read_text().strip(), "```", "",
        f"{len(p):,} seeded synthetic patients (every whole age 40-79, both sexes and races,",
        "treated and untreated SBP, within CVrisk's accepted lipid and SBP ranges).", "",
        "| Reference | Rows compared | Max rel diff |", "|---|---|---|",
        f"| CVrisk 1.2.1 (rounding removed) | {(~na).sum():,} | {d_cv:.1e} |",
        f"| PooledCohort 0.0.2, as is | {len(p):,} | {d_pc.max():.1e} (median {np.median(d_pc):.1e}) |",
        "| PooledCohort 0.0.2, its constants substituted | 20,000 | < 1e-12 (test) |", "",
        "1. **CVrisk returns NA for ages 79.01-79.99** (it requires age <= 79). The guideline's",
        "   \"40 to 79 years\" is read here as [40, 80), as for SCORE2 and PREVENT. CVrisk also",
        "   floors risks below 1% to 1% and accepts ages from 20; the guideline does neither.",
        "2. **PooledCohort uses more decimals than the guideline prints** for the baseline",
        "   survivals, means, and all African American women coefficients (e.g. mean 86.6081",
        "   vs 86.61). This shifts risks by up to 1% relative. With its constants substituted,",
        "   this package matches it to < 1e-12, so the equations agree and only the constants",
        "   differ. Their provenance (presumably the ACC/AHA spreadsheet) is not documented; this",
        "   package uses the published values.", "",
        f"Scoring {len(p):,} rows: {secs:.3f} s; 1,000,000 rows: {secs_1m:.2f} s.", "",
        "## 3. Mutation check", "",
        f"Every value in Table A, +1 in its last printed digit: {sum(caught)}/{len(caught)} caught by CVrisk.", "",
        "## 4. Open items", "",
        "- Compare against the official ACC/AHA ASCVD Risk Estimator (evidence level A); its",
        "  site did not respond to automated requests.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[18:]))


if __name__ == "__main__":
    main()
