"""Build reference/framingham/REPORT.md. Run from repo root: python reference/framingham/report.py"""

from __future__ import annotations

import copy
import datetime as dt
import importlib
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.prevent.prevent import MG_PER_MMOL as K

mod = importlib.import_module("cvd_risk.framingham.framingham")
HERE = Path(__file__).parent
FX = HERE / "fixtures"


def last_digit(v: float) -> float:
    return float(Decimal(10) ** Decimal(repr(v)).as_tuple().exponent)


def main():
    p = pd.read_csv(FX / "patients.csv")
    r = pd.read_csv(FX / "reference.csv")
    run = lambda q, m: mod.framingham(  # noqa: E731
        q.age, q.sex, q.sbp, q.bp_treated, q.smoker, q.diabetes, model=m,
        **(dict(total_chol=q.total_chol_mgdl / K, hdl_chol=q.hdl_chol_mgdl / K) if m == "lab" else dict(bmi=q.bmi)))
    na = r.lab.isna().to_numpy()
    ok = lambda g, m: np.allclose(g[~na], r[m][~na], rtol=1e-12, atol=1e-15)  # noqa: E731
    rows = []
    for m in ("lab", "bmi"):
        t0 = time.perf_counter()
        g = run(p, m)
        rows.append((m, float(np.max(np.abs(g[~na] / r[m][~na] - 1))), time.perf_counter() - t0))

    original, caught = mod.MODELS, []
    for m, by_sex in original.items():
        for sex, coefs in by_sex.items():
            for k, v in coefs.items():
                t = copy.deepcopy(original)
                t[m][sex][k] = v + last_digit(v)
                mod.MODELS = t
                try:
                    caught.append(not ok(run(p, m), m))
                finally:
                    mod.MODELS = original

    big = pd.concat([p] * 50, ignore_index=True)
    t0 = time.perf_counter()
    run(big, "lab")
    secs_1m = (time.perf_counter() - t0) * 1e6 / len(big)

    lines = [
        "# Framingham 2008 general CVD — verification report", "",
        f"Generated {dt.date.today()} by `reference/framingham/report.py`.", "",
        "## 1. Sources", "",
        "- Lab model: D'Agostino et al., Circulation 2008, Table 2; mean sums from the worked",
        "  examples. BMI model: supplementary Table I. `check_coefficients.py` compares all 32",
        "  printed values with the PDF text: 32/32 match.",
        "- The BMI model's mean sums (26.0145, 23.9388) are not printed in either PDF. They come",
        "  from the Framingham Heart Study's own calculator spreadsheets (`source/gencardio_*.xls`,",
        "  framinghamheartstudy.org), which also confirm both baseline survivals.", "",
        "## 2. Official calculator and worked examples (evidence level A + C)", "",
        "- Each spreadsheet caches one example patient scored four ways (sex x treated SBP).",
        "  All 8 cached risks are reproduced to < 1e-12 (`test_reproduces_official_spreadsheet`).",
        "- Paper Score Sheet, case 1 (woman, 61) and case 2 (man, 53): 10.5% and 15.6%,",
        "  as printed (`test_paper_worked_examples`). Together these exercise every term.", "",
        "## 3. Agreement with CVrisk (evidence level B)", "",
        "```", (FX / "VERSION").read_text().strip(), "```", "",
        f"{len(p):,} seeded synthetic patients; {(~na).sum():,} compared.", "",
        "| Model | Max rel diff | Time (s) |", "|---|---|---|",
        *[f"| {m} | {d:.1e} | {t:.3f} |" for m, d, t in rows], "",
        f"Scoring 1,000,000 rows (lab): {secs_1m:.2f} s.", "",
        "CVrisk differs from the source in three ways, all removed or excluded before comparing:",
        "it returns NA for ages 74.01-74.99 (this package reads \"30-74\" as [30, 75)), floors",
        "risks below 1% to 1%, and caps BMI-model risks at 30%. The publication does none of these.", "",
        "## 4. Mutation check", "",
        f"Every value, +1 in its last printed digit: {sum(caught)}/{len(caught)} caught by CVrisk.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[17:]))


if __name__ == "__main__":
    main()
