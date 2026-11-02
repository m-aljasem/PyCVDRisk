"""Build reference/prevent/REPORT.md. Run from repo root: python reference/prevent/report.py"""

from __future__ import annotations

import datetime as dt
import importlib
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.prevent import MODELS, OUTCOMES

mod = importlib.import_module("cvd_risk.prevent.prevent")
HERE = Path(__file__).parent
FX = HERE / "fixtures"
REQUIRED = ["age", "sex", "sbp", "bp_treated", "total_chol", "hdl_chol", "statin",
            "diabetes", "smoker", "egfr", "bmi"]


def run(q, model, years, outcome):
    return mod.prevent(**{c: q[c] for c in REQUIRED}, uacr=q.uacr, hba1c=q.hba1c, sdi=q.sdi,
                       model=model, years=years, outcome=outcome)


def last_digit(v: float) -> float:
    return float(Decimal(10) ** Decimal(repr(v)).as_tuple().exponent)


def main():
    p = pd.read_csv(FX / "patients.csv").set_index("id")
    r = pd.read_csv(FX / "reference.csv")
    original = mod._table()
    rows, muts = [], []
    for model in MODELS:
        for years in (10, 30):
            g = r[(r.model == model) & (r.years == years)]
            q = p.loc[g.id]
            ok = ~np.isnan(g.total_cvd.to_numpy())
            worst, secs = 0.0, 0.0
            for oi, outcome in enumerate(OUTCOMES):
                ref = g[outcome].to_numpy()[ok]
                t0 = time.perf_counter()
                got = run(q, model, years, outcome)[ok]
                secs += time.perf_counter() - t0
                worst = max(worst, float(np.max(np.abs(got / ref - 1))))
                terms, coefs = original[(model, years)]
                for ti, term in enumerate(terms):
                    for si, sex in enumerate(("female", "male")):
                        v = coefs[ti, oi, si]
                        if v == 0:  # term not in this equation
                            continue
                        c = coefs.copy()
                        c[ti, oi, si] = v + last_digit(float(v))
                        mod._table = lambda c=c: {**original, (model, years): (terms, c)}
                        try:
                            caught = not np.allclose(run(q, model, years, outcome)[ok], ref, rtol=1e-12, atol=1e-15)
                        finally:
                            mod._table = lambda: original
                        muts.append((model, years, f"{outcome}.{sex}.{term}", caught))
            rows.append((model, years, len(g), int((~ok).sum()), worst, secs))

    m = pd.DataFrame(muts, columns=["model", "years", "coef", "caught"])
    big = pd.concat([p] * 200, ignore_index=True)
    t0 = time.perf_counter()
    run(big, "full", 10, "total_cvd")
    one_m = time.perf_counter() - t0

    lines = [
        "# PREVENT — verification report", "",
        f"Generated {dt.date.today()} by `reference/prevent/report.py`.", "",
        "## 1. Primary source and the authors' worked examples", "",
        "Coefficients are parsed from Supplemental Tables S12.A-J of Khan et al., Circulation",
        "2024;149:430-449 (`build_coefficients.py`; files in `source/`). The original (2023) and",
        "revised (2024-01-26) supplement spreadsheets agree on all 267 coefficient rows.", "",
        "Each S12 table also contains the authors' own worked example (one patient, 10 risks).",
        "All 100 published values are reproduced to rtol 1e-12",
        "(`test_authors_worked_example`), using the coefficients as printed.", "",
        "**HbA1c erratum (2026-10-14, doi:10.1161/CIR.0000000000001491).** The coefficients",
        "labelled *HbA1c in DM* and *HbA1c no DM* are swapped in the 10- and 30-year HbA1c",
        "and full models. The erratum text (`source/erratum_CIR.0000000000001491.txt`) states",
        "that the two row headers were inverted in exactly these four tabs (S12C, S12E, S12H,",
        "S12J) and in Table 3 of the paper. Independently, preventr 0.12.2 differs from S12",
        "in exactly these values and nothing else. The S12 worked examples reproduce with the",
        "values as originally labelled, so they are tested against the uncorrected table.", "",
        "## 2. Agreement with the reference", "",
        "Reference: third-party R package preventr, with its 3-decimal rounding disabled",
        "(evidence level B).", "",
        "```", (FX / "VERSION").read_text().strip(), "```", "",
        "5,000 seeded synthetic patients (`make_patients.py`): every whole age 30-79, all",
        "inputs across the allowable ranges, about 10% missing UACR, HbA1c and SDI each (so the",
        "missing-indicator terms are exercised).", "",
        "| Model | Years | Patients | Reference NA | Max rel diff (5 outcomes) | Time (s) |",
        "|---|---|---|---|---|---|",
        *[f"| {mo} | {y} | {n:,} | {na} | {w:.1e} | {s:.3f} |" for mo, y, n, na, w, s in rows],
        "",
        f"Scoring 1,000,000 rows (full model, 10-year total CVD): {one_m:.2f} s.", "",
        "## 3. Coefficient mutation check", "",
        "Each coefficient was nudged by one unit in its last printed decimal place (1e-7).",
        "A nudge counts as caught if the comparison with preventr then fails.", "",
        "| Model | Years | Coefficients mutated | Caught |", "|---|---|---|---|",
        *[f"| {mo} | {y} | {len(g)} | {int(g.caught.sum())} |" for (mo, y), g in m.groupby(["model", "years"], sort=False)],
        "",
    ]
    missed = m[~m.caught]
    if len(missed):
        lines += ["Not caught:", "", *[f"- {a} {b}: `{c}`" for a, b, c, _ in missed.itertuples(index=False)], ""]
    lines += [
        "## 4. Findings", "",
        "1. **preventr rejects ages 79.01-79.99.** The source's allowable range is",
        "   \"30-79 years\", which this package reads as [30, 80), the same convention used for",
        "   SCORE2. preventr returns NA for these ages (89 of 5,000 patients here). All other",
        "   rows agree.",
        "2. **The HbA1c labels in S12 are wrong** (corrected by the 2026-10-14 erratum;",
        "   see section 1). Every implementation built from the supplement before then,",
        "   including preventr <= 0.12.0, gives wrong HbA1c-model risks: up to 30%",
        "   relative error in 10-year total CVD on this synthetic set.",
        "3. **preventr's ranges differ from the source's.** Source: eGFR 15-150, HbA1c 3-15%,",
        "   BMI 18.5 to <40. preventr: eGFR 15-140, HbA1c 4.5-15, BMI 18.5-39.9. This package",
        "   follows the source.",
        "4. **The original (2023) S12 tables label HDL-C \"per 1 mml/L\"**, but the term is",
        "   (HDL - 1.3)/0.3. The revised supplement's own formulas and worked examples use",
        "   0.3, and so does this package.", "",
        "## 5. Open items", "",
        "- Compare against the AHA online calculator (evidence level A).",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[20:45]))
    print(m.groupby(["model", "years"]).caught.agg(["size", "sum"]))


if __name__ == "__main__":
    main()
