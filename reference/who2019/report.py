"""Build reference/who2019/REPORT.md. Run from repo root: python reference/who2019/report.py"""

from __future__ import annotations

import copy
import datetime as dt
import importlib
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.who2019 import coefficients as C

mod = importlib.import_module("cvd_risk.who2019.who2019")
HERE = Path(__file__).parent
FX = HERE / "fixtures"
CONFIGS = [(v, rc) for v in ("lab", "nonlab") for rc in ("region", "country")]


def run(p, version, recal):
    kw = dict(total_chol=p.total_chol, diabetes=p.diabetes) if version == "lab" else dict(bmi=p.bmi)
    return mod.who2019(p.age, p.sex, p.sbp, p.smoker, **kw, country=p.country, version=version,
                       recalibration=recal, year=p.year)


def last_digit(v: float) -> float:
    return float(Decimal(10) ** Decimal(repr(v)).as_tuple().exponent)


def main():
    p = pd.read_csv(FX / "patients.csv", keep_default_na=False, na_values=[""])
    r = pd.read_csv(FX / "reference.csv")
    ok = lambda got, ref: np.allclose(got, ref, rtol=1e-12, atol=1e-15)  # noqa: E731

    rows = []
    for v, rc in CONFIGS:
        t0 = time.perf_counter()
        got = run(p, v, rc)
        rows.append((v, rc, float(np.max(np.abs(got / r[f"{v}_{rc}"] - 1))), time.perf_counter() - t0))

    # Mutation 1: model coefficients (log HRs, baseline survival), +1 in last printed digit.
    coef_muts = []
    for name in ("LAB", "NONLAB"):
        original = getattr(C, name)
        v = name.lower()
        for outcome, by_sex in original.items():
            for sex, coefs in by_sex.items():
                for key, val in coefs.items():
                    t = copy.deepcopy(original)
                    t[outcome][sex][key] = val + last_digit(val)
                    setattr(C, name, t)
                    try:
                        caught = not ok(run(p, v, "region"), r[f"{v}_region"])
                    finally:
                        setattr(C, name, original)
                    coef_muts.append(caught)

    # Mutation 2: recalibration parameters, x (1 + 1e-6), each one tested on the
    # patients whose risk uses that parameter row.
    countries, region_of, tables = mod._data()
    s = (p.sex == "male").to_numpy().astype(int)
    y = np.floor(p.year.to_numpy())
    yi = np.searchsorted(mod.YEARS, np.where(y <= 2015, 5 * np.floor(y / 5), y))
    ci = np.array([countries.index(c) for c in p.country])
    ri = region_of[ci]
    has_country = ~np.isnan(tables["country"][ci, s, yi, 0])
    total, exercised, caught = 0, 0, 0
    for level, idx, use in (("region", ri, ~has_country), ("country", ci, has_country)):
        tab = tables[level]
        for code in range(tab.shape[0] - 1):
            for sx in (0, 1):
                for yy in range(len(mod.YEARS)):
                    if np.isnan(tab[code, sx, yy, 0]):
                        continue
                    sel = (idx == code) & (s == sx) & (yi == yy)
                    for col in range(8):
                        total += 1
                        v = "lab" if col < 4 else "nonlab"
                        rcs = (["region"] if level == "region" else []) + ["country"]
                        pats = [(rc, sel if rc == "region" else sel & (use if level == "region" else has_country))
                                for rc in rcs]
                        pats = [(rc, m) for rc, m in pats if m.any()]
                        if not pats:
                            continue
                        exercised += 1
                        old = tab[code, sx, yy, col]
                        tab[code, sx, yy, col] = old * (1 + 1e-6)
                        try:
                            caught += any(not ok(run(p[m], v, rc), r[f"{v}_{rc}"][m]) for rc, m in pats)
                        finally:
                            tab[code, sx, yy, col] = old

    big = pd.concat([p] * 34, ignore_index=True)
    t0 = time.perf_counter()
    run(big, "lab", "country")
    secs_1m = (time.perf_counter() - t0) * 1e6 / len(big)

    lines = [
        "# WHO 2019 — verification report", "",
        f"Generated {dt.date.today()} by `reference/who2019/report.py`.", "",
        "## 1. Sources", "",
        "- Log hazard ratios: Kaptoge et al., Lancet Glob Health 2019, appendix 1 Table 1.6.",
        "  `check_coefficients.py` confirms all 64 values against the PDF text.",
        "- Baseline survival: the paper (Table 2) prints 3 decimals. The 7-decimal values come",
        "  from the authors' Stata program `whocvdrisk` via WHORiskCalculator 1.1.0, and all 8",
        "  round to the printed ones.",
        "- Recalibration parameters (3,346 rows: 21 regions and 190 countries, by sex, for",
        "  1990-2017): the authors' Stata data file via WHORiskCalculator 1.1.0 (MIT),",
        "  exported by `extract_data.R`. These are not printed in the publication.", "",
        "## 2. The official WHO risk charts", "",
        "`parse_charts.py` reads all 42 published charts (supplementary appendix 2; 21 regions,",
        "lab and non-lab), 44,100 cells in total. **Every cell is reproduced exactly**",
        "(`test_reproduces_every_official_chart_cell`), with cells evaluated at band midpoints:",
        "age band + 2.5; SBP 110/130/150/170/190; cholesterol 3.5-7.5; BMI 17.5-37.5. The",
        "midpoints were found by search and are sharply determined: moving a BMI midpoint by",
        "0.5 breaks 8-13% of that band's cells, and moving age to band + 2 breaks 31% of lab",
        "cells. This independently confirms the region-level 2017 recalibration",
        "parameters against WHO's own publication (evidence level A for the region model).", "",
        "## 3. Agreement with the reference", "",
        "Reference: WHORiskCalculator (R, CRAN), which by its own account is tested against",
        "a re-implementation of the authors' Stata code (evidence level B).", "",
        "```", (FX / "VERSION").read_text().strip(), "```", "",
        "30,000 seeded synthetic patients: every country (so every region), every whole age",
        "40-80, years 1990-2017.", "",
        "| Model | Recalibration | Max rel diff | Time (s) |", "|---|---|---|---|",
        *[f"| {v} | {rc} | {d:.1e} | {t:.3f} |" for v, rc, d, t in rows], "",
        f"Scoring 1,000,000 rows (lab, country recalibration): {secs_1m:.2f} s.", "",
        "## 4. Mutation check", "",
        f"- Model coefficients, +1 in the last printed digit: {sum(coef_muts)}/{len(coef_muts)} caught.",
        f"- Recalibration parameters, x (1 + 1e-6): {exercised:,} of {total:,} are used by at least one",
        f"  synthetic patient, and {caught:,}/{exercised:,} of those are caught.", "",
        "## 5. Notes", "",
        "- v1 of this package validated against `whoishRisk`, which is the 2007 WHO/ISH model,",
        "  not this one.",
        "- Five territories (SSD, VIR, ASM, GUM, MNP) have no country parameters. With",
        "  `recalibration=\"country\"` they fall back to their region, as WHORiskCalculator does.",
        "- The author email drafted in `README.md` is no longer needed.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[26:]))


if __name__ == "__main__":
    main()
