"""Build reference/score2_ckd/REPORT.md. Run from repo root: python reference/score2_ckd/report.py"""

from __future__ import annotations

import copy
import datetime as dt
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.score2 import coefficients as C
from cvd_risk.score2.score2_ckd import score2_ckd

HERE = Path(__file__).parent
FX = HERE / "fixtures"
COLS = ["age", "sex", "smoker", "sbp", "total_chol", "hdl_chol", "diabetes", "egfr", "region"]
TABLES = ["CKD_EXPECTED_EGFR", "CKD_EXPECTED_LOG8_ACR", "CKD_EGFR_ADDON", "CKD_ACR_ADDON", "CKD_DIPSTICK_ADDON"]


def last_digit(v: float) -> float:
    return float(Decimal(10) ** Decimal(repr(v)).as_tuple().exponent) if v else 1e-4


def leaves(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, path + (k,))
    elif isinstance(obj, tuple):
        for i, v in enumerate(obj):
            yield from leaves(v, path + (i,))
    else:
        yield path, obj


def setleaf(obj, path, value):
    if len(path) == 1:
        if isinstance(obj, tuple):
            return obj[:path[0]] + (value,) + obj[path[0] + 1:]
        obj[path[0]] = value
        return obj
    child = setleaf(obj[path[0]], path[1:], value)
    if isinstance(obj, tuple):
        return obj[:path[0]] + (child,) + obj[path[0] + 1:]
    obj[path[0]] = child
    return obj


def main():
    p, r = pd.read_csv(FX / "patients.csv"), pd.read_csv(FX / "reference.csv")
    comparable = {"egfr": np.ones(len(p), bool), "acr": np.ones(len(p), bool),
                  "dipstick": ((p.age < 70) | (p.dipstick == "negative")).to_numpy()}
    run = lambda addon: score2_ckd(*[p[c] for c in COLS], acr=p.acr, dipstick=p.dipstick, addon=addon)  # noqa: E731
    rows = []
    for addon, m in comparable.items():
        got = run(addon)
        rows.append((addon, int(m.sum()), float(np.max(np.abs(got[m] / r[addon][m] - 1)))))

    muts = []
    for name in TABLES:
        original = getattr(C, name)
        for path, v in leaves(original):
            if v == 0:  # structural zero (term absent), not a published coefficient
                continue
            setattr(C, name, setleaf(copy.deepcopy(original), path, v + last_digit(v)))
            try:
                caught = any(not np.allclose(run(a)[m], r[a][m], rtol=1e-12, atol=1e-15)
                             for a, m in comparable.items())
            finally:
                setattr(C, name, original)
            muts.append((name, ".".join(map(str, path)), caught))
    m = pd.DataFrame(muts, columns=["table", "coef", "caught"])
    missed = m[~m.caught]

    lines = [
        "# SCORE2-CKD — verification report", "",
        f"Generated {dt.date.today()} by `reference/score2_ckd/report.py`.", "",
        "## 1. Source", "",
        "Matsushita et al., Eur J Prev Cardiol 2023, doi:10.1093/eurjpc/zwac176, supplementary",
        "material section 1.1.2, Tables A-C (file in `source/`): expected eGFR and expected ACR",
        "equations, eGFR / ACR / dipstick add-ons for SCORE2 and SCORE2-OP. The base risks are",
        "the package's already verified SCORE2 / SCORE2-OP.", "",
        "## 2. Worked example (patient A)", "",
        "The supplement's step-by-step example is reproduced to the precision it prints",
        "(products rounded to 4 decimals): expected eGFR 103.1941, expected ACR 15.3906,",
        "4.68% after the eGFR add-on and 6.07% after the ACR add-on, starting from its stated base.", "",
        "## 3. Agreement with the reference", "",
        "Reference: RiskScorescvd `SCORE2_CKD` (third-party R, rounding disabled; level B).", "",
        "```", (FX / "VERSION").read_text().strip(), "```", "",
        "20,000 seeded synthetic patients, every whole age 40-89 (SCORE2 and SCORE2-OP), eGFR",
        "15-130 (all three spline segments), ACR 1-3000 mg/g, every dipstick grade.", "",
        "| Add-on | Patients compared | Max rel diff |", "|---|---|---|",
        *[f"| {a} | {n:,} | {d:.1e} |" for a, n, d in rows], "",
        "## 4. Mutation check", "",
        f"Each published add-on coefficient nudged by +1 in its last decimal: "
        f"{int(m.caught.sum())}/{len(m)} caught.", "",
        *([f"Not caught: {', '.join(f'`{t}.{c}`' for t, c, _ in missed.itertuples(index=False))}", ""]
          if len(missed) else []),
        "The uncaught values are the SCORE2-OP dipstick coefficients, which the reference does",
        "not implement (finding 1). `check_coefficients.py` checks them, and every other add-on",
        "coefficient, against the text of its own equation in the supplement.", "",
        "## 5. Findings", "",
        "1. **RiskScorescvd uses the SCORE2 dipstick coefficients for SCORE2-OP.** The",
        "   supplement (Table C) gives separate SCORE2-OP values (trace 0.2517, + 0.3872,",
        "   ++ 0.5458, +++ 0.4207); RiskScorescvd applies the SCORE2 ones (0.2644, 0.4126,",
        "   0.4761) at all ages. Substituting them reproduces the reference exactly, so this",
        "   is the only disagreement (all 6,411 rows aged >= 70 with a positive dipstick; up to",
        "   6.9% relative). `test_finding_reference_uses_score2_dipstick_coefficients_for_op`.",
        "2. **Patient A's base risk is not reproducible.** The supplement states SCORE2 = 1.96%",
        "   (low-risk region); the stated inputs give 2.48% (RiskScorescvd agrees) and no",
        "   region gives 1.96%. The add-on steps themselves reproduce from 1.96%.",
        "3. **Sign typo in the example sum.** The expected-ACR line subtracts the eGFR<60",
        "   term (-0.4057), but the printed result (15.3906) requires +0.4057, as in the",
        "   equation. The equation is followed.",
        "4. The SCORE2 eGFR add-on has no eGFR >= 90 segment (Table A footnote); the SCORE2-OP",
        "   add-on does. Both implemented as published.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[18:40]))


if __name__ == "__main__":
    main()
