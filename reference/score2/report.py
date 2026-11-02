"""Build reference/score2/REPORT.md: agreement with the R reference and a
coefficient mutation check. Run from the repo root: python reference/score2/report.py
"""

from __future__ import annotations

import copy
import datetime as dt
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from cvd_risk.score2 import coefficients as C
import importlib

mod = importlib.import_module("cvd_risk.score2.score2")  # the module, not the same-named function

HERE = Path(__file__).parent
FX = HERE / "fixtures"
BASE = ["age", "sex", "smoker", "sbp", "total_chol", "hdl_chol", "diabetes"]
DM = BASE + ["age_at_diabetes_diagnosis", "hba1c", "egfr"]
VARIANTS = {  # variant -> (function, input columns, coefficient tables it uses)
    "score2": (mod.score2, BASE, ["SCORE2", "SCORE2_SCALES"]),
    "score2_op": (mod.score2_op, BASE, ["SCORE2_OP", "SCORE2_OP_SCALES"]),
    "score2_diabetes": (mod.score2_diabetes, DM, ["SCORE2_DIABETES", "SCORE2_SCALES"]),
    "score2_asia_pacific": (mod.score2_asia_pacific, BASE, ["SCORE2", "SCORE2_ASIA_PACIFIC_SCALES"]),
}
RTOL = 1e-12


def run(variant):
    fn, cols, _ = VARIANTS[variant]
    p = pd.read_csv(FX / f"{variant}_patients.csv")
    ref = pd.read_csv(FX / f"{variant}_reference.csv").risk.to_numpy()
    return p, ref, (lambda: fn(**{c: p[c] for c in cols}, region=p.region))


def last_digit(v: float) -> float:
    """One unit in the last published decimal place of v (e.g. 0.3742 -> 0.0001)."""
    return float(Decimal(10) ** Decimal(repr(v)).as_tuple().exponent)


def mutations(table_name):
    """Yield (label, mutated table) for every coefficient, nudged by +1 last digit."""
    table = getattr(C, table_name)
    for sex, coefs in table.items():
        items = coefs.items() if isinstance(coefs, dict) else (
            (f"{C.REGIONS[i]}.scale{j + 1}", pair[j]) for i, pair in enumerate(coefs) for j in (0, 1))
        for key, val in items:
            t = copy.deepcopy(table)
            if isinstance(coefs, dict):
                t[sex][key] = val + last_digit(val)
            else:
                region, sc = key.split(".scale")
                i, j = C.REGIONS.index(region), int(sc) - 1
                pair = list(t[sex][i])
                pair[j] = val + last_digit(val)
                t[sex] = tuple(tuple(pair) if k == i else x for k, x in enumerate(t[sex]))
            yield f"{table_name}.{sex}.{key}", t


def main():
    rows, mut_rows = [], []
    for variant, (_, _, tables) in VARIANTS.items():
        p, ref, call = run(variant)
        t0 = time.perf_counter()
        got = call()
        secs = time.perf_counter() - t0
        diff = np.abs(got - ref)
        rows.append((variant, len(p), int(np.isnan(got).sum()), diff.max(),
                     (diff / ref).max(), bool(np.allclose(got, ref, rtol=RTOL, atol=1e-15)), secs))
        for name in tables:
            original = getattr(C, name)
            for label, mutated in mutations(name):
                setattr(C, name, mutated)
                try:
                    caught = not np.allclose(call(), ref, rtol=RTOL, atol=1e-15)
                finally:
                    setattr(C, name, original)
                mut_rows.append((variant, label, caught))

    mut = pd.DataFrame(mut_rows, columns=["variant", "coefficient", "caught"])
    version = (FX / "VERSION").read_text().strip()
    lines = [
        "# SCORE2 family — verification report",
        "",
        f"Generated {dt.date.today()} by `reference/score2/report.py`.",
        "",
        "## 1. Agreement with the reference",
        "",
        "Reference: third-party R package RiskScorescvd, run with its 0.1% rounding",
        "disabled so answers compare at machine precision (evidence level B).",
        "",
        "```", version, "```", "",
        "| Variant | Patients | NaN | Max abs diff | Max rel diff | Pass (rtol 1e-12) | Time (s) |",
        "|---|---|---|---|---|---|---|",
        *[f"| {v} | {n:,} | {na} | {a:.2e} | {r:.2e} | {'yes' if ok else '**NO**'} | {s:.3f} |"
          for v, n, na, a, r, ok, s in rows],
        "",
        "Patients: seeded synthetic set (`make_patients.py`) covering every sex x region x",
        "smoker x diabetes combination and every integer age in range, both ends included.",
        "",
        "## 2. Coefficient mutation check",
        "",
        "Each published coefficient, baseline survival, mean linear predictor and region",
        "scale was nudged by one unit in its last published decimal place (e.g. 0.3742 ->",
        "0.3743). The change counts as caught if the reference comparison then fails.",
        "",
        "| Variant | Coefficients mutated | Caught |",
        "|---|---|---|",
        *[f"| {v} | {len(g)} | {int(g.caught.sum())} |" for v, g in mut.groupby("variant", sort=False)],
        "",
    ]
    missed = mut[~mut.caught]
    if len(missed):
        lines += ["Not caught:", "", *[f"- {v}: `{c}`" for v, c, _ in missed.itertuples(index=False)], ""]
    lines += [
        "## 3. Behaviour of the published models found by property testing",
        "",
        "Reproduced identically by the R reference, so these are properties of the",
        "published equations, not implementation errors. Each is pinned by a",
        "`test_finding_*` test in `tests/score2/test_score2.py`.",
        "",
        "1. **Region order is not monotone at low risk.** Because recalibration slopes",
        "   differ by region, the *moderate* region gives a higher SCORE2 risk than the",
        "   *high* region for low-risk people (11.9% of a uniform synthetic population,",
        "   risks up to ~4.9%, i.e. across the 2.5% and 5% treatment thresholds), and",
        "   *low* exceeds *moderate* for some women (risks < 0.7%).",
        "2. **SCORE2 risk can fall with age** for extreme profiles (e.g. male diabetic",
        "   smoker, SBP 175, total cholesterol 9, HDL 0.75): interaction terms cancel the",
        "   age effect.",
        "3. **SCORE2-OP: smoking lowers risk in men older than ~87.3 years**",
        "   (0.3524 - 0.0247 x (age - 73) < 0), and risk can fall with age for some",
        "   profiles (1.7% of synthetic 70-88-year-olds).",
        "",
        "4. **SCORE2 Asia-Pacific region order inverts at both ends** (uniform synthetic",
        "   population): *high* exceeds *very high* for 1.4% (high-risk profiles, risks up",
        "   to ~87%), and *low* exceeds *moderate* for 4.5% (risks up to ~5%). Follows",
        "   directly from the published rescaling factors.",
        "",
        "To do before publication: confirm each against the official ESC risk charts.",
        "",
        "## 4. Coefficient provenance — open items",
        "",
        "- SCORE2 Asia-Pacific rescaling factors were transcribed from the primary",
        "  source (Supplementary Table 5, file in `source/`) and agree exactly with",
        "  RiskScorescvd. Its *very high* region has a lower intercept than *high* for",
        "  both sexes, as printed in the source.",
        "- SCORE2/-OP/-Diabetes values were transcribed twice independently (this package; RiskScorescvd)",
        "  and agree exactly. `check_coefficients.py` also compares all 116 of them with the",
        "  supplement text (files in `source/`), located by row label: 116/116 match. It",
        "  flags a mutated value (checked by hand with four mutations).",
        "- Not yet compared with the official ESC calculator (HeartScore) or charts",
        "  (evidence level A).",
        "- No published worked examples have been located yet.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:30]))
    print(mut.groupby("variant").caught.agg(["size", "sum"]))


if __name__ == "__main__":
    main()
