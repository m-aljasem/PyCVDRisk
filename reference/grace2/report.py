"""Build reference/grace2/REPORT.md. Run from repo root, after node reference/grace2/run_app.js:

    PYTHONPATH=src python reference/grace2/report.py
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tests" / "grace"))
T = importlib.import_module("test_grace2")
M = importlib.import_module("cvd_risk.grace.grace2")
from cvd_risk import grace2  # noqa: E402

APP = "https://www.outcomes-umassmed.org/grace/acs_risk2/"
MERCK = ("https://www.merckmanuals.com/professional/multimedia/clinical-calculator/"
         "grace-20-score-for-acute-coronary-syndrome-prognosis-non-linear-model-2014")


def app_rows():
    a = pd.read_csv(HERE / "fixtures" / "app_reference.csv")
    yn = lambda s: (s == "yes").astype(int)
    out = []
    for model, g in a.groupby("model"):
        kw = (dict(creatinine=g.creatinine, killip=g.killip) if model == "full"
              else dict(renal_insufficiency=yn(g.renal), diuretic=yn(g.diuretic)))
        for col, h, o in (("d1", 1, "death"), ("dmi1", 1, "death_mi"), ("d3", 3, "death"), ("dmi3", 3, "death_mi")):
            got = 100 * grace2(g.age, g.pulse, g.sbp, yn(g.cardiac), yn(g.stemi), yn(g.enzymes), horizon=h, outcome=o, **kw)
            out.append((model, f"{h}-year {o.replace('_mi', ' or MI')}", len(g), int((T._shown(got) == g[col + "_shown"]).sum()),
                        float(np.max(np.abs(got - g[col]) / g[col]))))
    return out


# Merck Manuals "GRACE 2.0 Score" (iframe medical-calculators/grace2score.htm, function GRACE2Score_fx), transcribed.
def _piece(x, steps, last):
    for lim, f in steps:
        if x < lim:
            return f(x)
    return last


MERCK_POINTS = {
    "SixMonth_Age": lambda a: _piece(a, [(35, lambda a: 0), (45, lambda a: (a - 35) * 1.8), (55, lambda a: 18 + (a - 45) * 1.8),
                                         (65, lambda a: 36 + (a - 55) * 1.8), (75, lambda a: 54 + (a - 65) * 1.9),
                                         (85, lambda a: 73 + (a - 75) * 1.8), (90, lambda a: 91 + (a - 85) * 1.8)], 100),
    "SixMonth_Pulse": lambda h: _piece(h, [(70, lambda h: 0), (80, lambda h: (h - 70) * 0.3), (90, lambda h: 3 + (h - 80) * 0.2),
                                           (100, lambda h: 5 + (h - 90) * 0.3), (110, lambda h: 8 + (h - 100) * 0.2),
                                           (150, lambda h: 10 + (h - 110) * 0.3), (200, lambda h: 22 + (h - 150) * 0.3)], 34),
    "SixMonth_BloodPressure": lambda s: _piece(s, [(80, lambda s: 40), (100, lambda s: 40 - (s - 80) * 0.3),
                                                   (110, lambda s: 34 - (s - 100) * 0.3), (120, lambda s: 31 - (s - 110) * 0.4),
                                                   (130, lambda s: 27 - (s - 120) * 0.3), (140, lambda s: 24 - (s - 130) * 0.3),
                                                   (150, lambda s: 20 - (s - 140) * 0.4), (160, lambda s: 17 - (s - 150) * 0.3),
                                                   (180, lambda s: 14 - (s - 160) * 0.3), (200, lambda s: 8 - (s - 180) * 0.4)], 0),
    "SixMonth_Creatine": lambda c: _piece(c, [(0.2, lambda c: c * 5), (0.4, lambda c: 1 + (c - 0.2) * 10),
                                              (0.6, lambda c: 3 + (c - 0.4) * 5), (0.8, lambda c: 4 + (c - 0.6) * 10),
                                              (1.0, lambda c: 6 + (c - 0.8) * 5), (1.2, lambda c: 7 + (c - 1.0) * 5),
                                              (1.4, lambda c: 8 + (c - 1.2) * 10), (1.6, lambda c: 10 + (c - 1.4) * 5),
                                              (1.8, lambda c: 11 + (c - 1.6) * 10), (2.0, lambda c: 13 + (c - 1.8) * 5),
                                              (3.0, lambda c: 14 + (c - 2) * 7), (4.0, lambda c: 21 + (c - 3) * 7)], 28),
}


def merck_vs_app():
    s = (HERE / "source" / "grace_lookups.js").read_text(encoding="utf-8")
    out = []
    for name, f in MERCK_POINTS.items():
        t = re.search(rf"var {name}\s*=\s*\{{(.*?)\}}", s, re.S).group(1)
        pairs = [(float(k), float(v)) for k, v in re.findall(r'"([\d.]+)"\s*:\s*"([-\d.]+)"', t)]
        out.append((name, len(pairs), [(k, v, round(f(k), 4)) for k, v in pairs if abs(f(k) - v) > 0.006]))
    return out


def main():
    q = json.loads((HERE / "fixtures" / "qunit.json").read_text())
    j = pd.read_csv(HERE / "fixtures" / "gracejs_reference.csv")
    args = (j.age, j.pulse, j.sbp, j.cardiac, j.stemi, j.enzymes)
    js = max(np.max(np.abs(grace2(*args, creatinine=j.creatinine, killip=j.killip) - j.d1)),
             np.max(np.abs(grace2(*args, renal_insufficiency=j.renal, diuretic=j.diuretic) - j.d1_substituted)))
    rows = app_rows()
    shown_ok, shown_n = sum(r[3] for r in rows), sum(r[2] for r in rows)
    merck = merck_vs_app()
    v = M._M[3, "death", False][3]
    seg = lambda i, k, s1, s2: float(np.exp((v[k] - v[i]) / ((s2 - s1) / -20)))
    mid = (79, 89.5, 104.5, 114.5, 124.5)
    low = [seg(i, i + 1, mid[i], mid[i + 1]) for i in range(4)]
    L = [
        "# GRACE 2.0: verification report", "",
        f"Generated by reference/grace2/report.py. Function: `cvd_risk.grace2` "
        "(1- and 3-year death, and death or MI, after admission for ACS; full and substituted models).", "",
        "## 1. Sources", "",
        f"- **Developers' calculator (evidence A).** The GRACE 2.0 web app of the Center for Outcomes Research, "
        f"UMass ({APP}), from the Wayback Machine. `source/grace_lookups.js`, `grace.js` and `tests.js` (QUnit) "
        "are byte-identical in the 2021 and 2024-07-24 captures; `index.html` and `commonFunctions.js` are from "
        "the 2024-07-24 capture (20240724120620). The files are unmodified.",
        "- **Paper (evidence C).** Fox KAA, FitzGerald G, Puymirat E, et al. BMJ Open 2014;4:e004425 "
        "(`source/fox2014.pdf`). It prints hazard ratios only (Table 2), no coefficients, knots or baseline "
        "survival.",
        "- The coordinating centre's *Methods and formulas used to calculate the GRACE Risk Scores* (2014-01-30, "
        "cited by the paper as the source of the model estimates) covers only the GRACE 1.0 models "
        "(Granger 2003, Eagle 2004, Fox 2006); it is not stored here.", "",
        "## 2. What the calculator computes", "",
        "`commonFunctions.js` calls only `grace_lookups.js`. In-hospital and 6-month results are the GRACE 1.0 "
        "models (Granger logistic model; the in-hospital death/MI, Fox 2006 admission-to-6-month death and "
        "death/MI point nomograms) and are not part of `grace2`. The 1- and 3-year results are eight Cox models, "
        "risk = 1 - S0^exp(sum of terms), each term read from a table at the dropdown entry chosen: every year of "
        "age from 50 to 89 (one value each for <30, 30-39, 40-44, 45-49, 90-94 and >=95), and one value per "
        "band of heart rate, SBP and creatinine (e.g. heart rate 50-69 is computed at 59.5). If creatinine or "
        "Killip class is 'not available' it uses the substituted models (renal insufficiency, diuretics). "
        "`grace.js`, loaded by the page only for its QUnit tests, codes the two 1-year death models with "
        "continuous inputs.", "",
        "## 3. Coefficients", "",
        "The 1-year death models' knots and coefficients are those of `grace.js` (age per 10 years, knots "
        "4.8165/6.6571/8.3095; heart rate per 30 bpm, 1.71/2.31/2.77/3.95; SBP per -20 mmHg, -9.02/-6.97/-5.32; "
        "creatinine, 0.685/0.925/1.145/2.065 mg/dL). For the other six models the table values were fitted with "
        "the same knots and spline terms; the coefficients come out at 5 decimals, as in `grace.js`, and the "
        "baseline survivals are those in `grace_lookups.js`. "
        "Every table value (836: 8 models x 72 ages, 10 heart rates, 9 SBPs, 7 creatinines in the full models, "
        "Killip or renal + diuretics, 3 yes/no items) is reproduced to the digits printed "
        "(`test_coefficients_reproduce_every_calculator_lookup_value`). The exception is the SBP term of the "
        "two 3-year death models: no restricted cubic spline with 3 or 4 knots (positions searched over 30-260 "
        "mmHg), polynomial or linear spline reproduces it (a 5-knot spline has as many parameters as values), so `grace2` uses the calculator's value for each SBP band.", "",
        "## 4. Against the developers' code", "",
        "`run_app.js` runs the unmodified files in Node: the calculator's functions on the base patient, every "
        "dropdown entry changed one at a time, and 3,000 random combinations, for each model type.", "",
        "| Model | Outcome | Patients | Displayed value equal | Max relative difference (unrounded) |",
        "|---|---|---|---|---|",
        *[f"| {m} | {o} | {n:,} | {k:,} | {d:.1e} |" for m, o, n, k, d in rows], "",
        f"All {shown_ok:,} of {shown_n:,} displayed values are equal (the calculator shows one decimal below 10% "
        "and whole numbers from 10%); the unrounded differences are the tables' printed precision. "
        f"`grace.js` on 500 random patients with continuous inputs: max |difference| {js:.1e}. The developers' "
        "four worked examples in `tests.js` (1-year death 28% and 2%, substituted 30% and 2%) are reproduced.", "",
        "## 5. Against the paper", "",
        "Fox 2014 Table 2 prints hazard ratios to one decimal. The 24 categorical and linear terms of the 1-year "
        "death, 1-year death/MI and 3-year death models give exactly those values when exp(b) is rounded to two "
        "decimals and then to one (22 of 24 when rounded once: 1-year death Killip III exp(0.85325) = 2.347 and "
        "IV exp(1.29372) = 3.646 print as 2.4 and 3.7). Table 2 has no 3-year death/MI column. For the 3-year "
        "death SBP term Table 2 gives per -20 mmHg HR 0.9 at >=160, 1.6 at 130-159 and 1.3 below 130; the "
        f"calculator's band values give {seg(7, 8, 169.5, 180):.2f} (160-179 to >=180) and "
        f"{seg(5, 6, 134.5, 149.5):.2f} (130-139 to 140-159), but {min(low):.2f} to {max(low):.2f} between the bands below 130.", "",
        "## 6. Findings", "",
        "1. **The calculator bands its inputs.** The paper says version 2.0 'directly utilises model estimates "
        "themselves' instead of 'using intervals for continuous variables such as age, as in V.1.0'. The web "
        "calculator evaluates the spline models at one value per band of heart rate, SBP and creatinine, and of "
        "age below 50 and from 90. `grace2` takes continuous inputs and equals the calculator at its band values.",
        "2. **3-year death SBP term.** Its values fit no spline and, below 130 mmHg, not the hazard ratio in "
        "Table 2 (section 5); `grace2` uses the band values.",
        f"3. **`grace.js` fails its own tests.** With the `Number.prototype.roundup` the calculator ships "
        f"(which returns Math.round(x), a whole number, for x >= 10) {q['shipped_roundup']['failed']} of "
        f"{q['shipped_roundup']['asserts']} QUnit checks fail, because spline terms of 10 and above are rounded to "
        f"whole numbers; without that branch {q['roundup_without_ge10_branch']['failed']} fails "
        "(`age_toDecades(0)` expected to be 1). The calculator does not use `grace.js`, so its results are not "
        "affected.",
        "4. **6-month death in the calculator** (GRACE 1.0, Fox 2006 admission-to-6-month nomogram; not part of "
        "`grace2`): the heart-rate value for 150-199 bpm is 34.37 (computed at 174.5), the in-hospital death/MI "
        "nomogram's value; the Fox 2006 nomogram in the coordinating centre's formulas gives 22 + (174.5 - 150) "
        "x 0.3 = 29.35. Its score-to-probability steps leave scores above 6 and up to 7 without a probability "
        "(the result is NaN) and set 113% above 240 points; the result is then multiplied by 80/91 and "
        "values above 90% are set to 92%.",
        "5. **Merck Manuals 'GRACE 2.0 Score (non-linear model 2014)'** "
        f"({MERCK}) computes the Fox 2006 admission-to-6-month death points (Killip 0/15/29/44, cardiac arrest "
        "30, enzymes 13, ST deviation 17; age, heart rate, SBP and creatinine interpolated within bands), not "
        "the GRACE 2.0 Cox models. Its formulas equal the developers' calculator's 6-month death tables at "
        + ", ".join(f"{n - len(bad)}/{n} {name.split('_')[1]}" for name, n, bad in merck)
        + " entries; the exception is the heart-rate band of finding 4, where Merck gives 29.35.",
        "6. In the 3-year models positive enzymes have no effect and Killip III and IV share one coefficient "
        "(as Table 2 reports).", "",
    ]
    (HERE / "REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
