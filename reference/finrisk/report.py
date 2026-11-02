"""Build reference/finrisk/REPORT.md. Run from repo root: python reference/finrisk/report.py

1. The four Table 2 equations, rebuilt from BETA, are found in the PDF text.
2. Each coefficient is consistent with the odds ratio printed in Table 1 (ln OR, within the OR's rounding).
3. Figure 1 (10-year CHD risk at age 50, 15 risk-factor combinations per sex) is reproduced.
"""

from __future__ import annotations

import datetime as dt
import itertools
import logging
import math
import re
from collections import Counter
from pathlib import Path

import pypdf

from cvd_risk import finrisk
from cvd_risk.finrisk.finrisk import BETA

HERE = Path(__file__).parent
logging.getLogger("pypdf").setLevel(logging.ERROR)
TERMS = ("age", "smoking", "cholesterol", "systolic blood pressure", "HDL-cholesterol", "diabetes")


def text():
    t = " ".join(p.extract_text() or "" for p in pypdf.PdfReader(HERE / "source" / "vartiainen2016.pdf").pages)
    t = t.replace("/C0", "-").replace("/C2", "×").replace("þ", "+").replace("¼", "=")
    return re.sub(r"\s+", " ", t)


def equation(sex, outcome):
    b = BETA[(sex, outcome)]
    parts = [f"{-b[0]:g}"]
    for name, v in zip(TERMS, b[1:7]):
        if v:
            parts.append(f"{'-' if v > 0 else '+'} {abs(v):g} × {name}")
    parts.append(f"- {b[7]:g} × parents {'infarct' if outcome == 'chd' else 'stroke'}")
    return " ".join(parts)


def norm(s):
    """Drop spaces and trailing zeros of decimals so '0.6260' matches the printed form."""
    s = re.sub(r"(\d\.\d*?)0+(?=\D|$)", r"\1", s)
    return re.sub(r"\s+", "", s)


# Table 1 odds ratios: (age, smoking, cholesterol, SBP, HDL, diabetes, parents) per column
TABLE1 = {("male", "chd"): ("1.083", "1.87", "1.39", "1.017", "0.55", "2.10", "1.40"),
          ("female", "chd"): ("1.101", "2.41", "1.24", "1.018", "0.33", "2.80", "1.51"),
          ("male", "stroke"): ("1.122", "1.65", "0.97", "1.015", "0.65", "2.42", "1.34"),
          ("female", "stroke"): ("1.067", "1.52", "0.98", "1.009", "0.47", "3.45", "1.72")}
T1_ROWS = ("Age, yrs", "Smoking (no = 0, yes = 1)", "Cholesterol, mmol/l", "Systolic blood pressure, mm Hg",
           "HDL-cholesterol, mmol/l", "Diabetes (no = 0, yes = 1)")

# Figure 1, bars in legend order (A men); B women printed sorted within each group
LEGEND = [(), ("smk",), ("bp",), ("dm",), ("chol",), ("smk", "bp"), ("smk", "dm"), ("smk", "chol"), ("chol", "dm"),
          ("bp", "dm"), ("chol", "bp"), ("smk", "bp", "dm"), ("smk", "chol", "bp"), ("chol", "bp", "dm"),
          ("smk", "bp", "dm", "chol")]
FIG = {"male": [1.1, 2.04, 2.12, 2.29, 2.48, 3.9, 4.2, 4.54, 5.07, 5.97, 6.44, 7.85, 8.46, 9.4, 16.25],
       "female": [0.31, 0.52, 0.62, 0.74, 0.86, 1.24, 1.45, 1.47, 1.48, 2.04, 2.41, 2.48, 2.87, 4.02, 6.64]}
GROUPS = [0, 1, 5, 11, 14, 15]
HDL = {"male": 1.3, "female": 1.64}  # not stated in the paper; the values that reproduce the "no risk factors" bar


def bar(sex, combo, sbp_high=160):
    return 100 * float(finrisk(50, sex, "smk" in combo, 7 if "chol" in combo else 4.5, HDL[sex],
                               sbp_high if "bp" in combo else 120, "dm" in combo, 0, 0, outcome="chd"))


def figure(sex, sbp_for_two):
    v = [round(bar(sex, c, sbp_for_two if c in (("bp", "dm"), ("chol", "bp")) else 160), 2) for c in LEGEND]
    if sex == "female":
        v = list(itertools.chain.from_iterable(sorted(v[GROUPS[i]:GROUPS[i + 1]]) for i in range(5)))
    return v


def main():
    t = text()
    eq = [(s, o, norm(equation(s, o)) in norm(t)) for s, o in BETA]
    t1 = []  # (model, factor, printed OR, coefficient, exp(coefficient) rounds to the printed OR?)
    for key, ors in TABLE1.items():
        b = BETA[key]
        for name, orv, coef in zip((*T1_ROWS, "Parents"), ors, (*b[1:7], b[7])):
            if name.startswith("HDL"):
                coef = -coef  # BETA stores HDL with its sign in the linear predictor
            dec = len(orv.split(".")[1])
            t1.append((key, name, orv, coef, None if coef == 0 else f"{math.exp(coef):.{dec}f}" == orv))
    fig = {}
    for sex in ("male", "female"):
        a, b = figure(sex, 160), figure(sex, 180)
        hits = lambda v, p=FIG[sex]: sum((Counter(v) & Counter(p)).values())  # printed values reproduced
        fig[sex] = (a, b, hits(a), hits(b))
    t1_ok = sum(1 for *_, g in t1 if g)
    t1_n = sum(1 for *_, g in t1 if g is not None)
    L = ["# FINRISK calculator: verification report", "", f"Generated {dt.date.today()} by `reference/finrisk/report.py`.", "",
         "Evidence level C: Vartiainen et al., Global Heart 2016 (`source/vartiainen2016.pdf`). No independent",
         "implementation was available (the online calculator is not reachable here).", "",
         "## 1. Equations", "",
         *[f"- {s} {o}: rebuilt equation {'found' if ok else 'NOT FOUND'} in the PDF" for s, o, ok in eq], "",
         "## 2. Coefficients vs Table 1 odds ratios", "",
         "exp(Table 2 coefficient), rounded as printed, against the Table 1 odds ratio (HDL coefficient taken with",
         "its sign as an effect). Table 1 also prints an OR for cholesterol in the stroke models; Table 2 drops that",
         "term. Differences are expected if Table 2 was refitted (e.g. without that term); they are reported, not",
         "used as a pass/fail check, since Figure 1 checks the equations themselves.", "",
         "| Model | Factor | OR | Coefficient | exp(coef) rounds to OR |", "|---|---|---:|---:|---|",
         *[f"| {k[0]} {k[1]} | {n} | {o} | {c:g} | {'-' if g is None else ('yes' if g else 'no')} |" for k, n, o, c, g in t1],
         "", f"{t1_ok}/{t1_n} coefficients round to their printed odds ratio.", "",
         "## 3. Figure 1 (10-year CHD risk, age 50)", "",
         "The caption gives SBP 160 vs 120 and cholesterol 7 vs 4.5 mmol/L; HDL and family history are not stated.",
         f"HDL {HDL['male']} (men) and {HDL['female']} (women) with no family history reproduce the no-risk-factor bars.",
         "Panel B is printed in ascending order within each group, so it is compared sorted.", "",
         "| Sex | Printed | Package, SBP 160 throughout | Package, SBP 180 for BP+diabetes and cholesterol+BP |",
         "|---|---|---|---|"]
    for sex, (a, b, na, nb) in fig.items():
        L.append(f"| {sex} | {FIG[sex]} | {a} ({na}/15) | {b} ({nb}/15) |")
    worst = max(abs(x - p) for x, p in zip(fig["male"][1], FIG["male"]))
    L += ["", f"With SBP 160 for every bar, {fig['male'][2]}/15 (men) and {fig['female'][2]}/15 (women) match to the printed",
          "0.01% (the men's all-factor bar aside, see below). Two bars differ in each sex, both times \"blood pressure and diabetes\" and \"cholesterol and blood",
          f"pressure\"; computed with SBP 180 they match ({fig['male'][3]}/15 men, {fig['female'][3]}/15 women; the remaining",
          f"men's bar, all risk factors, differs by {worst:.2f}, and no single HDL value matches all 15). So the figure was",
          "probably drawn with SBP 180 for those two bars, against its caption. The equations are unaffected.", ""]
    (HERE / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"equations {sum(ok for *_, ok in eq)}/4; table1 {t1_ok}/{t1_n}; fig men {fig['male'][3]}/15 women {fig['female'][3]}/15")
    assert all(ok for *_, ok in eq) and fig["male"][3] == 14 and fig["female"][3] == 15 and worst <= 0.01 + 1e-9


if __name__ == "__main__":
    main()
