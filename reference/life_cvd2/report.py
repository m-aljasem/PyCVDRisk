"""LIFE-CVD2 verification against the authors' published charts (Appendix 1 of the supplement,
source/hageman2024_appendix1_charts.pdf): 2048 lifetime-risk cells and 7680 treatment-benefit
cells (median CVD-free life expectancy gained), 4 ESC risk regions each. Writes REPORT.md.
Run from repo root: python reference/life_cvd2/report.py
"""

from __future__ import annotations

import copy
import datetime
import importlib
import re
from pathlib import Path

import numpy as np
import pypdf

HERE = Path(__file__).parent
C = importlib.import_module("cvd_risk.life_cvd2.coefficients")
M = importlib.import_module("cvd_risk.life_cvd2.life_cvd2")
REGIONS = ("low", "moderate", "high", "very_high")
HDL = {"female": 1.4, "male": 1.2}  # chart assumption, fitted (section 3)
LDL_OFFSET = 0.8  # chart LDL = non-HDL - 0.8 mmol/L, fitted (section 3)
CELL = r"(?: (?:\d+(?:\.\d)?%?|-))"
FAMILIES = {  # first page, ages (top block first)
    "lifetime": (2, list(range(75, 39, -5))),
    "sbp": (6, list(range(85, 39, -5))), "ldl": (10, list(range(85, 39, -5))), "smoking": (14, list(range(85, 39, -5))),
}
DEFECT = (65, 110)  # (age band start, SBP band midpoint) of the row computed at another age


def charts():
    """{(family, region): array (rows, 16) of printed values, NaN for '-'}."""
    pages = [p.extract_text() for p in pypdf.PdfReader(HERE / "source" / "hageman2024_appendix1_charts.pdf").pages]
    out = {}
    for fam, (first, ages) in FAMILIES.items():
        for j, reg in enumerate(REGIONS):
            rows = [ln.split()[1:] for ln in pages[first - 1 + j].splitlines()
                    if re.fullmatch(rf"\d{{3}}-\d{{3}}{CELL}{{16}}", ln.strip())]
            assert len(rows) == 4 * len(ages), (fam, reg, len(rows))
            out[fam, reg] = np.array([[np.nan if v == "-" else float(v.rstrip("%")) for v in r] for r in rows])
    return out


def grid(fam, defect_age=None):
    """Chart cell inputs, broadcast to (rows, 16): band midpoints, women then men,
    non-smoking then smoking, non-HDL 3.5-6.5."""
    ages = np.repeat(np.array(FAMILIES[fam][1]) + 2, 4)[:, None]
    sbp = np.tile([170, 150, 130, 110], len(ages) // 4)[:, None]
    if defect_age is not None:
        ages = np.where((ages == DEFECT[0] + 2) & (sbp == DEFECT[1]), defect_age, ages)
    c = np.arange(16)
    sex = np.where(c < 8, "female", "male")
    hdl = np.where(c < 8, HDL["female"], HDL["male"])
    nonhdl = 3.5 + c % 4
    return {"age": ages, "sex": sex, "smoker": (c // 4) % 2, "sbp": sbp, "total_chol": nonhdl + hdl, "hdl_chol": hdl}, nonhdl


def predict(fam, reg, f, defect_age=None):
    x, nonhdl = grid(fam, defect_age)
    if fam == "lifetime":
        return 100 * f(**x, region=reg)
    le = lambda **kw: f(**x, region=reg, outcome="life_expectancy", **kw)
    if fam == "sbp":
        hr = {"hr_cvd": 0.8 ** (np.maximum(x["sbp"] - 135, 0) / 10)}
    elif fam == "ldl":
        hr = {"hr_cvd": 0.78 ** (0.4 * (nonhdl - LDL_OFFSET))}
    else:
        hr = {"hr_cvd": np.where(x["smoker"], 0.6, 1.0), "hr_non_cvd": np.where(x["smoker"], 0.73, 1.0)}
    return le(**hr) - le()


def compare(f=M.life_cvd2, defect_age=None, outliers=None, worst=None):
    """Per family: (cells, exact after rounding, within half a printed unit + 1e-9, dash cells consistent)."""
    res, worst = {}, worst if worst is not None else [0.0]
    for (fam, reg), printed in CH.items():
        p = predict(fam, reg, f, defect_age)
        unit = 1.0 if fam == "lifetime" else 0.1
        num = ~np.isnan(printed)
        treated = p != 0 if fam != "lifetime" else np.ones_like(num)
        cells = num & treated
        exact = np.round(p / unit) == np.round(printed / unit)
        near = np.abs(p - printed) <= unit / 2 + 0.01 * unit
        dash = (~treated) & (np.isnan(printed) | (printed == 0))
        r = res.setdefault(fam, np.zeros(4, int))
        r += [cells.sum(), (exact & cells).sum(), (near & cells).sum(), dash.sum()]
        if outliers is not None:
            x, _ = grid(fam, defect_age)
            for i, j in zip(*np.nonzero(cells & ~near)):
                outliers.append((fam, reg, int(np.ravel(x["age"])[i]), x["sex"][j], int(x["smoker"][j]),
                                 int(np.ravel(x["sbp"])[i]), 3.5 + j % 4, printed[i, j], p[i, j]))
            d = np.abs(p - printed)[cells] / unit
            worst[0] = max(worst[0], d.max())
    return res


def with_coefficients(edit):
    """life_cvd2 evaluated with a modified copy of the coefficient tables."""
    beta, scales, s0 = copy.deepcopy(C.BETA), copy.deepcopy(C.SCALES), M._S0.copy()
    edit(beta, scales, s0)
    saved = M._TABLES, M._S0
    tables = {o: (np.array([beta[s][o] for s in M._SEXES]),
                  np.array([[scales[o][r][s] for r in REGIONS] for s in M._SEXES])) for o in ("cvd", "non_cvd")}

    def f(**kw):
        M._TABLES, M._S0 = tables, s0
        try:
            return M.life_cvd2(**kw)
        finally:
            M._TABLES, M._S0 = saved
    return f


def total(res):
    return sum(v[2] for v in res.values())


def sensitivity(base):
    """Move each printed coefficient and scale by one unit of its last printed digit."""
    caught = n = 0
    def ulp(v):
        s = repr(v)
        return 10.0 ** -(len(s.split(".")[1]) if "." in s else 0)
    for sx in ("male", "female"):
        for o in ("cvd", "non_cvd"):
            for k, v in enumerate(C.BETA[sx][o]):
                for d in (ulp(v), -ulp(v)):
                    def e(b, s, s0, sx=sx, o=o, k=k, d=d):
                        t = list(b[sx][o]); t[k] += d; b[sx][o] = tuple(t)
                    n += 1; caught += total(compare(with_coefficients(e))) < base
            for r in REGIONS:
                for k in range(2):
                    v = C.SCALES[o][r][sx][k]
                    for d in (ulp(v), -ulp(v)):
                        def e(b, s, s0, sx=sx, o=o, r=r, k=k, d=d):
                            t = list(s[o][r][sx]); t[k] += d; s[o][r][sx] = tuple(t)
                        n += 1; caught += total(compare(with_coefficients(e))) < base
    return caught, n


def conventions():
    """Agreement (cells within half a printed unit) under the alternatives the paper leaves open."""
    global HDL, LDL_OFFSET
    rows = []
    for label, hdl in (("HDL 1.3 both sexes", {"female": 1.3, "male": 1.3}),
                       ("HDL 1.5 women / 1.3 men", {"female": 1.5, "male": 1.3})):
        saved, HDL = HDL, hdl
        rows.append((label, total(compare(defect_age=62))))
        HDL = saved
    for off in (0.6, 1.0):
        saved, LDL_OFFSET = LDL_OFFSET, off
        rows.append((f"LDL = non-HDL - {off}", total(compare(defect_age=62))))
        LDL_OFFSET = saved

    rows.append(("age terms fixed at entry age, not attained age", total(compare(_fixed_age_model, defect_age=62))))
    rows.append(("chart row 65-69 / SBP 100-119 at age 67 (its band midpoint)", total(compare(defect_age=67))))
    return rows


def _fixed_age_model(**kw):
    """The same life table with the age terms frozen at the entry age (one call per entry age)."""
    a = np.broadcast_to(kw["age"], np.broadcast_shapes(*(np.shape(v) for v in kw.values() if not isinstance(v, str))))
    out = np.full(a.shape, np.nan)
    saved = M._T
    for age in np.unique(a):
        sel = a == age
        M._T = np.full_like(saved, (age - 60) / 5.0)
        try:
            out[sel] = M.life_cvd2(**{k: v if isinstance(v, str) else np.broadcast_to(v, a.shape)[sel] for k, v in kw.items()})
        finally:
            M._T = saved
    return out


CH = charts()


def main():
    outliers, worst = [], [0.0]
    res = compare(defect_age=62, outliers=outliers, worst=worst)
    base = total(res)
    caught, n = sensitivity(base)
    raw67 = compare(defect_age=67)
    names = {"lifetime": "Lifetime risk to 80 (%)", "sbp": "Gain from SBP lowering to 135 mmHg (years)",
             "ldl": "Gain from 40% LDL lowering (years)", "smoking": "Gain from smoking cessation (years)"}
    lines = [
        "# LIFE-CVD2 — verification report", "",
        f"Generated {datetime.date.today()} by `reference/life_cvd2/report.py`.", "",
        "## 1. Source", "",
        "Hageman et al., Eur J Prev Cardiol 2024;31:1690-1699, and its online supplement, saved in",
        "`source/`. `check_coefficients.py` parses Supplementary Tables 3-5 from the PDF: all 36",
        "coefficients, 264 baseline survivals (ages 35-100) and 32 recalibration scales match.", "",
        "The supplement does not print how predictors are centred or the recalibration formula; it says",
        "both follow SCORE2. We use SCORE2's centring (age 60 per 5 years, SBP 120 per 20 mmHg, total",
        "cholesterol 6, HDL 1.3 per 0.5) and recalibration, 1 - exp(-exp(scale1 + scale2 ln(-ln(1 - risk)))),",
        "with the age terms evaluated at each attained age of the life table. Section 2 shows this",
        "reproduces the authors' own charts.", "",
        "## 2. The authors' charts (evidence level A)", "",
        "Appendix 1 of the supplement prints 16 charts: lifetime risk to 80, and CVD-free life years",
        "gained from lifelong SBP lowering to 135 mmHg (HR 0.80 per 10 mmHg), 40% LDL lowering",
        "(HR 0.78 per mmol/L) and smoking cessation (HR 0.60 CVD, 0.73 non-CVD), each for the four ESC",
        "risk regions. Cells are read from the PDF text layer (no transcription). Inputs are the band",
        "midpoints: age (e.g. 77 for 75-79), SBP 110/130/150/170, non-HDL 3.5/4.5/5.5/6.5 mmol/L.", "",
        "| Chart family | Cells | Exact after rounding | Within half a printed unit | Untreated cells shown as - |",
        "|---|---|---|---|---|",
    ]
    for fam, (cells, exact, near, dash) in res.items():
        lines.append(f"| {names[fam]} | {cells} | {exact} | {near} | {dash if fam in ('sbp', 'smoking') else 'n/a'} |")
    allc = sum(v[0] for v in res.values())
    lines += [
        "",
        f"**{base}/{allc} cells agree within half a printed unit** (plus 1% of a unit for ties); the largest",
        f"difference is {worst[0]:.3f} units. Cells outside: "
        + ("; ".join(f"{f} {r}, {sx} age {a}, smoker {sm}, SBP {b}, non-HDL {n}: printed {pr:g}, ours {v:.4f}"
                     for f, r, a, sx, sm, b, n, pr, v in outliers) or "none") + ".", "",
        "Two chart inputs are not stated in the paper and were found by fitting, and one chart row is defective:", "",
        "- **HDL in the charts is 1.4 mmol/L for women and 1.2 for men** (total cholesterol = non-HDL + HDL).",
        "- **LDL in the LDL-lowering charts is non-HDL - 0.8 mmol/L.**",
        "- **Chart defect: the row \"age 65-69, SBP 100-119\" is computed at age 62**, the midpoint of the",
        "  60-64 band, in all 16 charts. At age 67 the lifetime-risk row is higher than the SBP 120-139 row",
        "  above it. At age 62 the row is reproduced like every other. With the row taken at 67, agreement",
        f"  falls to {total(raw67)}/{allc}.", "",
        "## 3. How tightly the charts pin the model", "",
        "| Alternative | Cells within half a unit |", "|---|---|",
        f"| As implemented | {base} |",
    ]
    for label, v in conventions():
        lines.append(f"| {label} | {v} |")
    lines += [
        "",
        "Moving any one of the 36 coefficients or 32 recalibration scales by one unit of its last printed",
        f"digit pushes at least one chart cell outside half a printed unit in {caught}/{n} cases.", "",
        "## 4. Reference implementations", "",
        "The official calculator is the web tool at u-prevent.com, which cannot be queried offline. No",
        "open-source implementation was found.", "",
        "## 5. Scope", "",
        "- People without established CVD. The diabetes terms in Table 3 are left out, as the supplement",
        "  instructs.",
        "- Whole-year ages: 35-79 for lifetime risk, 35-99 for life expectancy. Life expectancy is the",
        "  first crossing of 0.5 by event-free survival, linear within the year. When survival is still",
        "  >= 0.5 at 101 it is inf; the supplement's last-survival rule for benefit beyond 100 is not",
        "  implemented.",
        "- Treatment effects enter as lifelong hazard ratios (`hr_cvd`, `hr_non_cvd`), applied to each",
        "  yearly risk after recalibration, as in the supplement.",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
