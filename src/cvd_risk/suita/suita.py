"""Suita score (Nishimura K et al. J Atheroscler Thromb 2014;21:784-98): 10-year coronary heart disease risk for
Japanese adults, from the paper's prediction score sheets (Table 5a, total cholesterol; Table 5b, LDL cholesterol;
both "with CKD").

The paper's Cox models (Tables 3-4) cannot be used directly: their baseline survival and cohort means are not
published. Points follow the sheets as printed; see reference/suita/REPORT.md for where the TC sheet departs from
its own coefficients and band edges.

Band edges on real values: blood pressure by the Framingham (JNC-VI) categories the paper adopts (optimal < 120/80,
normal 120-129/80-84, high normal 130-139/85-89, stage 1 140-159/90-99, stage 2 >= 160/100; the higher of the
systolic and diastolic categories applies); CKD stage 3 eGFR 30-59, stage 4-5 < 30 mL/min/1.73 m2 (K/DOQI, as in
the paper). The sheets' gaps are closed as follows: TC sheet HDL 35 goes with 36-49 (Table 3: 35-50 reference)
and TC 280 with "> 280" (Table 3: "over 280"). Ages: TC sheet 36-79, LDL sheet 35-79 (the cohort was 30-79; the
top age bands are open-ended in the sheets); other ages give NaN.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = "Nishimura K, Okamura T, Watanabe M, et al. J Atheroscler Thromb 2014;21(8):784-798, Table 5"

# (edges, points): points[i] for values in [edges[i-1], edges[i])
SHEETS = {
    "tc": {"age": ((36, 46, 56, 66, 80), (np.nan, 30, 39, 46, 58, np.nan)),
           "female": -6, "smoker": 5, "diabetes": 5, "bp": (-7, 0, 0, 3, 6),
           "chol": ((160, 240, 280), (-11, 0, 5, 9)),
           "hdl": ((35, 50, 60), (6, 0, -5, -5)),
           "ckd": (15, 3, 0)},
    "ldl": {"age": ((35, 45, 55, 65, 70, 80), (np.nan, 30, 38, 45, 51, 53, np.nan)),
            "female": -7, "smoker": 5, "diabetes": 6, "bp": (-7, 0, 0, 4, 6),
            "chol": ((100, 140, 160, 180), (0, 5, 7, 10, 11)),
            "hdl": ((40, 60), (0, -5, -6)),
            "ckd": (14, 3, 0)},
}
# printed 10-year CHD probability (%) by total score: (upper score of each row, label)
RISK = {
    "tc": ((30, "<1"), (35, "1"), (40, "2"), (45, "4"), (50, "6"), (55, "10"), (60, "15"), (np.inf, "24")),
    "ldl": ((35, "<1"), (40, "1"), (45, "2"), (50, "3"), (55, "5"), (60, "9"), (65, "14"), (70, "22"), (np.inf, ">28")),
}


def _band(x, edges, points):
    x = as_float(x)
    return np.where(np.isnan(x) | (x < 0), np.nan, np.asarray(points, float)[np.digitize(np.nan_to_num(x), edges)])


def bp_category(sbp, dbp):
    """0 optimal, 1 normal, 2 high normal, 3 stage 1, 4 stage 2 (the higher of the SBP and DBP categories)."""
    s, d = as_float(sbp), as_float(dbp)
    c = np.maximum(np.digitize(np.nan_to_num(s), (120, 130, 140, 160)), np.digitize(np.nan_to_num(d), (80, 85, 90, 100)))
    return np.where((s > 0) & (d > 0), c, np.nan)


def suita(age, sex, smoker, diabetes, sbp, dbp, hdl, egfr, *, tc=None, ldl=None) -> np.ndarray:
    """Suita score points (10-year CHD), from the TC sheet (tc=) or the LDL sheet (ldl=); exactly one.
    Cholesterol in mg/dL, eGFR in mL/min/1.73 m2; smoker = current smoker; flags 0/1."""
    if (tc is None) == (ldl is None):
        raise ValueError("give exactly one of tc= (Table 5a) or ldl= (Table 5b)")
    s = SHEETS["tc" if tc is not None else "ldl"]
    sx = as_sex(sex)
    bp = bp_category(sbp, dbp)
    pts = (_band(age, *s["age"]) + np.where(sx == 0, s["female"], np.where(sx == 1, 0, np.nan))
           + s["smoker"] * as_bool(smoker) + s["diabetes"] * as_bool(diabetes)
           + np.asarray(s["bp"], float)[np.nan_to_num(bp).astype(int)] + 0 * bp
           + _band(tc if tc is not None else ldl, *s["chol"]) + _band(hdl, *s["hdl"])
           + _band(egfr, (30, 60), s["ckd"]))
    return pts


def suita_risk(points, sheet) -> np.ndarray:
    """The sheet's printed 10-year CHD probability (%) for a total score, as its label ("<1", "4", ">28", ...);
    sheet is "tc" or "ldl". NaN points give ""."""
    rows = RISK[sheet]
    p = as_float(points)
    labels = np.array([lab for _, lab in rows] + [""], dtype=object)
    i = np.searchsorted([hi for hi, _ in rows], np.nan_to_num(p, nan=np.inf), side="left")
    return np.where(np.isnan(p), "", labels[i])
