"""Suita CVD risk model (Japan, 2020): points for the 10-year risk of a first cardiovascular event (stroke or
coronary heart disease), urban Japanese adults aged 30-79 without CVD.

Points and probabilities as printed in Nakai M, Watanabe M, Kokubo Y, et al. J Atheroscler Thromb 2020;27(11):
1160-75, Table 3 (score = 10 x Cox coefficient, rounded) and Table 4 (10-year probability by score band). Two models:
with ECG data (adds atrial fibrillation and left ventricular hypertrophy) and without; give both af and lvh for the
first, neither for the second.

Categories as in Table 3: blood pressure < 120 and < 80, 120-139 or 80-89, 140-159 or 90-99, >= 160 or >= 100 or on
antihypertensive medication; lipids high when non-HDL-C >= 170 or LDL-C >= 140 mg/dL; where SBP and DBP (or LDL-C
and non-HDL-C) fall in different categories the higher one applies (Table 3 footnote). Either lipid may be missing,
not both. Ages outside 30-79 give NaN.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = "Nakai M, Watanabe M, Kokubo Y, et al. J Atheroscler Thromb 2020;27(11):1160-1175, Tables 3-4"

# Table 3 scores: (with ECG, without ECG)
AGE = ((30, 40, 50, 60, 65, 70, 75, 80), ((0, 8, 14, 18, 22, 25, 28), (0, 8, 14, 18, 22, 26, 29)))
POINTS = {"men": (4, 4), "bp": ((-4, 0, 3, 6), (-4, 0, 3, 6)), "lipid_high": (2, 2), "hdl": ((0, -2, -4), (0, -2, -4)),
          "smoker": (4, 4), "diabetes": (6, 7), "proteinuria": (2, 2), "af": (8, None), "lvh": (5, None)}
# Table 4: (upper score of each band, probability label with ECG, without ECG)
RISK = ((0, "<1", "<1"), (20, "2", "2"), (25, "6", "6"), (30, "9", "9"), (35, "15", "14"), (np.inf, "26", "25"))
S10 = (0.99411, 0.99389)  # Table 4 baseline survival at 10 years (not used by the printed sheet)


def _model(af, lvh):
    if (af is None) != (lvh is None):
        raise ValueError("give both af and lvh (model with ECG) or neither (model without ECG)")
    return 0 if af is not None else 1


def suita_cvd(age, sex, sbp, dbp, bp_treated, hdl, smoker, diabetes, proteinuria, *, ldl=None, non_hdl=None,
              af=None, lvh=None) -> np.ndarray:
    """Suita CVD score points (Table 3). Cholesterol in mg/dL; smoker = current smoker; proteinuria = urinary protein
    1+ or more; af, lvh from the ECG (both or neither); flags 0/1."""
    if ldl is None and non_hdl is None:
        raise ValueError("give ldl=, non_hdl= or both")
    m = _model(af, lvh)
    a = as_float(age)
    i = np.digitize(np.nan_to_num(a, nan=-1), AGE[0]) - 1
    age_pts = np.where((a >= 30) & (a < 80), np.asarray(AGE[1][m], float)[np.clip(i, 0, 6)], np.nan)
    s, d = as_float(sbp), as_float(dbp)
    bp = np.maximum(np.digitize(np.nan_to_num(s), (120, 140, 160)), np.digitize(np.nan_to_num(d), (80, 90, 100)))
    treated = as_bool(bp_treated)
    bp = np.where(treated == 1, 3, bp)
    bp_pts = np.where(np.isnan(s) | np.isnan(d) | np.isnan(treated), np.nan, np.asarray(POINTS["bp"][m], float)[bp])
    lipids = [(as_float(x), cut) for x, cut in ((ldl, 140), (non_hdl, 170)) if x is not None]
    high = sum(v >= cut for v, cut in lipids) > 0
    known = sum(~np.isnan(v) for v, _ in lipids) > 0
    lipid_pts = np.where(known, POINTS["lipid_high"][m] * high, np.nan)
    h = as_float(hdl)
    hdl_pts = np.where(np.isnan(h), np.nan, np.asarray(POINTS["hdl"][m], float)[np.digitize(np.nan_to_num(h), (40, 60))])
    sx = as_sex(sex)
    pts = (age_pts + np.where(sx == 1, POINTS["men"][m], np.where(sx == 0, 0, np.nan)) + bp_pts + lipid_pts + hdl_pts
           + POINTS["smoker"][m] * as_bool(smoker) + POINTS["diabetes"][m] * as_bool(diabetes)
           + POINTS["proteinuria"][m] * as_bool(proteinuria))
    if m == 0:
        pts = pts + POINTS["af"][0] * as_bool(af) + POINTS["lvh"][0] * as_bool(lvh)
    return pts


def suita_cvd_risk(points, ecg) -> np.ndarray:
    """Table 4's printed 10-year CVD probability (%) for a total score, as its label ("<1", "2", ..., "26");
    ecg: True for the model with ECG. NaN points give ""."""
    p = as_float(points)
    labels = np.array([r[1 if ecg else 2] for r in RISK] + [""], dtype=object)
    i = np.searchsorted([r[0] for r in RISK], np.nan_to_num(p, nan=np.inf), side="left")
    return np.where(np.isnan(p), "", labels[i])
