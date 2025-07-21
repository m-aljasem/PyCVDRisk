"""FINRISKI 2.0 calculator (Vartiainen et al. 2020): 10-year risk of an acute coronary event and of stroke.

Logistic models from Table 1: risk = 1 / (1 + exp(-(b0 + sum(b*x)))). Combined (CVD) risk is
1 - (1 - CHD risk)(1 - stroke risk). Cholesterol in mmol/L; the stroke models have no total-cholesterol
term. Ages 30-74 (the cohorts' range). Family history is a parent's myocardial infarction (CHD model) or
stroke (stroke model). This is the model of the official Finnish calculator (Duodecim/THL); it replaces
the 2016 version (Global Heart 2016;11:213, ages 30-64).
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = ("Vartiainen E, Laatikainen T, Jousilahti P, Peltonen M, Niiranen T, Salomaa V. Sepelvaltimotaudin ja "
          "aivohalvauksen riskin arviointi FINRISKI 2.0 -laskurilla. Suomen Lääkärilehti 2020;75:2778-84, Table 1")
# intercept, age, smoking, total cholesterol, SBP, HDL, diabetes, parent history (signs as in the linear predictor)
BETA = {
    ("male", "chd"): (-9.081, 0.075, 0.579, 0.320, 0.011, -1.082, 0.729, 0.338),
    ("male", "stroke"): (-9.928, 0.083, 0.369, 0.0, 0.014, -0.329, 0.705, 0.249),
    ("female", "chd"): (-11.250, 0.095, 0.639, 0.244, 0.013, -0.845, 1.315, 0.421),
    ("female", "stroke"): (-9.553, 0.085, 0.613, 0.0, 0.012, -0.623, 0.914, 0.023),
}
OUTCOMES = ("cvd", "chd", "stroke")


def _logistic(b, age, smoker, chol, sbp, hdl, diabetes, parent):
    lp = b[0] + b[1] * age + b[2] * smoker + b[3] * chol + b[4] * sbp + b[5] * hdl + b[6] * diabetes + b[7] * parent
    return 1 / (1 + np.exp(-lp))


def finrisk(age, sex, smoker, total_chol, hdl_chol, sbp, diabetes, parent_mi, parent_stroke, outcome="cvd") -> np.ndarray:
    """10-year risk (fraction) of "chd", "stroke" or "cvd" (1 - (1 - chd)(1 - stroke), Table 1). Source: SOURCE."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    a, tc, hdl, s = as_float(age), as_float(total_chol), as_float(hdl_chol), as_float(sbp)
    a = np.where((a >= 30) & (a < 75), a, np.nan)
    tc, hdl, s = (np.where(v > 0, v, np.nan) for v in (tc, hdl, s))
    smk, dm, pmi, pst = as_bool(smoker), as_bool(diabetes), as_bool(parent_mi), as_bool(parent_stroke)
    sx = as_sex(sex)
    out = np.nan
    for code, name in ((1, "male"), (0, "female")):
        chd = _logistic(BETA[(name, "chd")], a, smk, tc, s, hdl, dm, pmi)
        stroke = _logistic(BETA[(name, "stroke")], a, smk, tc, s, hdl, dm, pst)
        r = {"chd": chd, "stroke": stroke, "cvd": 1 - (1 - chd) * (1 - stroke)}[outcome]
        # the stroke model ignores cholesterol; keep the row NaN if it is invalid anyway
        out = np.where(sx == code, np.where(np.isnan(tc), np.nan, r), out)
    return out
