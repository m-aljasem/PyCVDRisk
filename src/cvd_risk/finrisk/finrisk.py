"""FINRISK calculator (Vartiainen et al. 2016): 10-year risk of an acute coronary event and of stroke.

Logistic models from Table 2: risk = 1 / (1 + exp(-(b0 + sum(b*x)))). CVD risk is defined in the paper as
CHD risk + stroke risk. Cholesterol in mmol/L; the stroke models have no total-cholesterol term. Ages
30-64 (the cohorts' range). Family history is a parent's myocardial infarction (CHD model) or stroke
(stroke model).
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = ("Vartiainen E, Laatikainen T, Peltonen M, Puska P. Predicting coronary heart disease and stroke: "
          "the FINRISK calculator. Global Heart 2016;11:213-6, Table 2")
# intercept, age, smoking, total cholesterol, SBP, HDL, diabetes, parent history (signs as in the linear predictor)
BETA = {
    ("male", "chd"): (-11.213, 0.0802, 0.6260, 0.3293, 0.0166, -0.5893, 0.7417, 0.3138),
    ("male", "stroke"): (-11.6994, 0.1153, 0.4981, 0.0, 0.0149, -0.4406, 0.879, 0.2933),
    ("female", "chd"): (-11.839, 0.0962, 0.8776, 0.2119, 0.0175, -1.1009, 1.0303, 0.4090),
    ("female", "stroke"): (-7.966, 0.0633, 0.4163, 0.0, 0.00893, -0.7636, 1.2383, 0.547),
}
OUTCOMES = ("cvd", "chd", "stroke")


def _logistic(b, age, smoker, chol, sbp, hdl, diabetes, parent):
    lp = b[0] + b[1] * age + b[2] * smoker + b[3] * chol + b[4] * sbp + b[5] * hdl + b[6] * diabetes + b[7] * parent
    return 1 / (1 + np.exp(-lp))


def finrisk(age, sex, smoker, total_chol, hdl_chol, sbp, diabetes, parent_mi, parent_stroke, outcome="cvd") -> np.ndarray:
    """10-year risk (fraction) of "chd", "stroke" or "cvd" (their sum, as defined in Table 2). Source: SOURCE."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    a, tc, hdl, s = as_float(age), as_float(total_chol), as_float(hdl_chol), as_float(sbp)
    a = np.where((a >= 30) & (a < 65), a, np.nan)
    tc, hdl, s = (np.where(v > 0, v, np.nan) for v in (tc, hdl, s))
    smk, dm, pmi, pst = as_bool(smoker), as_bool(diabetes), as_bool(parent_mi), as_bool(parent_stroke)
    sx = as_sex(sex)
    out = np.nan
    for code, name in ((1, "male"), (0, "female")):
        chd = _logistic(BETA[(name, "chd")], a, smk, tc, s, hdl, dm, pmi)
        stroke = _logistic(BETA[(name, "stroke")], a, smk, tc, s, hdl, dm, pst)
        r = {"chd": chd, "stroke": stroke, "cvd": chd + stroke}[outcome]
        # the stroke model ignores cholesterol; keep the row NaN if it is invalid anyway
        out = np.where(sx == code, np.where(np.isnan(tc), np.nan, r), out)
    return out
