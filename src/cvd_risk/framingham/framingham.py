"""Framingham general CVD risk profile (2008): 10-year risk of CVD (coronary death, MI,
coronary insufficiency, angina, stroke, TIA, intermittent claudication, heart failure).

Accepts scalars or array-likes (broadcast like numpy) and returns a float64 array of
risks as fractions in [0, 1]. Cholesterol in mmol/L (converted to the source's mg/dL with
the same factor as PREVENT); BMI in kg/m². A row is NaN when an input is missing or
invalid, sex is unrecognised, or age is outside 30-74.

model="lab": total_chol and hdl_chol required.  model="bmi": bmi required (office model).
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex
from cvd_risk.framingham.coefficients import AGE_RANGE, MODELS
from cvd_risk.prevent.prevent import MG_PER_MMOL


def framingham(age, sex, sbp, bp_treated, smoker, diabetes, *, total_chol=None, hdl_chol=None, bmi=None,
               model: str = "lab") -> np.ndarray:
    """Framingham 2008 10-year general CVD risk. Source: coefficients.SOURCE."""
    if model not in MODELS:
        raise ValueError('model must be "lab" or "bmi"')
    if model == "lab" and (total_chol is None or hdl_chol is None):
        raise ValueError('model="lab" needs total_chol and hdl_chol')
    if model == "bmi" and bmi is None:
        raise ValueError('model="bmi" needs bmi')
    tc = as_float(np.nan if total_chol is None else total_chol) * MG_PER_MMOL
    hdl = as_float(np.nan if hdl_chol is None else hdl_chol) * MG_PER_MMOL
    a, sx, sb, trt, sm, dm, tc, hdl, b = np.broadcast_arrays(
        as_float(age), as_sex(sex), as_float(sbp), as_bool(bp_treated), as_bool(smoker), as_bool(diabetes),
        tc, hdl, as_float(np.nan if bmi is None else bmi))
    lo, hi = AGE_RANGE
    ok = (a >= lo) & (a < hi) & (sx >= 0) & (sb > 0)
    ok &= (tc > 0) & (hdl > 0) if model == "lab" else (b > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lsbp = np.log(sb)
        x = {"ln_age": np.log(a), "ln_tc": np.log(tc), "ln_hdl": np.log(hdl), "ln_bmi": np.log(b),
             "ln_sbp_untreated": lsbp * (1 - trt), "ln_sbp_treated": lsbp * trt, "smoker": sm, "diabetes": dm}
        risk = np.full(a.shape, np.nan)
        for sex_name, c in MODELS[model].items():
            lp = sum(v * x[k] for k, v in c.items() if k not in ("s0", "mean"))
            rows = ok & (sx == (sex_name == "male"))
            risk = np.where(rows, -np.expm1(np.exp(lp - c["mean"]) * np.log(c["s0"])), risk)
    return risk
