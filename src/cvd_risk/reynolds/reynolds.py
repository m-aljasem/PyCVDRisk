"""Reynolds Risk Score: 10-year risk of MI, stroke, coronary revascularisation or cardiovascular death.

Accepts scalars or array-likes (broadcast like numpy); returns risks as fractions. Cholesterol in
mmol/L (converted to the source's mg/dL as in PREVENT), hsCRP in mg/L, HbA1c in %. hba1c is needed
only for women with diabetes. A row is NaN when an input is missing or invalid, or for a man with
diabetes (the men's model was derived in non-diabetic men). The papers state no age range
(cohorts: women >= 45, men >= 50); none is enforced.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex
from cvd_risk.prevent.prevent import MG_PER_MMOL
from cvd_risk.reynolds.coefficients import MEN, WOMEN


def reynolds(age, sex, sbp, total_chol, hdl_chol, hscrp, smoker, parental_history, diabetes=0,
             hba1c=np.nan) -> np.ndarray:
    """Reynolds Risk Score (model B), 10-year risk. Source: coefficients.SOURCE."""
    a, sx, sb, tc, hdl, crp, sm, fh, dm, a1c = np.broadcast_arrays(
        as_float(age), as_sex(sex), as_float(sbp), as_float(total_chol) * MG_PER_MMOL,
        as_float(hdl_chol) * MG_PER_MMOL, as_float(hscrp), as_bool(smoker), as_bool(parental_history),
        as_bool(diabetes), as_float(hba1c))
    with np.errstate(invalid="ignore", divide="ignore"):
        w, m = WOMEN, MEN
        bw = (w["age"] * a + w["ln_sbp"] * np.log(sb) + w["ln_hscrp"] * np.log(crp) + w["ln_tc"] * np.log(tc)
              + w["ln_hdl"] * np.log(hdl) + w["hba1c_if_diabetic"] * np.where(dm == 1, a1c, 0 * dm)
              + w["smoker"] * sm + w["parental_history"] * fh)
        bm = (m["ln_age"] * np.log(a) + m["ln_sbp"] * np.log(sb) + m["ln_tc"] * np.log(tc)
              + m["ln_hdl"] * np.log(hdl) + m["smoker"] * sm + m["ln_hscrp"] * np.log(crp)
              + m["parental_history"] * fh + np.where(dm == 0, 0.0, np.nan))
        risk_w = -np.expm1(np.exp(bw - w["mean"]) * np.log(w["s0"]))
        risk_m = -np.expm1(np.exp(bm - m["mean"]) * np.log(m["s0"]))
    ok = (a > 0) & (sb > 0) & (tc > 0) & (hdl > 0) & (crp > 0)
    return np.where(ok, np.select([sx == 1, sx == 0], [risk_m, risk_w], np.nan), np.nan)
