"""REGICOR: Framingham hard-CHD function (categorical) recalibrated to Gerona, Spain (Marrugat et al. 2003).

10-year risk of coronary death or non-fatal MI = 1 - S ** exp(sum(b*X) - sum(b*Xmean)), with b from
Table 1 and the Gerona (default) or Framingham means and event rate S = 1 - rate from Table 2
(Appendix). Ages 35-74. Cholesterol in mmol/L, converted to the source's mg/dL; bands are in whole
mg/dL ("200-239" is [200, 240)). The blood-pressure category is the higher of the systolic and
diastolic categories (the convention of Wilson et al. 1998, whose function this is; Marrugat does not
restate it).
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex
from cvd_risk.prevent.prevent import MG_PER_MMOL
from cvd_risk.regicor.coefficients import BETA, MEANS

SOURCE = ("Marrugat J, D'Agostino R, Sullivan L, et al. An adaptation of the Framingham coronary heart disease "
          "risk function to European Mediterranean areas. J Epidemiol Community Health 2003;57:634-8, "
          "Tables 1-2 and Appendix")
CATEGORICAL = ("tc", "hdl", "bp")


def mean_lp(sex: str, calibration: str) -> float:
    """sum(b * Xmean) for one sex and calibration ("gerona" or "framingham")."""
    b, m = BETA[sex], MEANS[(sex, calibration)]
    return (b["age"] * m["age"] + b["age2"] * m["age2"]
            + sum(bi * mi / 100 for k in CATEGORICAL for bi, mi in zip(b[k], m[k]))
            + (b["diabetes"] * m["diabetes"] + b["smoker"] * m["smoker"]) / 100)


def _lp(sex, age, tc_cat, hdl_cat, bp_cat, diabetes, smoker):
    b = BETA[sex]
    return (b["age"] * age + b["age2"] * age**2 + np.take(b["tc"], tc_cat) + np.take(b["hdl"], hdl_cat)
            + np.take(b["bp"], bp_cat) + b["diabetes"] * diabetes + b["smoker"] * smoker)


def regicor(age, sex, total_chol, hdl_chol, sbp, dbp, diabetes, smoker, calibration="gerona") -> np.ndarray:
    """10-year hard-CHD risk (fraction; NaN for invalid rows or ages outside 35-74). Source: SOURCE."""
    if calibration not in ("gerona", "framingham"):
        raise ValueError("calibration must be 'gerona' or 'framingham'")
    a, sx, tc, hdl, s, d, dm, smk = np.broadcast_arrays(
        as_float(age), as_sex(sex), as_float(total_chol) * MG_PER_MMOL, as_float(hdl_chol) * MG_PER_MMOL,
        as_float(sbp), as_float(dbp), as_bool(diabetes), as_bool(smoker))
    tc_cat = np.digitize(tc, [160, 200, 240, 280])
    hdl_cat = np.digitize(hdl, [35, 45, 50, 60])
    bp_cat = np.maximum(np.digitize(s, [120, 130, 140, 160]), np.digitize(d, [80, 85, 90, 100]))
    out = np.full(a.shape, np.nan)
    for code, name in ((1, "male"), (0, "female")):
        surv = 1 - MEANS[(name, calibration)]["event10"]
        lp = _lp(name, a, tc_cat, hdl_cat, bp_cat, dm, smk)
        out = np.where(sx == code, 1 - surv ** np.exp(lp - mean_lp(name, calibration)), out)
    valid = (a >= 35) & (a < 75) & (tc > 0) & (hdl > 0) & (s > 0) & (d > 0)
    return np.where(valid, out, np.nan)  # NaN in dm/smk/sex already propagates
