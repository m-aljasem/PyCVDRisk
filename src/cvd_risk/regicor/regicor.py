"""REGICOR: the Framingham coronary function (Wilson et al. 1998, categorical, total cholesterol)
calibrated to Girona, Spain (Marrugat et al., Rev Esp Cardiol 2003), the function of the official
Framingham-REGICOR charts.

10-year risk of a coronary event (angina, fatal or non-fatal MI) = 1 - S0 ** exp(sum(b*X) - sum(b*Xmean)),
b from Wilson Table 6; calibration "gerona" (default: Girona prevalences and S0, Rev Esp Cardiol
Table 1 and Appendix) or "framingham" (Wilson's own G and S0). Ages 35-74. Cholesterol in mmol/L, converted to the source's mg/dL; bands are in whole
mg/dL ("200-239" is [200, 240)). The blood-pressure category is the higher of the systolic and
diastolic categories (Wilson et al. 1998). Replaces the J Epidemiol Community Health 2003 version
(coefficients printed to 2 decimals), which does not reproduce the official charts.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex
from cvd_risk.prevent.prevent import MG_PER_MMOL
from cvd_risk.regicor.coefficients import BETA, G_FRAMINGHAM, GIRONA, S0

SOURCE = ("Marrugat J, Solanas P, D'Agostino R, et al. Estimación del riesgo coronario en España mediante la "
          "ecuación de Framingham calibrada. Rev Esp Cardiol 2003;56:253-61, Table 1 and Appendix; coefficients "
          "from Wilson PWF, D'Agostino RB, Levy D, et al. Prediction of coronary heart disease using risk factor "
          "categories. Circulation 1998;97:1837-47, Table 6")
CATEGORICAL = ("tc", "hdl", "bp")


def mean_lp(sex: str, calibration: str) -> float:
    """sum(b * Xmean) for one sex and calibration ("gerona" or "framingham")."""
    if calibration == "framingham":
        return G_FRAMINGHAM[sex]
    b, m = BETA[sex], GIRONA[sex]
    return (b["age"] * m["age"] + b["age2"] * m["age2"]
            + sum(bi * mi / 100 for k in CATEGORICAL for bi, mi in zip(b[k], m[k]))
            + (b["diabetes"] * m["diabetes"] + b["smoker"] * m["smoker"]) / 100)


def _lp(sex, age, tc_cat, hdl_cat, bp_cat, diabetes, smoker):
    b = BETA[sex]
    return (b["age"] * age + b["age2"] * age**2 + np.take(b["tc"], tc_cat) + np.take(b["hdl"], hdl_cat)
            + np.take(b["bp"], bp_cat) + b["diabetes"] * diabetes + b["smoker"] * smoker)


def regicor(age, sex, total_chol, hdl_chol, sbp, dbp, diabetes, smoker, calibration="gerona") -> np.ndarray:
    """10-year coronary risk (fraction; NaN for invalid rows or ages outside 35-74). Source: SOURCE."""
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
        surv = S0[(name, calibration)]
        lp = _lp(name, a, tc_cat, hdl_cat, bp_cat, dm, smk)
        out = np.where(sx == code, 1 - surv ** np.exp(lp - mean_lp(name, calibration)), out)
    valid = (a >= 35) & (a < 75) & (tc > 0) & (hdl > 0) & (s > 0) & (d > 0)
    return np.where(valid, out, np.nan)  # NaN in dm/smk/sex already propagates
