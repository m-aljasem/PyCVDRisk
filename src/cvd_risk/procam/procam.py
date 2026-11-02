"""PROCAM score (Assmann et al. 2002): points for the 10-year risk of an acute coronary event
(fatal or non-fatal MI, sudden cardiac death) in men aged 35-65.

Points from Table 3; RISK_PERCENT is Table 4 (points -> 10-year risk, %; the table prints
<= 20 as "< 1.0" and >= 60 as ">= 30.0", so those totals have no point value there).
Cholesterol and triglycerides are in mmol/L, converted to the source's mg/dL (cholesterol with
PREVENT's factor, triglycerides x 88.57). Bands are in whole source units: "100-129" is [100, 130),
age "60-65" is [60, 66). Women give NaN: the score was derived in men only.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex
from cvd_risk.prevent.prevent import MG_PER_MMOL

SOURCE = ("Assmann G, Cullen P, Schulte H. Simple scoring scheme for calculating the risk of acute coronary "
          "events based on the 10-year follow-up of the PROCAM study. Circulation 2002;105:310-5, Tables 3-4")
TG_MG_PER_MMOL = 88.57
RISK_PERCENT = dict(zip(range(21, 60), (
    1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9, 2.3, 2.4, 2.8, 2.9, 3.3, 3.5, 4.0, 4.2, 4.8, 5.1, 5.7, 6.1,
    7.0, 7.4, 8.0, 8.8, 10.2, 10.5, 10.7, 12.8, 13.2, 15.5, 16.8, 17.5, 19.6, 21.7, 22.2, 23.8, 25.1, 28.0, 29.4)))


def _band(x, edges, points):
    x = np.asarray(x, float)
    return np.where(x >= 0, np.asarray(points, float)[np.digitize(x, edges)], np.nan)


def procam(age, sex, ldl_chol, hdl_chol, triglycerides, sbp, smoker, diabetes, family_history) -> np.ndarray:
    """PROCAM score in points (Table 3). Source: SOURCE."""
    a = as_float(age)
    age_pts = np.where((a >= 35) & (a < 66), _band(a, [40, 45, 50, 55, 60], [0, 6, 11, 16, 21, 26]), np.nan)
    male = np.where(as_sex(sex) == 1, 0.0, np.nan)
    return (male + age_pts
            + _band(as_float(ldl_chol) * MG_PER_MMOL, [100, 130, 160, 190], [0, 5, 10, 14, 20])
            + _band(as_float(hdl_chol) * MG_PER_MMOL, [35, 45, 55], [11, 8, 5, 0])
            + _band(as_float(triglycerides) * TG_MG_PER_MMOL, [100, 150, 200], [0, 2, 3, 4])
            + 8 * as_bool(smoker) + 6 * as_bool(diabetes) + 4 * as_bool(family_history)
            + _band(as_float(sbp), [120, 130, 140, 160], [0, 2, 3, 5, 8]))
