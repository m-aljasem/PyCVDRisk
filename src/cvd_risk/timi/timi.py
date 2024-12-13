"""TIMI risk score for unstable angina / non-ST elevation MI (Antman et al., JAMA 2000).

One point for each of the 7 predictors (Table 1, bold rows), total 0-7:
  age >= 65; >= 3 CAD risk factors (family history of CAD, hypertension, hypercholesterolaemia,
  diabetes, current smoking); known coronary stenosis >= 50%; ST deviation on the presenting ECG;
  >= 2 anginal events in the last 24 h; aspirin in the last 7 days; elevated cardiac markers
  (CK-MB and/or cardiac troponin).
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float

SOURCE = ("Antman EM et al. The TIMI risk score for unstable angina/non-ST elevation MI. "
          "JAMA 2000;284:835-42, Table 1 and Results")


def timi(age, risk_factors, coronary_stenosis, st_deviation, anginal_events_24h, aspirin_7d,
         elevated_markers) -> np.ndarray:
    """TIMI UA/NSTEMI score, 0-7 (float64; NaN for invalid rows). Source: SOURCE."""
    def count_at_least(x, k):
        x = as_float(x)
        return np.where((x >= 0) & (x == np.floor(x)), x >= k, np.nan)

    a = as_float(age)
    return (np.where(a >= 0, a >= 65, np.nan) + count_at_least(risk_factors, 3) + as_bool(coronary_stenosis)
            + as_bool(st_deviation) + count_at_least(anginal_events_24h, 2) + as_bool(aspirin_7d)
            + as_bool(elevated_markers))
