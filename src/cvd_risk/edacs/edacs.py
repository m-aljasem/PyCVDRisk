"""Emergency Department Assessment of Chest pain Score (EDACS; Than et al. 2014, Box 3).

Age in completed years: 18-45 +2, 46-50 +4, then +2 per 5 years to 86+ +20. Male +6. Aged 18-50 with
known CAD (previous MI, CABG or PCI) or >= 3 risk factors (family history of premature CAD,
dyslipidaemia, diabetes, hypertension, current smoking) +4. Diaphoresis +3; pain radiating to arm or
shoulder +5; pain occurring or worsening with inspiration -4; pain reproduced by palpation -6.
EDACS-ADP: low risk if EDACS < 16, no new ischaemia on ECG and 0 and 2 h troponins negative.
Defined for adults only: ages under 18 give NaN.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = ("Than M et al. Development and validation of the Emergency Department Assessment of Chest pain "
          "Score and 2 h accelerated diagnostic protocol. Emerg Med Australas 2014;26:34-44, Box 3")


def edacs(age, sex, known_cad, risk_factors, diaphoresis, radiates_arm_shoulder, pleuritic,
          palpation) -> np.ndarray:
    """EDACS (float64; NaN for invalid rows). Source: SOURCE."""
    a = np.floor(as_float(age))  # bands are in completed years
    n = as_float(risk_factors)
    n = np.where((n >= 0) & (n == np.floor(n)), n, np.nan)
    cad = as_bool(known_cad)
    sx = as_sex(sex)
    age_pts = np.where(a >= 18, np.clip(2 * np.ceil((a - 40) / 5), 2, 20), np.nan)  # 45 -> 2, 46 -> 4, 86 -> 20
    young = np.where(a <= 50, ((cad == 1) | (n >= 3)) * 4.0, 0.0)
    young = np.where(np.isnan(cad) | np.isnan(n), np.nan, young)
    return (age_pts + np.where(sx >= 0, 6.0 * sx, np.nan) + young + 3 * as_bool(diaphoresis)
            + 5 * as_bool(radiates_arm_shoulder) - 4 * as_bool(pleuritic) - 6 * as_bool(palpation))
