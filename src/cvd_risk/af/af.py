"""Atrial fibrillation point scores: CHADS2 and CHA2DS2-VASc (stroke), HAS-BLED (major bleeding).

Inputs are 0/1 flags (plus age in years); a row with an invalid flag or age is NaN.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = {
    "chads2": "Gage BF et al. JAMA 2001;285:2864-70, Methods and Table 2 footnote",
    "cha2ds2_vasc": "Lip GYH et al. Chest 2010;137:263-72, Table 2",
    "has_bled": "Pisters R et al. Chest 2010;138:1093-100, Table 2 and Methods",
}


def _age(age):
    a = as_float(age)
    return np.where(a >= 0, a, np.nan)


def chads2(age, heart_failure, hypertension, diabetes, stroke_tia) -> np.ndarray:
    """CHADS2, 0-6: 1 each for recent CHF, hypertension, age >= 75, diabetes; 2 for prior stroke/TIA."""
    a = _age(age)
    return (as_bool(heart_failure) + as_bool(hypertension) + np.where(np.isnan(a), np.nan, a >= 75)
            + as_bool(diabetes) + 2 * as_bool(stroke_tia))


def cha2ds2_vasc(age, sex, heart_failure, hypertension, diabetes, stroke_tia_te, vascular_disease) -> np.ndarray:
    """CHA2DS2-VASc, 0-9: CHF/LV dysfunction 1, hypertension 1, age >= 75 2, diabetes 1,
    stroke/TIA/thromboembolism 2, vascular disease (prior MI, PAD, aortic plaque) 1, age 65-74 1, female 1."""
    a = _age(age)
    sx = as_sex(sex)
    age_pts = np.select([np.isnan(a), a >= 75, a >= 65], [np.nan, 2.0, 1.0], 0.0)
    return (as_bool(heart_failure) + as_bool(hypertension) + age_pts + as_bool(diabetes)
            + 2 * as_bool(stroke_tia_te) + as_bool(vascular_disease) + np.where(sx >= 0, sx == 0, np.nan))


def has_bled(age, hypertension, abnormal_renal, abnormal_liver, stroke, bleeding, labile_inr,
             drugs, alcohol) -> np.ndarray:
    """HAS-BLED, 0-9: 1 point each for uncontrolled hypertension (SBP > 160), abnormal renal function,
    abnormal liver function, stroke, bleeding history or predisposition, labile INR, age > 65
    ("elderly (>65 years)"), antiplatelet/NSAID drugs, alcohol (>= 8 units/week)."""
    a = _age(age)
    return (as_bool(hypertension) + as_bool(abnormal_renal) + as_bool(abnormal_liver) + as_bool(stroke)
            + as_bool(bleeding) + as_bool(labile_inr) + np.where(np.isnan(a), np.nan, a > 65)
            + as_bool(drugs) + as_bool(alcohol))
