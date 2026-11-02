"""HEART score for chest pain in the emergency department (Six et al. 2008; Backus et al. 2010).

Five items, 0-2 points each, total 0-10 (Melki & Jernberg 2013, Table 1 and Methods):
  history     clinician's grade: 0 slightly, 1 moderately, 2 highly suspicious
  ecg         clinician's grade: 0 normal, 1 non-specific repolarisation disturbance (incl. bundle
              branch block, pacemaker rhythm, LVH), 2 significant ST depression
  age         <45 -> 0, 45-64 -> 1, >=65 -> 2
  risk        risk_factors = number of: diabetes, hypertension, current smoking (<90 days),
              hypercholesterolaemia, family history of CAD, obesity (BMI >30), angina pectoris.
              0 -> 0, 1-2 -> 1, >=3 or atherosclerosis (prior revascularisation, MI, stroke/TIA,
              peripheral artery disease) -> 2
  troponin    troponin_ratio = first troponin / local decision limit:
              <1 -> 0, 1-3 -> 1, >3 -> 2 ("between 1 and 3 times the decision limit", Methods)
Published groups: 0-3 low, 4-6 intermediate, 7-10 high risk of a major adverse cardiac event.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float

SOURCE = ("Melki D, Jernberg T. HEART score: a simple and useful tool that may lower the proportion "
          "of chest pain patients who are admitted. Crit Pathw Cardiol 2013;12:127-31, Table 1 and "
          "Methods (score from Six AJ et al., Neth Heart J 2008;16:191-6; www.heartscore.nl)")


def heart(history, ecg, age, risk_factors, atherosclerosis, troponin_ratio) -> np.ndarray:
    """HEART score, 0-10 (float64; NaN for invalid rows). Source: SOURCE."""
    h, e, a, n, ath, t = np.broadcast_arrays(as_float(history), as_float(ecg), as_float(age),
                                             as_float(risk_factors), as_bool(atherosclerosis),
                                             as_float(troponin_ratio))
    grade = lambda x: np.where(np.isin(x, (0, 1, 2)), x, np.nan)
    age_pts = np.select([a < 45, a < 65, a >= 65], [0.0, 1.0, 2.0], np.nan)
    n = np.where((n >= 0) & (n == np.floor(n)), n, np.nan)
    risk_pts = np.where(ath == 1, 2.0, np.select([n == 0, n <= 2, n >= 3], [0.0, 1.0, 2.0], np.nan))
    risk_pts = np.where(np.isnan(ath) | np.isnan(n), np.nan, risk_pts)
    trop_pts = np.select([t < 0, t < 1, t <= 3, t > 3], [np.nan, 0.0, 1.0, 2.0], np.nan)
    return grade(h) + grade(e) + np.where(a >= 0, age_pts, np.nan) + risk_pts + trop_pts
