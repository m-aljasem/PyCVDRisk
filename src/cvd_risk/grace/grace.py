"""GRACE risk scores for acute coronary syndrome, as the published point nomograms.

grace_hospital: in-hospital death (Granger et al. 2003, Figure 4), 8 items.
grace_6month:   death from discharge to 6 months (Eagle et al. 2004, Figure 2), 9 items.
Both return total points; the papers map points to risk only through a coarse table (Granger) or a
plot (Eagle). Creatinine is in mg/dL (umol/L / 88.4).

Band edges. Eagle prints closed bands (<=49.9, 50-69.9, ...). Granger prints overlapping ones
(<=80, 80-99, 100-119, ...; 0-0.39, 0.40-0.79, ...; >4.0): the explicit "<=" bands include their edge,
every other band is [lower, next lower), and creatinine 4.0 scores 28. Granger's own worked examples
put SBP 100 and creatinine 0.4 in the band below (see reference/grace/REPORT.md).
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float

SOURCE = {
    "hospital": "Granger CB et al. Predictors of hospital mortality in the Global Registry of Acute Coronary "
                "Events. Arch Intern Med 2003;163:2345-53, Figure 4",
    "6month": "Eagle KA et al. A validated prediction model for all forms of acute coronary syndrome. "
              "JAMA 2004;291:2727-33, Figure 2",
}


def _band(x, edges, points, first_le=False):
    """points[i] for edges[i-1] <= x < edges[i]; with first_le the first band is x <= edges[0]."""
    x = as_float(x)
    i = np.digitize(x, edges)
    if first_le:
        i = np.where(x == edges[0], 0, i)
    return np.where(x >= 0, np.asarray(points, float)[i], np.nan)


def grace_hospital(killip, sbp, heart_rate, age, creatinine, cardiac_arrest, st_deviation,
                   elevated_enzymes) -> np.ndarray:
    """GRACE in-hospital death score (points). Killip class 1-4; creatinine mg/dL. Source: SOURCE."""
    k = as_float(killip)
    return (np.select([k == 1, k == 2, k == 3, k == 4], [0.0, 20.0, 39.0, 59.0], np.nan)
            + _band(sbp, [80, 100, 120, 140, 160, 200], [58, 53, 43, 34, 24, 10, 0], first_le=True)
            + _band(heart_rate, [50, 70, 90, 110, 150, 200], [0, 3, 9, 15, 24, 38, 46], first_le=True)
            + _band(age, [30, 40, 50, 60, 70, 80, 90], [0, 8, 25, 41, 58, 75, 91, 100], first_le=True)
            + _band(creatinine, [0.4, 0.8, 1.2, 1.6, 2.0, 4.0], [1, 4, 7, 10, 13, 21, 28])
            + 39 * as_bool(cardiac_arrest) + 28 * as_bool(st_deviation) + 14 * as_bool(elevated_enzymes))


def grace_6month(age, chf_history, mi_history, heart_rate, sbp, st_depression, creatinine,
                 elevated_enzymes, in_hospital_pci) -> np.ndarray:
    """GRACE 6-month post-discharge death score (points). Creatinine mg/dL; in_hospital_pci = 1 if
    PCI was done in hospital (its absence scores 14). Source: SOURCE."""
    return (_band(age, [40, 50, 60, 70, 80, 90], [0, 18, 36, 55, 73, 91, 100])
            + 24 * as_bool(chf_history) + 12 * as_bool(mi_history)
            + _band(heart_rate, [50, 70, 90, 110, 150, 200], [0, 3, 9, 14, 23, 35, 43])
            + _band(sbp, [80, 100, 120, 140, 160, 200], [24, 22, 18, 14, 10, 4, 0])
            + 11 * as_bool(st_depression)
            + _band(creatinine, [0.4, 0.8, 1.2, 1.6, 2.0, 4.0], [1, 3, 5, 7, 9, 15, 20])
            + 15 * as_bool(elevated_enzymes) + 14 * (1 - as_bool(in_hospital_pci)))
