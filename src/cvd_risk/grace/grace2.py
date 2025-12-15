"""GRACE 2.0: risk of death, or death or MI, 1 and 3 years after admission for acute coronary syndrome.
Fox KAA, FitzGerald G, Puymirat E, et al. BMJ Open 2014;4:e004425, doi:10.1136/bmjopen-2013-004425.

Cox models, risk = 1 - S0 ** exp(xb), with restricted cubic splines for age (per 10 years), heart rate
(per 30 bpm), systolic blood pressure (per -20 mmHg) and creatinine (mg/dL). The paper prints only
hazard ratios (Table 2); every number here is from the developers' calculator (Center for Outcomes
Research, UMass, outcomes-umassmed.org/grace/acs_risk2, archived; see reference/grace2/):
- the 1-year death models are coded in its grace.js, knots and coefficients as below;
- grace_lookups.js, which the calculator actually runs, holds each model's term values at its
  dropdown entries (every year of age, one value per band of the other inputs) and the baseline
  survivals. The spline coefficients of the other models are recovered from those values with the
  grace.js knots, at the 5 decimals grace.js uses: every value is reproduced to its printed
  precision (|error| <= 5e-8).
Exception: the 3-year death models' SBP term fits no spline (nor Table 2's piecewise hazard ratios),
so it is the calculator's value for the SBP band (<80, 80-99, 100-109, ..., 160-179, >=180 mmHg).

Substituted models (Fox 2014 Table 4) replace creatinine with renal insufficiency and Killip class
with diuretic use; pass either creatinine and killip, or renal_insufficiency and diuretic. (The
calculator switches to them when either is missing, setting renal insufficiency = creatinine > 1.59
or diuretic = Killip III-IV from the one given.) The calculator's in-hospital and 6-month outcomes
are the GRACE 1.0 models (grace_hospital, grace_6month and the Fox 2006 nomograms), not these.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float

_KNOTS = {"age": (4.8165, 6.6571, 8.3095), "pulse": (1.71, 2.31, 2.77, 3.95),
          "sbp": (-9.02, -6.97, -5.32), "creatinine": (0.685, 0.925, 1.145, 2.065)}
_SBP_BANDS = (80, 100, 110, 120, 130, 140, 160, 180)

# (horizon, outcome, substituted): s0, age, pulse, sbp, creatinine | renal, killip II-IV | diuretic,
# cardiac arrest, elevated enzymes, ST deviation. Spline terms: (linear, nonlinear 1[, nonlinear 2]).
_M = {
    (1, "death", False): (0.9983577131, (0.41157, 0.01290), (0.13138, 0.40176, -1.37249), (0.08222, 0.01020),
                          (-0.51259, 7.52634, -18.23023), (0.63827, 0.85325, 1.29372), 0.87185, 0.37660, 0.44303),
    (1, "death", True): (0.9994190637, (0.44230, 0.01231), (0.16705, 0.43926, -1.48002), (0.02338, 0.01835),
                         0.44469, 0.68008, 1.18193, 0.39944, 0.46524),
    (1, "death_mi", False): (0.9865696068, (0.14677, 0.02090), (0.04230, 0.37817, -1.25482), (0.01797, 0.01020),
                             (-0.15965, 5.02142, -12.33887), (0.53625, 0.68594, 1.15850), 0.67071, 0.22710, 0.32831),
    (1, "death_mi", True): (0.9919559302, (0.17692, 0.02013), (0.03091, 0.42160, -1.36224), (-0.02154, 0.01577),
                            0.47766, 0.53363, 0.92949, 0.25469, 0.34746),
    (3, "death", False): (0.9998715509, (0.60935, 0), (-0.05444, 0.82884, -2.65671),
                          (3.5872931, 3.0903948, 2.6394952, 2.442021, 2.2607121, 2.0407927, 1.6991854, 1.5533218, 1.5973204),
                          (0.39403, 0, 0), (0.08466, 0.82842, 0.82842), 1.07623, 0, 0.41228),
    (3, "death", True): (0.9998961080, (0.58998, 0), (-0.09487, 0.86637, -2.75341),
                         (4.5695482, 3.9853504, 3.4167333, 3.1524173, 2.9277477, 2.7001331, 2.4021178, 2.3409886, 2.4324792),
                         0.69806, 0.68142, 1.17867, 0, 0.32912),
    (3, "death_mi", False): (0.9875014749, (0.43299, 0), (0.18097, 0, 0), (0.14521, 0), (0.35945, 0, 0),
                             (0.21991, 1.06610, 1.06610), 0.61622, 0, 0.42540),
    (3, "death_mi", True): (0.9958878333, (0.42198, 0), (0.22480, 0, 0), (-0.02504, 0.01403),
                            0.51167, 0.65284, 0.72786, 0, 0.38290),
}


def _spline(x, knots, b):
    """b[0] * x + restricted cubic spline terms, written as grace.js writes them."""
    p = lambda t: np.maximum(x - t, 0) ** 3
    a, z = knots[-2], knots[-1]
    return b[0] * x + sum(bj * (p(t) - p(a) * (z - t) / (z - a) + p(z) * (a - t) / (z - a))
                          for bj, t in zip(b[1:], knots))


def grace2(age, heart_rate, sbp, cardiac_arrest, st_deviation, elevated_enzymes, *, creatinine=None, killip=None,
           renal_insufficiency=None, diuretic=None, horizon=1, outcome="death") -> np.ndarray:
    """GRACE 2.0 risk (fraction) of `outcome` ("death" or "death_mi") within `horizon` (1 or 3) years of
    admission. Creatinine mg/dL (umol/L / 88.4), Killip class 1-4; or the substituted model with
    renal_insufficiency and diuretic (0/1) instead. NaN for invalid rows."""
    full, sub = creatinine is not None and killip is not None, renal_insufficiency is not None and diuretic is not None
    if full == sub or (horizon, outcome, sub) not in _M:
        raise ValueError("give creatinine and killip, or renal_insufficiency and diuretic; horizon 1 or 3; "
                         "outcome 'death' or 'death_mi'")
    s0, b_age, b_pulse, b_sbp, b_renal, b_killip, b_arrest, b_enz, b_st = _M[horizon, outcome, sub]
    a, hr, s = as_float(age), as_float(heart_rate), as_float(sbp)
    flags = [as_bool(v) for v in (cardiac_arrest, elevated_enzymes, st_deviation)]
    if sub:
        r, d = as_bool(renal_insufficiency), as_bool(diuretic)
        renal, heart = b_renal * r, b_killip * d
    else:
        c, k = as_float(creatinine), as_float(killip)
        r, d = np.where(c >= 0, 0.0, np.nan), np.select([k == 1, k == 2, k == 3, k == 4], [0.0, 1, 2, 3], np.nan)
        renal = _spline(np.where(c >= 0, c, 0), _KNOTS["creatinine"], b_renal)
        heart = np.asarray((0,) + b_killip)[np.where(np.isnan(d), 0, d).astype(int)]
    if len(b_sbp) == len(_SBP_BANDS) + 1:  # 3-year death: one value per SBP band
        sbp_term = np.asarray(b_sbp)[np.digitize(np.where(s > 0, s, 0), _SBP_BANDS)]
    else:
        sbp_term = _spline(s / -20, _KNOTS["sbp"], b_sbp)
    xb = (_spline(a / 10, _KNOTS["age"], b_age) + _spline(hr / 30, _KNOTS["pulse"], b_pulse) + sbp_term + renal + heart
          + b_arrest * flags[0] + b_enz * flags[1] + b_st * flags[2])
    ok = (a > 0) & (hr > 0) & (s > 0) & ~np.isnan(r + d) & ~np.isnan(flags[0] + flags[1] + flags[2])
    return np.where(ok, 1 - s0 ** np.exp(np.where(ok, xb, 0)), np.nan)
