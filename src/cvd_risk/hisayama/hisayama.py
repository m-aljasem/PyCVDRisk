"""Hisayama ASCVD risk model (Japan, 2022): 10-year risk of a first atherosclerotic cardiovascular event (CHD or
atherothrombotic brain infarction), Japanese adults aged 40-84 without CVD.

As printed in Honda T, Chen S, Hata J, et al. J Atheroscler Thromb 2022;29(3):345-61:
- hisayama: the multivariable equation, p = 1 - 0.9696^exp(sum(beta x) - 6.7963) (Table 2, LDL-C model), or with
  non_hdl= the alternative model (Supplemental Table 4, constant 6.9895). S10 accounts for the competing risk of death.
- hisayama_points / hisayama_points_risk: the simplified score (Fig. 1; points from Supplemental Table 3) and its
  stated formula p = 1 - 0.9696^exp((score + age points) x 0.144 - 2.4767). Fig. 1's own risk table does not
  follow this formula (it runs higher, e.g. 15.7% against 11.9% at score 27, ages 40-49); the formula is what the
  JAS 2022 chart and the R package Jcvrisk use (reference/hisayama/REPORT.md).

Cholesterol in mg/dL; flags 0/1. Ages outside 40-84 give NaN.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = "Honda T, Chen S, Hata J, et al. J Atheroscler Thromb 2022;29(3):345-61, Table 2, Fig. 1, Supplemental Tables 3-4"

S10 = 0.9696
# Table 2 (LDL-C model) and Supplemental Table 4 (non-HDL-C model): beta, constant (sum of beta x mean)
COEF = {"ldl": (dict(age=0.077, male=0.984, sbp=0.010, diabetes=0.459, hdl=-0.012, lipid=0.005, proteinuria=0.632,
                     smoker=0.336, no_exercise=0.339), 6.7963),
        "non_hdl": (dict(age=0.078, male=0.979, sbp=0.010, diabetes=0.436, hdl=-0.010, lipid=0.005, proteinuria=0.629,
                         smoker=0.341, no_exercise=0.334), 6.9895)}

# Fig. 1 points: (category cut points, points)
AGE = ((40, 50, 60, 70, 80, 85), (0, 5, 11, 16, 20))
SBP = ((120, 130, 140, 160), (0, 1, 2, 3, 4))
HDL = ((40, 60), (2, 1, 0))
LIPID = {"ldl": ((120, 140, 160), (0, 1, 2, 3)), "non_hdl": ((150, 170, 190), (0, 1, 2, 3))}
FLAGS = dict(male=7, diabetes=3, proteinuria=4, smoker=2, no_exercise=2)
POINT_SLOPE, POINT_CONST = 0.144, 2.4767


def _lipid(ldl, non_hdl):
    if (ldl is None) == (non_hdl is None):
        raise ValueError("give ldl= (main model) or non_hdl= (alternative model), not both")
    return ("ldl", ldl) if ldl is not None else ("non_hdl", non_hdl)


def _age_ok(age):
    a = as_float(age)
    return np.where((a >= 40) & (a < 85), a, np.nan)


def hisayama(age, sex, sbp, diabetes, hdl, proteinuria, smoker, exercise, *, ldl=None, non_hdl=None) -> np.ndarray:
    """10-year ASCVD risk (fraction) from the multivariable equation. sbp mmHg; hdl, ldl / non_hdl mg/dL;
    smoker = current smoker; proteinuria = dipstick 1+ or more; exercise = regular exercise (sports or other
    exercise >= 3 times a week)."""
    kind, lipid = _lipid(ldl, non_hdl)
    b, mean = COEF[kind]
    sx = as_sex(sex)
    lp = (b["age"] * _age_ok(age) + b["male"] * np.where(sx == 1, 1.0, np.where(sx == 0, 0.0, np.nan))
          + b["sbp"] * as_float(sbp) + b["diabetes"] * as_bool(diabetes) + b["hdl"] * as_float(hdl)
          + b["lipid"] * as_float(lipid) + b["proteinuria"] * as_bool(proteinuria) + b["smoker"] * as_bool(smoker)
          + b["no_exercise"] * (1 - as_bool(exercise)))
    return 1 - S10 ** np.exp(lp - mean)


def _band(x, cuts, pts):
    v = as_float(x)
    return np.where(np.isnan(v), np.nan, np.asarray(pts, float)[np.digitize(np.nan_to_num(v), cuts)])


def hisayama_points(sex, sbp, diabetes, hdl, proteinuria, smoker, exercise, *, ldl=None, non_hdl=None) -> np.ndarray:
    """Fig. 1 total score (0-27, age excluded). With non_hdl= the alternative score's non-HDL-C points."""
    kind, lipid = _lipid(ldl, non_hdl)
    sx = as_sex(sex)
    return (np.where(sx == 1, FLAGS["male"], np.where(sx == 0, 0.0, np.nan)) + _band(sbp, *SBP) + _band(hdl, *HDL)
            + _band(lipid, *LIPID[kind]) + FLAGS["diabetes"] * as_bool(diabetes)
            + FLAGS["proteinuria"] * as_bool(proteinuria) + FLAGS["smoker"] * as_bool(smoker)
            + FLAGS["no_exercise"] * (1 - as_bool(exercise)))


def hisayama_points_risk(points, age) -> np.ndarray:
    """10-year ASCVD risk (fraction) for a Fig. 1 total score at a given age, by the paper's formula."""
    a = _age_ok(age)
    age_pts = np.where(np.isnan(a), np.nan, np.asarray(AGE[1], float)[np.clip(np.digitize(np.nan_to_num(a), AGE[0]) - 1, 0, 4)])
    return 1 - S10 ** np.exp((as_float(points) + age_pts) * POINT_SLOPE - POINT_CONST)
