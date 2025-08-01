"""PROCAM scores (Assmann et al. 2007): 10-year risk of a major coronary event (men and women, ages
20-75; Weibull score) and of a cerebral ischaemic event (ischaemic stroke or TIA; ages 35-65).

Coronary: points from Table 2, then Table 3 (men) or Table 4 (women) give the 10-year risk category
for the person's age (CATEGORIES). Those tables print each category's upper bound with "<"; it is
read as "<=", the only reading under which every row covers 0-71 points without gaps. The paper does
not print the full Weibull model's coefficients, so this is the published point score.
Stroke: points from Table 6, then STROKE_RISK (Table 7, points -> 10-year risk, %; it prints < 42 as
"<= 0.1" and > 80 as "> 10.0", so those totals have no value there).
Cholesterol and triglycerides are in mmol/L, converted to the source's mg/dL (cholesterol with
PREVENT's factor, triglycerides x 88.57). Bands are in whole source units: LDL "101-105" is
[101, 106), age 47 is [47, 48). Replaces the 2002 score (Circulation 2002;105:310; men 35-65).
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex
from cvd_risk.prevent.prevent import MG_PER_MMOL

SOURCE = ("Assmann G, Schulte H, Cullen P, Seedorf U. Assessing risk of myocardial infarction and stroke: new "
          "data from the Prospective Cardiovascular Münster (PROCAM) study. Eur J Clin Invest 2007;37:925-32, "
          "Tables 2-4, 6-7")
TG_MG_PER_MMOL = 88.57
CATEGORIES = ("0-4%", "5-9%", "10-19%", "20-29%", ">=30%")
# Tables 3-4: per age, the highest points of each category in order (the last printed category is open);
# ages below the first key (men 20-24, women 20-33) are "<= 71": every total is 0-4%.
_M = {25: (66,), 26: (63,), 27: (60, 69), 28: (57, 67), 29: (55, 64), 30: (53, 62), 31: (51, 60, 62), 32: (49, 58, 67),
      33: (47, 56, 65), 34: (45, 54, 63, 69), 35: (43, 52, 62, 67), 36: (41, 51, 60, 66), 37: (40, 49, 58, 64),
      38: (38, 48, 57, 63), 39: (37, 46, 55, 61), 40: (35, 45, 54, 60), 41: (34, 43, 53, 58), 42: (33, 42, 51, 57),
      43: (31, 41, 50, 56), 44: (30, 39, 49, 55), 45: (29, 38, 48, 53), 46: (28, 37, 46, 52), 47: (27, 36, 45, 51),
      48: (26, 35, 44, 50), 49: (25, 34, 43, 49), 50: (23, 33, 42, 48), 51: (23, 32, 41, 47), 52: (22, 31, 40, 46),
      53: (21, 30, 39, 45), 54: (20, 29, 38, 44), 55: (19, 28, 37, 43), 56: (18, 27, 37, 42), 57: (17, 26, 36, 41),
      58: (16, 26, 35, 41), 59: (15, 25, 34, 40), 60: (15, 24, 33, 39), 61: (14, 23, 33, 38), 62: (13, 22, 32, 38),
      63: (12, 22, 31, 37), 64: (12, 21, 30, 36), 65: (11, 20, 30, 35), 66: (10, 20, 29, 35), 67: (10, 19, 28, 34),
      68: (9, 18, 28, 33), 69: (8, 17, 27, 33), 70: (8, 17, 26, 32), 71: (7, 16, 26, 31), 72: (6, 16, 25, 31),
      73: (6, 15, 24, 30), 74: (5, 14, 24, 29), 75: (4, 14, 23, 29)}
_F = {34: (69,), 35: (66,), 36: (64,), 37: (62, 70), 38: (59, 68), 39: (57, 66), 40: (55, 64), 41: (53, 62),
      42: (51, 60, 69), 43: (49, 58, 67), 44: (48, 56, 65), 45: (46, 55, 64, 69), 46: (44, 53, 62, 67),
      47: (43, 51, 60, 66), 48: (41, 50, 59, 64), 49: (39, 48, 57, 62), 50: (38, 47, 56, 61), 51: (36, 45, 54, 60),
      52: (35, 44, 53, 58), 53: (34, 42, 51, 57), 54: (32, 41, 50, 55), 55: (31, 40, 49, 54), 56: (30, 39, 47, 53),
      57: (28, 37, 46, 51), 58: (27, 36, 45, 50), 59: (26, 35, 44, 49), 60: (25, 34, 42, 48), 61: (24, 32, 41, 47),
      62: (23, 31, 40, 46), 63: (21, 30, 39, 45), 64: (20, 29, 38, 43), 65: (19, 28, 37, 42), 66: (18, 27, 36, 41),
      67: (17, 26, 35, 40), 68: (16, 25, 34, 39), 69: (15, 24, 33, 38), 70: (14, 23, 32, 37), 71: (13, 22, 31, 36),
      72: (12, 21, 30, 35), 73: (12, 20, 29, 35), 74: (11, 19, 28, 34), 75: (10, 19, 27, 33)}
CORONARY_TABLES = {"male": _M, "female": _F}
STROKE_RISK = dict(zip(range(42, 81), (
    0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.4, 1.6,
    1.7, 1.9, 2.2, 2.4, 2.7, 3.0, 3.4, 3.8, 4.2, 4.7, 5.2, 5.8, 6.5, 7.3, 8.1, 9.0, 10.0)))


def _band(x, edges, points):
    x = np.asarray(x, float)
    return np.where(x >= 0, np.asarray(points, float)[np.digitize(x, edges)], np.nan)


def _mg(x, factor):
    return np.round(as_float(x) * factor, 6)  # so a band edge entered in mmol/L is not lost to rounding error


def _category(table, age, pts):
    out = np.full(np.broadcast(age, pts).shape, np.nan)
    age, pts = np.broadcast_arrays(age, pts)
    for i, (a, p) in enumerate(zip(age.flat, pts.flat)):
        if 20 <= a < 76 and p >= 0:
            out.flat[i] = sum(p > u for u in table.get(int(a), ()))
    return out


def procam(age, sex, ldl_chol, hdl_chol, triglycerides, sbp, smoker, diabetes, family_history,
           output="points") -> np.ndarray:
    """PROCAM coronary score: Table 2 points (output="points") or the 10-year risk category from
    Tables 3-4 as an index into CATEGORIES (output="category"). Source: SOURCE."""
    if output not in ("points", "category"):
        raise ValueError('output must be "points" or "category"')
    a, sx = as_float(age), as_sex(sex)
    valid = np.where((a >= 20) & (a < 76) & ((sx == 0) | (sx == 1)), 0.0, np.nan)
    pts = (valid
           + _band(_mg(ldl_chol, MG_PER_MMOL), range(101, 197, 5), range(21))
           + _band(_mg(hdl_chol, MG_PER_MMOL), range(36, 57, 2), range(11, -1, -1))
           + _band(as_float(sbp), range(110, 181, 10), range(9))
           + _band(_mg(triglycerides, TG_MG_PER_MMOL), [100, 150, 200], [0, 2, 3, 4])
           + 12 * as_bool(smoker) + 5 * as_bool(family_history) + np.where(sx == 1, 9, 11) * as_bool(diabetes))
    if output == "points":
        return pts
    return np.where(sx == 1, _category(_M, a, pts), _category(_F, a, pts))


def procam_stroke(age, sex, smoker, diabetes, sbp) -> np.ndarray:
    """PROCAM cerebral ischaemic event score in points (Table 6); STROKE_RISK gives the risk. Source: SOURCE."""
    a, sx = as_float(age), as_sex(sex)
    age_pts = np.where((a >= 35) & (a < 66), np.floor(a), np.nan)
    sex_pts = np.where(sx == 1, 6.0, np.where(sx == 0, 0.0, np.nan))
    return (age_pts + sex_pts + 9 * as_bool(smoker) + 7 * as_bool(diabetes)
            + _band(as_float(sbp), range(140, 181, 5), range(10)))
