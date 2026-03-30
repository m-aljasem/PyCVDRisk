"""JALS score (Japan Arteriosclerosis Longitudinal Study, 2019): points for the 5- and 10-year absolute risk of
stroke, acute myocardial infarction (AMI), their composite, and cardiovascular death, Japanese adults aged 40-89.

As printed in Harada A, Ueshima H, Kinoshita Y, et al. Hypertens Res 2019;42(4):567-79. Tables 2-5 give each
outcome's points (10 x log2 of the incidence rate ratio) for a model with atrial fibrillation (AF) and one without;
give af= for the first. Blood pressure by the higher of the SBP and DBP category (JSH 2014: optimal < 120 and < 80,
normal 120-129 / 80-84, high-normal 130-139 / 85-89, grade I 140-159 / 90-99, grade II 160-179 / 100-109, grade
III >= 180 / >= 110), scored separately with and without antihypertensive medication.

jals returns Table 6's "score": the total minus the age points. jals_risk turns it into Table 6's absolute risk:
5-year 1 - exp(-c x 2^(score/10)), c per outcome and age group; 10-year 1 - (1 - 5-year risk) x (1 - 5-year
risk of the next age group). Table 6 prints the AF models only, to 2 decimals; C below is recovered from it
(reference/jals/fit_table6.py: 257/260 printed 5-year and 235/260 10-year values, the rest within 0.01 point).
The paper gives no risk table for the models without AF.

Ages outside 40-89 give NaN.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = "Harada A, Ueshima H, Kinoshita Y, et al. Hypertens Res 2019;42(4):567-79, Tables 2-6"

OUTCOMES = ("stroke", "ami", "composite", "cvd_death")
AGE_CUTS = (50, 60, 70, 80)  # 40-49, 50-59, 60-69, 70-79, 80-89
BP_CUTS = ((120, 130, 140, 160, 180), (80, 85, 90, 100, 110))  # optimal, normal, high-normal, grade I-III

# Tables 2-5 points: item -> (model without AF, model with AF); category tuples in the order of the cut points
POINTS = {
    "stroke": dict(af=(None, 21), bp=((0, 7, 7, 14, 21, 27), (0, 7, 7, 14, 21, 27)),
                   bp_treated=((12, 14, 13, 18, 19, 19), (11, 14, 13, 18, 19, 19)),
                   age=((0, 4, 13, 24, 31), (0, 4, 13, 23, 31)), diabetes=(6, 6), men=(5, 4), smoker=(7, 7)),
    "ami": dict(af=(None, 13), hdl=((11, 7, 0), (11, 7, 0)), bp=((0, 3, 6, 7, 9, 17), (0, 3, 6, 7, 9, 17)),
                bp_treated=((2, 14, 17, 11, 6, 9), (2, 14, 17, 11, 6, 9)),
                age=((0, 11, 14, 21, 28), (0, 11, 13, 21, 28)), non_hdl=((0, 10, 13, 17), (0, 10, 13, 17)),
                diabetes=(9, 9), men=(17, 17), smoker=(10, 10)),
    "composite": dict(af=(None, 20), hdl=((4, 2, 0), (4, 2, 0)), bp=((0, 6, 7, 13, 19, 26), (0, 6, 7, 14, 19, 26)),
                      bp_treated=((11, 14, 14, 17, 17, 18), (10, 14, 14, 17, 18, 18)),
                      age=((0, 5, 13, 23, 31), (0, 5, 13, 23, 30)), diabetes=(6, 6), men=(6, 5), smoker=(7, 8)),
    "cvd_death": dict(af=(None, 18), bmi=((6, 0, -1), (7, 0, -2)), hdl=((3, -1, 0), (3, -1, 0)),
                      bp=((0, 0, 4, 4, 10, 13), (0, 0, 4, 4, 11, 13)), bp_treated=((7, 8, 3, 6, 8, 5), (6, 8, 4, 6, 9, 5)),
                      age=((0, 6, 16, 35, 56), (0, 6, 16, 35, 55)), egfr=((11, 2, 0, 1), (11, 2, 0, 2)),
                      diabetes=(4, 4), men=(7, 6), smoker=(9, 9)),
}
CUTS = dict(hdl=(40, 60), non_hdl=(130, 150, 170), bmi=(18.5, 25), egfr=(45, 60, 90))
# Table 6 (AF models): c = 5-year cumulative hazard at score 0, per age group (reference/jals/fixtures)
C = dict(stroke=(0.00112124, 0.00147186, 0.00269892, 0.0055283, 0.00933885),
         ami=(6.69028e-05, 0.000146371, 0.000168871, 0.00028776, 0.000466331),
         composite=(0.00117965, 0.00171427, 0.00291729, 0.00575496, 0.0094721),
         cvd_death=(0.000672136, 0.00101138, 0.00202594, 0.00738996, 0.0310137))


def _age_group(age):
    a = as_float(age)
    return np.where((a >= 40) & (a < 90), np.digitize(np.nan_to_num(a), AGE_CUTS), -1)


def _pts(x, cuts, pts):
    v = as_float(x)
    return np.where(np.isnan(v), np.nan, np.asarray(pts, float)[np.digitize(np.nan_to_num(v), cuts)])


def jals(age, sex, sbp, dbp, bp_treated, smoker, diabetes, *, outcome, af=None, hdl=None, non_hdl=None, bmi=None,
         egfr=None) -> np.ndarray:
    """Table 6 score (total points minus the age points) for outcome = "stroke", "ami", "composite" or "cvd_death".
    af given: the model with AF. Also needed: hdl (mg/dL) for ami, composite and cvd_death; non_hdl (mg/dL) for ami;
    bmi (kg/m2) and egfr (mL/min/1.73 m2, J-CKDI equation) for cvd_death. Flags 0/1; smoker = current smoker."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    p = POINTS[outcome]
    given = dict(hdl=hdl, non_hdl=non_hdl, bmi=bmi, egfr=egfr)
    missing = [k for k in CUTS if k in p and given[k] is None]
    if missing:
        raise ValueError(f'outcome="{outcome}" needs ' + ", ".join(f"{k}=" for k in missing))
    m = int(af is not None)
    s, d = as_float(sbp), as_float(dbp)
    cat = np.maximum(np.digitize(np.nan_to_num(s), BP_CUTS[0]), np.digitize(np.nan_to_num(d), BP_CUTS[1]))
    t = as_bool(bp_treated)
    bp = np.where(t == 1, np.asarray(p["bp_treated"][m], float)[cat], np.asarray(p["bp"][m], float)[cat])
    bp = np.where(np.isnan(s) | np.isnan(d) | np.isnan(t), np.nan, bp)
    sx = as_sex(sex)
    score = (bp + np.where(sx == 1, p["men"][m], np.where(sx == 0, 0.0, np.nan)) + p["smoker"][m] * as_bool(smoker)
             + p["diabetes"][m] * as_bool(diabetes) + np.where(_age_group(age) < 0, np.nan, 0.0))
    for k in CUTS:
        if k in p:
            score = score + _pts(given[k], CUTS[k], p[k][m])
    if m:
        score = score + p["af"][1] * as_bool(af)
    return score


def jals_risk(score, age, outcome, years=5) -> np.ndarray:
    """Table 6 absolute risk (fraction) over 5 or 10 years for a jals score (model with AF) at a given age."""
    if outcome not in OUTCOMES or years not in (5, 10):
        raise ValueError(f"outcome must be one of {OUTCOMES}; years 5 or 10")
    g = _age_group(age)
    c = np.asarray(C[outcome] + (np.nan,))
    sc = 2 ** (as_float(score) / 10)
    first = 1 - np.exp(-c[g] * sc)
    if years == 5:
        return first
    nxt = np.where(g < 0, -1, np.minimum(g + 1, 4))  # 80-89: the last 5 years use the same group
    return 1 - (1 - first) * np.exp(-c[nxt] * sc)
