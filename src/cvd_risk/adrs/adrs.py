"""Abu Dhabi Risk Study (ADRS) equations (2024): 10-year risk of ASCVD, hard CAD and stroke.

Cox models from Table 2 with the centring values and baseline survival of the supplement's SPSS
syntax (Data S1): risk = 1 - S0 ** exp(sum(b * (x - mean))). Vitamin D in ng/mL; tc_hdl is total/HDL
cholesterol (mmol/L); egfr is CKD-EPI 2009 (mL/min/1.73 m2), entered as its deviation from the cohort
mean, which the paper does not print: 111.4, from Table 1 (women 114.0 - 2.6, men 108.9 + 2.5).
"Mean blood pressure" is mean arterial pressure, DBP + (SBP - DBP) / 3 (it reproduces Table 1).
The CAD model's blood-pressure categories are the ACC/AHA 2017 stages named in Table 2.
Stroke: Data S1 computes the score twice, the second time without the hypertension x sex term; this
package keeps the term (Table 2), with which the printed mean score 6.37 is consistent (REPORT.md).
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = ("Abu Dhabi Risk Study. Disease risk score derivation and validation in Abu Dhabi, United Arab Emirates: "
          "a retrospective cohort study. J Am Heart Assoc 2024;13:e035930, Table 2 and Data S1")
OUTCOMES = ("ascvd", "cad", "stroke")
GFR_MEAN = 111.4
S0 = {"ascvd": 0.994, "cad": 0.994, "stroke": 0.998}
# (coefficient, centring value) from Data S1
ASCVD = {"vitamin_d": (-0.007, 35.53), "dev_gfr": (-0.029, 0.0), "male": (1.516, 0.51), "smoker": (0.611, 0.09),
         "age": (0.204, 38.6), "bp_treated": (0.305, 0.13), "tc_hdl": (0.068, 4.10), "diabetes": (0.332, 0.21),
         "dev_gfr_x_male": (0.031, -1.28), "map": (0.012, 89.32), "age2": (-0.001, 1749.50)}
CAD = {"vitamin_d": (-0.007, 35.53), "dev_gfr": (-0.039, 0.0), "male": (2.136, 0.51), "smoker": (0.55, 0.09),
       "age": (0.067, 38.6), "hypertension": (0.327, 0.18), "tc_hdl": (0.067, 4.1), "diabetes": (1.543, 0.21),
       "dev_gfr_x_male": (0.027, -1.28), "diabetes_x_male": (-0.669, 0.12), "diabetes_x_dev_gfr": (0.019, -2.72),
       "bp_normal": (-1.118, 0.46), "bp_elevated": (-1.148, 0.15), "bp_stage1": (-1.378, 0.25),
       "bp_stage2": (-1.093, 0.14), "bp_crisis": (-0.698, 0.0)}
STROKE = {"smoker": 1.053, "sbp": 0.024, "age": 0.084, "hypertension_x_male": -0.646, "age_x_male": 0.011}
STROKE_MEAN = 6.37


def bp_stage(sbp, dbp):
    """ACC/AHA 2017 stage: 0 normal, 1 elevated, 2 stage 1, 3 stage 2, 4 crisis (SBP > 180 and/or DBP > 120)."""
    s, d = as_float(sbp), as_float(dbp)
    st = np.select([(s > 180) | (d > 120), (s >= 140) | (d >= 90), (s >= 130) | (d >= 80), s >= 120], [4, 3, 2, 1], 0)
    return np.where((s > 0) & (d > 0), st, np.nan)


def adrs(age, sex, smoker, sbp, dbp, tc_hdl, diabetes, hypertension, bp_treated, vitamin_d, egfr,
         outcome="ascvd") -> np.ndarray:
    """10-year risk (fraction) of "ascvd", "cad" or "stroke". Ages 18 and over. Source: SOURCE."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    a = as_float(age)
    a = np.where(a >= 18, a, np.nan)
    sx = as_sex(sex).astype(float)
    male = np.where(sx >= 0, sx, np.nan)
    smk, dm, htn = as_bool(smoker), as_bool(diabetes), as_bool(hypertension)
    s, d = as_float(sbp), as_float(dbp)
    s, d = np.where(s > 0, s, np.nan), np.where(d > 0, d, np.nan)
    if outcome == "stroke":
        lp = (STROKE["smoker"] * smk + STROKE["sbp"] * s + STROKE["age"] * a
              + STROKE["hypertension_x_male"] * htn * male + STROKE["age_x_male"] * a * male) - STROKE_MEAN
        return 1 - S0["stroke"] ** np.exp(lp)
    vd, ratio, g = as_float(vitamin_d), as_float(tc_hdl), as_float(egfr)
    dev = np.where(g > 0, g - GFR_MEAN, np.nan)
    x = {"vitamin_d": np.where(vd >= 0, vd, np.nan), "dev_gfr": dev, "male": male, "smoker": smk, "age": a,
         "tc_hdl": np.where(ratio > 0, ratio, np.nan), "diabetes": dm, "dev_gfr_x_male": dev * male}
    if outcome == "ascvd":
        model = ASCVD
        x.update(bp_treated=as_bool(bp_treated), map=d + (s - d) / 3, age2=a**2)
    else:
        model = CAD
        st = bp_stage(s, d)
        x.update(hypertension=htn, diabetes_x_male=dm * male, diabetes_x_dev_gfr=dm * dev,
                 **{k: np.where(np.isnan(st), np.nan, st == i)
                    for i, k in enumerate(("bp_normal", "bp_elevated", "bp_stage1", "bp_stage2", "bp_crisis"))})
    lp = sum(b * (x[k] - m) for k, (b, m) in model.items())
    return 1 - S0[outcome] ** np.exp(lp)
