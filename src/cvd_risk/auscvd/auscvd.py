"""Aus CVD Risk Calculator (2023 Australian guideline): AUS-PREDICT and AUS-PREDICT-Diabetes, 5-year CVD risk.

Cox models, risk = f(age, sex) * (1 - S0 ** exp(LP)) with continuous inputs centred on the cohort means;
f is the published recalibration multiplier (Australian / New Zealand CVD mortality rate ratio, MJA 2025
Box 5), applied after the equation. Coefficients from the Heart Foundation's official calculator
(cvdcheck.org.au); see reference/auscvd/REPORT.md.

SEIFA is the Australian area disadvantage quintile from the postcode (1 = most disadvantaged); the
equations were fitted on the reverse-ordered NZ deprivation quintile, so they use 6 - SEIFA, as the
calculator does.

Differences from the calculator, all deliberate:
- Results are not rounded. The calculator shows round(max(1%, unrecalibrated risk) * f), a whole percent.
- The true exponential is used; the calculator raises 2.718282 to the linear predictor (relative
  difference below 1e-6).
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex
from cvd_risk.predict.predict import _pos, _smoking

SOURCE = ("Heart Foundation / Australian Chronic Disease Prevention Alliance: Aus CVD Risk Calculator, "
          "cvdcheck.org.au (2023); Paige E et al. Med J Aust 2025;223(4), doi:10.5694/mja2.52718")

# sex -> (centre, beta, s0); "dep" is the deprivation quintile 6 - SEIFA
GENERAL = {
    "male": ({"age": 53.77579, "dep": 2.628206, "sbp": 129.1706, "tc_hdl": 4.325589},
             {"age": 0.0691512, "dep": 0.0900502, "ex_smoker": 0.0635803, "smoker": 0.5782258, "af": 0.474629,
              "diabetes": 0.5829839, "sbp": 0.0139242, "tc_hdl": 0.1350063, "bp_meds": 0.2676661,
              "lipid_meds": -0.0383788, "antithrombotic": 0.0981934, "age_x_diabetes": -0.018881,
              "age_x_sbp": -0.0001952, "bp_meds_x_sbp": -0.0054419},
             0.9728782),
    "female": ({"age": 58.47472, "dep": 2.625575, "sbp": 129.2052, "tc_hdl": 3.641988},
               {"age": 0.0905192, "dep": 0.0978413, "ex_smoker": 0.1162342, "smoker": 0.8235061, "af": 0.709571,
                "diabetes": 0.6207569, "sbp": 0.0114172, "tc_hdl": 0.1067098, "bp_meds": 0.3092255,
                "lipid_meds": -0.0863215, "antithrombotic": 0.1622663, "age_x_diabetes": -0.0230893,
                "age_x_sbp": -0.0002353, "bp_meds_x_sbp": -0.0066143},
               0.9828361),
}
# "constant": a fixed term in the calculator's diabetes equations, not documented (REPORT.md)
DIABETES = {
    "male": ({"age": 56.57672, "dep": 2.906154, "years": 5.255285, "bmi": 29.21475, "sbp": 131.8995,
              "tc_hdl": 4.267316, "egfr": 88.15267, "hba1c": 57.84909},
             {"age": 0.0510591, "dep": 0.0931026, "smoker": 0.318424, "af": 0.3149766, "years": 0.0117042,
              "bmi": 0.0156861, "sbp": 0.004488, "tc_hdl": 0.0858267, "egfr": -0.002239, "hba1c": 0.0065082,
              "acr_3_30": 0.4277236, "acr_gt_30": 0.803022, "insulin": 0.3899748, "constant": 0.1316217,
              "bp_meds": 0.1674871, "lipid_meds": -0.0616832, "antithrombotic": 0.0273617},
             0.9483001),
    "female": ({"age": 57.8318, "dep": 3.033218, "years": 5.222868, "bmi": 30.33892, "sbp": 132.2317,
                "tc_hdl": 3.882821, "egfr": 88.38233, "hba1c": 56.06538},
               {"age": 0.0607418, "dep": 0.1111981, "smoker": 0.7319247, "af": 0.4150528, "years": 0.011011,
                "bmi": 0.0112387, "sbp": 0.0084873, "tc_hdl": 0.1025064, "egfr": -0.0082137, "hba1c": 0.0109019,
                "acr_3_30": 0.2030524, "acr_gt_30": 0.7023991, "insulin": 0.2966913, "constant": 0.1594838,
                "bp_meds": 0.1671067, "lipid_meds": -0.2702909, "antithrombotic": 0.1205109},
               0.9664823),
}
# recalibration multipliers by age band [30, 45, 55, 60, 65, 70, 80) (MJA 2025 Box 5; men 30-54 all 1.1)
_AGE_EDGES = (45, 55, 60, 65, 70)
FACTOR = {"female": (1.3, 1.2, 1.0, 0.9, 0.8, 0.7), "male": (1.1, 1.1, 1.0, 0.9, 0.8, 0.8)}
_DIABETES_ARGS = ("diabetes_years", "bmi", "egfr", "hba1c", "uacr", "insulin")


def _factor(age, sex_name):
    return np.asarray(FACTOR[sex_name])[np.digitize(np.nan_to_num(age), _AGE_EDGES)]


def auscvd(age, sex, smoking, sbp, tc_hdl, diabetes, af, bp_meds, lipid_meds, antithrombotic, seifa, *,
           diabetes_years=None, bmi=None, egfr=None, hba1c=None, uacr=None, insulin=None) -> np.ndarray:
    """5-year CVD risk, ages 30-79, people without CVD.

    smoking: "never" / "ex" / "current"; tc_hdl: total/HDL cholesterol ratio; seifa: SEIFA quintile 1-5
    (1 = most disadvantaged; the calculator uses 3 when no postcode is given); flags 0/1.
    With the six diabetes arguments (years since diagnosis, BMI kg/m2, eGFR mL/min/1.73m2, HbA1c mmol/mol,
    urine ACR mg/mmol, insulin 0/1), people with diabetes get AUS-PREDICT-Diabetes, the calculator's
    default for them; without, everyone gets the general AUS-PREDICT equation (the calculator's other
    choice for people with diabetes). Rows with diabetes = 0 always use AUS-PREDICT.
    """
    given = [k for k, v in zip(_DIABETES_ARGS, (diabetes_years, bmi, egfr, hba1c, uacr, insulin)) if v is not None]
    if given and len(given) < len(_DIABETES_ARGS):
        raise ValueError(f"the diabetes equation needs all of {_DIABETES_ARGS}; got {given}")
    a = as_float(age)
    a = np.where((a >= 30) & (a < 80), a, np.nan)
    q = as_float(seifa)
    dep = np.where(np.isin(q, (1, 2, 3, 4, 5)), 6 - q, np.nan)
    dm, bpl = as_bool(diabetes), as_bool(bp_meds)
    s = _smoking(smoking)
    common = {"af": as_bool(af), "bp_meds": bpl, "lipid_meds": as_bool(lipid_meds),
              "antithrombotic": as_bool(antithrombotic)}
    raw = {"age": a, "dep": dep, "sbp": _pos(sbp), "tc_hdl": _pos(tc_hdl)}
    if given:
        yrs, r = as_float(diabetes_years), as_float(uacr)
        r = np.where(r >= 0, r, np.nan)
        raw_dm = {**raw, "years": np.where(yrs >= 0, yrs, np.nan), "bmi": _pos(bmi), "egfr": _pos(egfr),
                  "hba1c": _pos(hba1c)}
        acr = {"acr_3_30": np.where(np.isnan(r), np.nan, (r >= 3) & (r <= 30)),
               "acr_gt_30": np.where(np.isnan(r), np.nan, r > 30)}
    sx = as_sex(sex)
    out = np.nan
    for code, name in ((1, "male"), (0, "female")):
        centre, beta, s0 = GENERAL[name]
        c = {k: v - centre[k] for k, v in raw.items()}
        x = {**s, **common, **c, "diabetes": dm, "age_x_diabetes": c["age"] * dm, "age_x_sbp": c["age"] * c["sbp"],
             "bp_meds_x_sbp": bpl * c["sbp"]}
        risk = 1 - s0 ** np.exp(sum(b * x[k] for k, b in beta.items()))
        if given:
            centre, beta, s0 = DIABETES[name]
            x = {**common, **acr, **{k: v - centre[k] for k, v in raw_dm.items()}, "smoker": s["smoker"],
                 "insulin": as_bool(insulin), "constant": 1}
            risk = np.where(dm == 1, 1 - s0 ** np.exp(sum(b * x[k] for k, b in beta.items())), risk)
        out = np.where(sx == code, risk * _factor(a, name), out)
    return out
