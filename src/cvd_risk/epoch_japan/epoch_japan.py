"""EPOCH-JAPAN (2021): 10-year risk of death from coronary heart disease, stroke or cardiovascular disease, pooled
Japanese cohorts aged 40-79.

risk = 1 - S10^exp(sum(beta x) - sum(beta xbar)), cohort-stratified Cox models, as printed in Li Y, Yatsuya H,
Tanaka S, et al. J Atheroscler Thromb 2021;28(8):816-25: Table 2 (coefficients, S10) and Supplementary Table 2
(the formula, the means and two worked examples). x = ln age, ln SBP, ln(TC/HDL-C) (CHD only), flags and the
ln age x ln SBP and ln age x smoking interactions; the CHD and CVD models include sex, the stroke model does not.
The mean term is computed as in Supplementary Table 2, from ln(mean) and mean proportions.

Ages outside 40-79 give NaN.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = "Li Y, Yatsuya H, Tanaka S, et al. J Atheroscler Thromb 2021;28(8):816-25, Table 2, Supplementary Table 2"

OUTCOMES = ("chd", "stroke", "cvd")
S10 = dict(chd=0.9981, stroke=0.9961, cvd=0.9942)
COEF = {
    "chd": dict(ln_age=61.19918, male=0.65869, smoker=15.36389, diabetes=0.56252, proteinuria=0.58243,
                ln_sbp=46.36999, ln_tc_hdl=0.35931, ln_age_ln_sbp=-10.61448, ln_age_smoker=-3.51974),
    "stroke": dict(ln_age=37.40606, male=0.0, smoker=8.24292, diabetes=0.45679, proteinuria=0.63621,
                   ln_sbp=26.39953, ln_tc_hdl=0.0, ln_age_ln_sbp=-5.92290, ln_age_smoker=-1.84505),
    "cvd": dict(ln_age=45.54988, male=0.30093, smoker=10.64932, diabetes=0.49413, proteinuria=0.62120,
                ln_sbp=33.30729, ln_tc_hdl=0.0, ln_age_ln_sbp=-7.54461, ln_age_smoker=-2.41998),
}
# Supplementary Table 2: means (age, SBP, TC/HDL-C) and proportions
MEAN = dict(age=57.0, sbp=131.1, tc_hdl=4.1, male=0.480, smoker=0.267, diabetes=0.046, proteinuria=0.028)


def _lp(b, ln_age, ln_sbp, ln_tc_hdl, male, smoker, diabetes, proteinuria):
    return (b["ln_age"] * ln_age + b["male"] * male + b["smoker"] * smoker + b["diabetes"] * diabetes
            + b["proteinuria"] * proteinuria + b["ln_sbp"] * ln_sbp + b["ln_tc_hdl"] * ln_tc_hdl
            + b["ln_age_ln_sbp"] * ln_age * ln_sbp + b["ln_age_smoker"] * ln_age * smoker)


def epoch_japan(age, sex, sbp, smoker, diabetes, proteinuria, *, total_chol=None, hdl_chol=None,
                outcome="cvd") -> np.ndarray:
    """10-year risk (fraction) of death from outcome = "chd", "stroke" or "cvd". sbp mmHg; smoker = current smoker;
    proteinuria = dipstick 1+ or more; flags 0/1. total_chol and hdl_chol (same unit) are needed for "chd" only."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    if outcome == "chd" and (total_chol is None or hdl_chol is None):
        raise ValueError('outcome="chd" needs total_chol= and hdl_chol=')
    b, m = COEF[outcome], MEAN
    a = as_float(age)
    sx = as_sex(sex)
    ratio = as_float(total_chol) / as_float(hdl_chol) if outcome == "chd" else 1.0
    lp = _lp(b, np.log(np.where((a >= 40) & (a < 80), a, np.nan)), np.log(as_float(sbp)), np.log(ratio),
             np.where(sx == 1, 1.0, np.where(sx == 0, 0.0, np.nan)), as_bool(smoker), as_bool(diabetes),
             as_bool(proteinuria))
    mean = _lp(b, np.log(m["age"]), np.log(m["sbp"]), np.log(m["tc_hdl"]), m["male"], m["smoker"], m["diabetes"],
               m["proteinuria"])
    return 1 - S10[outcome] ** np.exp(lp - mean)
