"""China-PAR (Prediction for ASCVD Risk in China): 10-year risk of a first ASCVD event (non-fatal acute MI or CHD
death, or fatal or non-fatal stroke), Chinese adults without ASCVD.

risk = 1 - exp(-exp(intercept + sum(beta * x))), sex-specific Cox models; x = ln(age), ln(SBP) (treated or untreated),
ln(TC), ln(HDL), ln(waist), flags, and ln(age) interactions, as in Yang X, Li J, Hu D, et al. Circulation
2016;134:1430-40, Supplemental Table 1. The intercept stands for ln(-ln S10) - mean(coefficient x value).

That table prints the coefficients to 2 decimals, S10 as 0.97 / 0.99 and the means to 2 decimals; at face value
they miss the official calculator (cvdrisk.com.cn, National Center for Cardiovascular Diseases) by up to 12.7
points. COEF below is a set within 0.005 of every printed coefficient (so it rounds to the paper) that reproduces
every displayed calculator value (550/550 patients). A set recovered from the first 400 alone matched 144 of 150
held-out patients, the other 6 within 0.1 point at a rounding edge (reference/china_par/REPORT.md). It is not the developers' unpublished full-precision set.

Inputs outside the calculator's form ranges (age 20-85, waist 50-130 cm, TC 80-400 and HDL 20-130 mg/dL, i.e.
about 2.07-10.36 and 0.52-3.37 mmol/L,
SBP 70-200 mmHg) give NaN. The model was derived at ages 35-74.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = ("Yang X, Li J, Hu D, et al. Predicting the 10-year risks of atherosclerotic cardiovascular disease in "
          "Chinese population: the China-PAR project. Circulation 2016;134:1430-40 (Supplemental Table 1); "
          "www.cvdrisk.com.cn/ASCVD/Eval")

MG_PER_MMOL = 1 / 0.0259  # Supplemental Table 1 footnote: mg/dL x 0.0259 = mmol/L

# fixtures/recovered_coefficients.json (reference/china_par/fit_calculator.py), rounded to 6 decimals
COEF = {
    "male": dict(intercept=-144.193788, ln_age=31.965, ln_sbp_treated=27.386684, ln_sbp_untreated=26.146298,
                 ln_tc=0.621542, ln_hdl=-0.695, ln_waist=-0.711884, smoker=3.956455, diabetes=0.356658,
                 north=0.475088, urban=-0.164075, family_history=6.223529, ln_age_ln_sbp_treated=-6.017496,
                 ln_age_ln_sbp_untreated=-5.730422, ln_age_smoker=-0.939471, ln_age_family_history=-1.534604),
    "female": dict(intercept=-121.4338, ln_age=24.86799, ln_sbp_treated=20.705, ln_sbp_untreated=19.977185,
                   ln_tc=0.057857, ln_hdl=-0.217331, ln_waist=1.475969, smoker=0.495, diabetes=0.566733,
                   north=0.543871, ln_age_ln_sbp_treated=-4.52705, ln_age_ln_sbp_untreated=-4.358851),
}


def _ln_within(x, lo, hi):
    x = as_float(x)
    return np.log(np.where((x >= lo) & (x <= hi), x, np.nan))


def china_par(age, sex, sbp, bp_treated, total_chol, hdl_chol, waist, smoker, diabetes, north, urban,
              family_history) -> np.ndarray:
    """10-year ASCVD risk (fraction). sbp mmHg; total_chol, hdl_chol mmol/L; waist cm; smoker = current smoker;
    north = lives in northern China (north of the Yangtze); urban = lives in an urban area; family_history = a parent
    or sibling with MI or stroke; flags 0/1. Women's equation has no urban or family-history terms."""
    la, ls = _ln_within(age, 20, 85), _ln_within(sbp, 70, 200)
    t, smk, fh = as_bool(bp_treated), as_bool(smoker), as_bool(family_history)
    x = dict(ln_age=la, ln_sbp_treated=ls * t, ln_sbp_untreated=ls * (1 - t), ln_tc=_ln_within(as_float(total_chol) * MG_PER_MMOL, 80, 400),
             ln_hdl=_ln_within(as_float(hdl_chol) * MG_PER_MMOL, 20, 130), ln_waist=_ln_within(waist, 50, 130), smoker=smk,
             diabetes=as_bool(diabetes), north=as_bool(north), urban=as_bool(urban), family_history=fh,
             ln_age_ln_sbp_treated=la * ls * t, ln_age_ln_sbp_untreated=la * ls * (1 - t), ln_age_smoker=la * smk,
             ln_age_family_history=la * fh)
    sx = as_sex(sex)
    out = np.nan
    for code, name in ((1, "male"), (0, "female")):
        c = COEF[name]
        lp = c["intercept"] + sum(b * x[k] for k, b in c.items() if k != "intercept")
        out = np.where(sx == code, 1 - np.exp(-np.exp(lp)), out)
    return out
