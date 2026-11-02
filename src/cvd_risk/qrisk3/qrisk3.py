"""QRISK3-2017: 10-year risk of CVD (CHD, ischaemic stroke, TIA) for ages 25-84.

A vectorised port of ClinRisk's open-source C implementation. Copyright 2017 ClinRisk
Ltd.; LGPL-3.0-or-later, as is this file (see LICENSE in this folder). ClinRisk's terms
require this disclaimer to be held with any score this code produces:

    The initial version of this file, to be found at http://svn.clinrisk.co.uk/opensource/qrisk2,
    faithfully implements QRISK3-2017. ClinRisk Ltd. have released this code under the GNU Lesser
    General Public License to enable others to implement the algorithm faithfully. However, the
    nature of the GNU Lesser General Public License is such that we cannot prevent, for example,
    someone accidentally altering the coefficients, getting the inputs wrong, or just poor
    programming. ClinRisk Ltd. stress, therefore, that it is the responsibility of the end user to
    check that the source that they receive produces the same results as the original code found at
    https://qrisk.org. Inaccurate implementations of risk scores can lead to wrong patients being
    given the wrong treatment.

Accepts scalars or array-likes (broadcast like numpy) and returns a float64 array of
risks as fractions in [0, 1]. Age is truncated to whole years, as ClinRisk's code takes
an integer. A row is NaN when an input is missing or invalid, a category is unrecognised,
or age is outside 25-84. The calculator's imputation of missing BMI, cholesterol ratio
and SBP is not part of ClinRisk's open-source release, so those inputs are required.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex
from cvd_risk.qrisk3.coefficients import MODELS

ETHNICITY = ("white_or_not_stated", "indian", "pakistani", "bangladeshi", "other_asian",
             "black_caribbean", "black_african", "chinese", "other")
SMOKING = ("non", "ex", "light", "moderate", "heavy")  # light <10/day, moderate 10-19, heavy 20+
DIABETES = ("none", "type1", "type2")
AGE_RANGE = (25, 85)


def _linear_predictor(m, age, bmi, x):
    p1, p2 = m["age_powers"]
    dage, dbmi = age / 10, bmi / 10
    mu = m["means"]
    f = dict(x, age_1=dage ** p1 - mu["age_1"], age_2=dage ** p2 - mu["age_2"],
             bmi_1=dbmi ** -2 - mu["bmi_1"], bmi_2=dbmi ** -2 * np.log(dbmi) - mu["bmi_2"])
    for k in mu.keys() - {"age_1", "age_2", "bmi_1", "bmi_2"}:
        f[k] = x[k] - mu[k]
    lp = np.take(m["ethnicity"], x["ethrisk"]) + np.take(m["smoking"], x["smoke_cat"])
    for parts, coef in m["terms"].items():
        lp = lp + np.prod([f[p] for p in parts], axis=0) * coef
    return lp


def qrisk3(
    age, sex, ethnicity, smoking, bmi, chol_hdl_ratio, sbp, sbp_sd, *, townsend=0.0,
    diabetes="none", af=0, atypical_antipsychotics=0, corticosteroids=0, erectile_dysfunction=0,
    migraine=0, rheumatoid_arthritis=0, ckd=0, severe_mental_illness=0, sle=0, bp_treated=0,
    family_history=0,
) -> np.ndarray:
    """QRISK3-2017 10-year CVD risk. Source: coefficients.SOURCE.

    ethnicity: one of ETHNICITY. smoking: one of SMOKING. diabetes: one of DIABETES.
    sbp_sd: standard deviation of at least two recent SBP readings (mmHg).
    townsend: Townsend deprivation score (standardised, 0 = UK average).
    ckd: chronic kidney disease stage 3-5. erectile_dysfunction is ignored for women.
    The remaining flags are 0/1, as defined in Hippisley-Cox et al., BMJ 2017;357:j2099.
    """
    eth, smk, dm = (as_category(ethnicity, ETHNICITY), as_category(smoking, SMOKING),
                    as_category(diabetes, DIABETES))
    flags = [as_bool(v) for v in (af, atypical_antipsychotics, corticosteroids, erectile_dysfunction,
                                  migraine, rheumatoid_arthritis, ckd, severe_mental_illness, sle,
                                  bp_treated, family_history)]
    a, sx, b, r, s, sd, t, eth, smk, dm, *flags = np.broadcast_arrays(
        np.floor(as_float(age)), as_sex(sex), as_float(bmi), as_float(chol_hdl_ratio),
        as_float(sbp), as_float(sbp_sd), as_float(townsend), eth, smk, dm, *flags)

    lo, hi = AGE_RANGE
    ok = ((a >= lo) & (a < hi) & (sx >= 0) & (eth >= 0) & (smk >= 0) & (dm >= 0)
          & (b > 0) & (r > 0) & (s > 0) & (sd >= 0) & np.isfinite(t))
    x = dict(zip(("b_AF", "b_atypicalantipsy", "b_corticosteroids", "b_impotence2", "b_migraine",
                  "b_ra", "b_renal", "b_semi", "b_sle", "b_treatedhyp", "fh_cvd"), flags))
    x.update(b_type1=(dm == 1).astype(float), b_type2=(dm == 2).astype(float),
             rati=r, sbp=s, sbps5=sd, town=t, ethrisk=np.where(ok, eth + 1, 0),
             smoke_cat=np.where(ok, smk, 0), **{f"smoke_{j}": (smk == j).astype(float) for j in range(1, 5)})
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        bb = np.where(ok, b, 25.0)  # placeholder for invalid rows (masked below)
        lp = np.where(sx == 1, _linear_predictor(MODELS["male"], a, bb, x),
                      _linear_predictor(MODELS["female"], a, bb, x))
        s0 = np.where(sx == 1, MODELS["male"]["survivor"], MODELS["female"]["survivor"])
        risk = -np.expm1(np.exp(lp) * np.log(s0))  # 1 - s0 ** exp(lp)
    return np.where(ok, risk, np.nan)
