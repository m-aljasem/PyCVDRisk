"""QRISK2-2014: 10-year risk of CVD for ages 25-84.

A vectorised port of ClinRisk's open-source C implementation. Copyright 2013 ClinRisk
Ltd.; LGPL-3.0-or-later, as is this file (see LICENSE in this folder). ClinRisk's terms
require this disclaimer to be held with any score this code produces:

    The initial version of this file, to be found at http://svn.clinrisk.co.uk/opensource/qrisk2,
    faithfully implements QRISK2-2014. ClinRisk Ltd. have released this code under the GNU Lesser
    General Public License to enable others to implement the algorithm faithfully. However, the
    nature of the GNU Lesser General Public License is such that we cannot prevent, for example,
    someone accidentally altering the coefficients, getting the inputs wrong, or just poor
    programming. ClinRisk Ltd. stress, therefore, that it is the responsibility of the end user to
    check that the source that they receive produces the same results as the original code posted
    at http://svn.clinrisk.co.uk/opensource/qrisk2. Inaccurate implementations of risk scores can
    lead to wrong patients being given the wrong treatment.

Returns a float64 array of risks as fractions. Age is truncated to whole years, as ClinRisk's
code takes an integer. A row is NaN when an input is invalid or outside ClinRisk's own
validation ranges: age 25-84, BMI 20-40, cholesterol/HDL ratio 1-12, SBP 70-210 mmHg,
Townsend -7 to 11 (all inclusive).
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex
from cvd_risk.qrisk2.coefficients import MODELS
from cvd_risk.qrisk3.qrisk3 import DIABETES, ETHNICITY, SMOKING, _linear_predictor

RANGES = {"age": (25, 84), "bmi": (20, 40), "chol_hdl_ratio": (1, 12), "sbp": (70, 210), "townsend": (-7, 11)}


def qrisk2(
    age, sex, ethnicity, smoking, bmi, chol_hdl_ratio, sbp, *, townsend=0.0, diabetes="none",
    af=0, rheumatoid_arthritis=0, ckd=0, bp_treated=0, family_history=0,
) -> np.ndarray:
    """QRISK2-2014 10-year CVD risk. Source: coefficients.SOURCE.

    ethnicity, smoking, diabetes: as in qrisk3 (ETHNICITY, SMOKING, DIABETES).
    ckd: chronic kidney disease. Flags are 0/1, as defined by ClinRisk (https://qrisk.org/2014).
    """
    eth, smk, dm = (as_category(ethnicity, ETHNICITY), as_category(smoking, SMOKING),
                    as_category(diabetes, DIABETES))
    flags = [as_bool(v) for v in (af, rheumatoid_arthritis, ckd, bp_treated, family_history)]
    a, sx, b, r, s, t, eth, smk, dm, *flags = np.broadcast_arrays(
        np.floor(as_float(age)), as_sex(sex), as_float(bmi), as_float(chol_hdl_ratio),
        as_float(sbp), as_float(townsend), eth, smk, dm, *flags)

    ok = (sx >= 0) & (eth >= 0) & (smk >= 0) & (dm >= 0) & np.all([f >= 0 for f in flags], axis=0)
    for v, (lo, hi) in zip((a, b, r, s, t), RANGES.values()):
        ok &= (v >= lo) & (v <= hi)
    x = dict(zip(("b_AF", "b_ra", "b_renal", "b_treatedhyp", "fh_cvd"), flags))
    x.update(b_type1=(dm == 1).astype(float), b_type2=(dm == 2).astype(float), rati=r, sbp=s, town=t,
             ethrisk=np.where(ok, eth + 1, 0), smoke_cat=np.where(ok, smk, 0),
             **{f"smoke_{j}": (smk == j).astype(float) for j in range(1, 5)})
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        aa, bb = np.where(ok, a, 50.0), np.where(ok, b, 25.0)  # placeholders for invalid rows (masked below)
        lp = np.where(sx == 1, _linear_predictor(MODELS["male"], aa, bb, x),
                      _linear_predictor(MODELS["female"], aa, bb, x))
        s0 = np.where(sx == 1, MODELS["male"]["survivor"], MODELS["female"]["survivor"])
        risk = -np.expm1(np.exp(lp) * np.log(s0))  # 1 - s0 ** exp(lp)
    return np.where(ok, risk, np.nan)
