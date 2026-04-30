"""ASSIGN 10-year risk of a first cardiovascular event (CHD, stroke, TIA) in Scotland.

Cox model with sex-specific betas: A = sum(beta x) - sum(beta x_mean); risk = 1 - S0 ** exp(A).
version=1: Woodward 2007 (Appendix 1): SHHEC means and 10-year survival 0.8831 (men) / 0.9365 (women).
version=2: Welsh 2025 (Supplementary Results, equations 1-2): same betas, recentred on the means of the
2006-2010 derivation cohort with survival 0.9130 / 0.9666.
Inputs: cholesterol mmol/L, SIMD score on its 0-100 scale (divided by 10 inside; version 1 used SIMD 2004,
version 2 SIMD 2020), cigarettes per day (0 for non-smokers), diabetes and family history 0/1. Returns
percent, NaN for a row with a missing or invalid input, an unrecognised sex, or age outside 30-74.
Women's family-history beta is 0.49159 in version 1 (2007 appendix) and 0.492 in version 2 (2025 supplement and the
official calculator). The 2025 centring lines print men's cholesterol beta as 0.2286; the official calculator uses
0.22286, as here (reference/assign/REPORT.md).
ASSIGN 1.5 (developers' 2013 update, run by the offline calculator 1.5.1) keeps the version-1 algorithm and scores
rheumatoid arthritis as 10 extra cigarettes per day: pass cigarettes_per_day + 10 for such patients with version=1.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = ("Woodward M, Brindle P, Tunstall-Pedoe H. Heart 2007;93:172-6, Appendix 1; "
          "Welsh P, Kimenai DM, Woodward M. Heart 2025;111:557-64, Supplementary Results")
AGE_RANGE = (30, 74)
# order: age, total cholesterol, HDL, SBP, diabetes, family history, cigarettes/day, SIMD score / 10
_M = (0.05698, 0.22286, -0.53684, 0.01183, 0.81558, 0.27500, 0.02005, 0.06296)
# women's family history: 0.49159 in the 2007 appendix, 0.492 in the 2025 supplement and the official v2 calculator
BETA = {1: {"male": _M, "female": (0.07203, 0.12720, -0.55836, 0.01064, 0.97727, 0.49159, 0.02724, 0.09386)},
        2: {"male": _M, "female": (0.07203, 0.12720, -0.55836, 0.01064, 0.97727, 0.492, 0.02724, 0.09386)}}
MEAN = {1: {"male": (48.8706, 6.22520, 1.35042, 133.810, 0.0152905, 0.263762, 7.95841, 2.74038),
            "female": (48.7959, 6.40706, 1.62837, 130.115, 0.0127275, 0.326328, 6.44058, 2.82470)},
        2: {"male": (55.3999, 5.5488, 1.2944, 141.5754, 0.0481, 0.5620, 2.0560, 1.4701),
            "female": (55.1321, 5.8199, 1.6086, 135.2402, 0.0268, 0.6129, 1.6107, 1.4740)}}
SURVIVAL = {1: {"male": 0.8831, "female": 0.9365}, 2: {"male": 0.9130, "female": 0.9666}}


def assign(age, sex, total_chol, hdl_chol, sbp, diabetes, family_history, cigarettes_per_day, simd_score,
           version: int = 2) -> np.ndarray:
    """ASSIGN 10-year cardiovascular risk (%). Source: SOURCE."""
    if version not in MEAN:
        raise ValueError("version must be 1 or 2")
    x = np.broadcast_arrays(as_float(age), as_sex(sex), as_float(total_chol), as_float(hdl_chol), as_float(sbp),
                            as_bool(diabetes), as_bool(family_history), as_float(cigarettes_per_day),
                            as_float(simd_score))
    a, sx = x[0], x[1]
    xs = [a, *x[2:]]
    xs[-1] = xs[-1] / 10
    ok = (sx >= 0) & (a >= AGE_RANGE[0]) & (a <= AGE_RANGE[1]) & (xs[1] > 0) & (xs[2] > 0) & (xs[3] > 0) \
        & (xs[6] >= 0) & (xs[7] >= 0)
    risk = np.full(a.shape, np.nan)
    with np.errstate(invalid="ignore", over="ignore"):
        for sex_name, male in (("male", True), ("female", False)):
            z = sum(b * (v - m) for b, v, m in zip(BETA[version][sex_name], xs, MEAN[version][sex_name]))
            r = 100 * (1 - SURVIVAL[version][sex_name] ** np.exp(z))
            risk = np.where(ok & (sx == male), r, risk)
    return risk
