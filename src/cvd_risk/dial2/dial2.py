"""DIAL2: 10-year and lifetime risk of MI, stroke or CVD death, and CVD-free life expectancy, for
people with type 2 diabetes without CVD. Østergaard et al., Eur J Prev Cardiol 2023 (core model).

A yearly life table from the person's age to 95 (supplement, Box 5D). In each year of attained age
t the CVD risk and the competing non-CVD death risk are 1 - S0(t)^exp(LP), with the age interactions
evaluated at age t, each recalibrated to the ESC risk region (low or moderate; the paper has no
scales for high and very high) by 1 - exp(-exp(scale1 + scale2 * ln(-ln(1 - risk)))).

- risk10: risk of a CVD event in the next 10 years, sum of G(t) * CVD risk(t).
- lifetime_risk: the same sum to age 95.
- life_expectancy: median CVD-free survival, the age at which G first falls below 0.5 (linear
  within the year); inf if G is still >= 0.5 at 95.

Optional lifelong treatment effects as in the paper's Figure 3: each yearly risk becomes
1 - (1 - risk)^HR, with hr_cvd on CVD (the paper: 0.80 per 10 mmHg SBP lowering, 0.78 per
1 mmol/L LDL lowering) and hr_non_cvd on non-CVD death.

Age must be a whole number of years: 30-85 for risk10, 30-94 otherwise. HbA1c in mmol/mol, eGFR in
mL/min/1.73 m2, cholesterol in mmol/L, SBP in mmHg.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex
from cvd_risk._lifetable import lifetime_risk, median_age, survival
from cvd_risk.dial2.coefficients import BASELINE, BETA, SCALES
from cvd_risk.score2.score2 import recalibrate

REGION_NAMES = ("low", "moderate")
OUTCOMES = ("risk10", "lifetime_risk", "life_expectancy")
_AGES = np.arange(30, 95)
_T = (_AGES - 63) / 12.0  # age interaction term at each attained age
_S0 = np.array([BASELINE[a] for a in _AGES], dtype=float)  # (age, outcome, sex: 0 male / 1 female)
_SEXES = ("female", "male")  # index = as_sex code
_TABLES = {o: (np.array([BETA[s][o] for s in _SEXES]),
               np.array([[SCALES[o][r][s] for r in REGION_NAMES] for s in _SEXES])) for o in ("cvd", "non_cvd")}


def _yearly_risks(outcome, sx, reg, x, hr):
    beta, scales = _TABLES[outcome]
    lp = np.einsum("...k,...tk->...t", beta[sx], x)
    s0 = _S0[:, 0 if outcome == "cvd" else 1, :].T[np.where(sx == 1, 0, 1)]
    sc = scales[sx, reg]
    risk = recalibrate(1 - s0 ** np.exp(lp), sc[..., :1], sc[..., 1:])
    return 1 - (1 - risk) ** hr[..., None]


def dial2(age, sex, smoker, sbp, total_chol, hdl_chol, hba1c, egfr, age_at_diagnosis, region, *,
          outcome="lifetime_risk", hr_cvd=1.0, hr_non_cvd=1.0) -> np.ndarray:
    """DIAL2 (core model) 10-year or lifetime CVD risk (fraction) or CVD-free life expectancy (years).

    region: "low" or "moderate" ESC risk region. age_at_diagnosis: age at type 2 diabetes
    diagnosis (<= age). hr_cvd, hr_non_cvd: lifelong treatment hazard ratios (default 1).
    Source: coefficients.SOURCE.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    a, sx, sm, s, tc, hdl, a1c, gfr, dx, reg, hc, hn = np.broadcast_arrays(
        as_float(age), as_sex(sex), as_bool(smoker), as_float(sbp), as_float(total_chol), as_float(hdl_chol),
        as_float(hba1c), as_float(egfr), as_float(age_at_diagnosis), as_category(region, REGION_NAMES),
        as_float(hr_cvd), as_float(hr_non_cvd))
    positive = [s, tc, hdl, a1c, gfr, dx, hc, hn]
    ok = ((a == np.floor(a)) & (a >= 30) & (a <= (85 if outcome == "risk10" else 94)) & (sx >= 0) & (reg >= 0)
          & (sm >= 0) & (dx <= a) & np.all([np.isfinite(v) & (v > 0) for v in positive], axis=0))
    z = lambda v, fill: np.where(ok, v, fill)
    sx, reg = z(sx, 0).astype(int), z(reg, 0).astype(int)

    main = np.stack([z(sm, 0), (z(s, 138) - 138) / 17, (np.log(z(gfr, 1)) - 4.4) / 0.26, (z(tc, 5.1) - 5.1) / 1.1,
                     (z(hdl, 1.3) - 1.3) / 0.4, (z(a1c, 55) - 55) / 16, (z(dx, 58) - 58) / 12], axis=-1)
    tt = np.broadcast_to(_T, sx.shape + _T.shape)[..., None]
    mm = np.broadcast_to(main[..., None, :], sx.shape + _T.shape + (7,))
    x = np.concatenate([mm, mm * tt], axis=-1)  # TERMS order

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h = _yearly_risks("cvd", sx, reg, x, z(hc, 1.0))
        i = _yearly_risks("non_cvd", sx, reg, x, z(hn, 1.0))
    start = z(a, 30.0)[..., None]
    ages, g = survival(h, i, _AGES, start)
    if outcome == "life_expectancy":
        out = median_age(g, ages, start)
    else:
        out = lifetime_risk(h, g, _AGES, start, start + 10 if outcome == "risk10" else 95)
    return np.where(ok, out, np.nan)
