"""LIFE-CVD2: lifetime risk of MI, stroke or CVD death (to age 80) and CVD-free life expectancy
for people without established CVD. Hageman et al., Eur J Prev Cardiol 2024.

A yearly life table from the person's age to 100. In each year of attained age t the model
gives a CVD risk and a competing non-CVD death risk, 1 - S0(t)^exp(LP), with the linear
predictors evaluated at age t (age enters every year, as the published charts require), each
recalibrated to the ESC risk region by 1 - exp(-exp(scale1 + scale2 * ln(-ln(1 - risk)))).
Event-free survival G starts at 1 and loses both risks each year.

- lifetime_risk: sum over t from the person's age to 79 of G(t) * CVD risk(t).
- life_expectancy: median CVD-free survival, the age at which G crosses 0.5, linear within the
  year; inf when G is still >= 0.5 at 101 (the supplement's last-survival rule for treatment
  benefit beyond the table is not implemented).

Optional lifelong treatment effects, as in the supplement's benefit estimates: each yearly risk
becomes 1 - (1 - risk)^HR (applied after recalibration), with hr_cvd on CVD and hr_non_cvd on
non-CVD death. The supplement uses HR 0.80 per 10 mmHg SBP lowering, 0.78 per 1 mmol/L LDL
lowering, and 0.60 / 0.73 for smoking cessation.

Age must be a whole number of years: 35-79 for lifetime risk, 35-99 for life expectancy.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex
from cvd_risk._lifetable import lifetime_risk, median_age, survival
from cvd_risk.life_cvd2.coefficients import BASELINE, BETA, SCALES
from cvd_risk.score2.score2 import recalibrate

REGION_NAMES = ("low", "moderate", "high", "very_high")
OUTCOMES = ("lifetime_risk", "life_expectancy")
_AGES = np.arange(35, 101)
_T = (_AGES - 60) / 5.0  # age term at each attained age
_S0 = np.array([BASELINE[a] for a in _AGES])  # (age, outcome, sex: 0 male / 1 female)
_SEXES = ("female", "male")  # index = as_sex code


def _table(outcome):
    """beta (sex, term) and scales (sex, region, 2) as arrays indexed by as_sex code."""
    beta = np.array([BETA[s][outcome] for s in _SEXES])
    scales = np.array([[SCALES[outcome][r][s] for r in REGION_NAMES] for s in _SEXES])
    return beta, scales


_TABLES = {o: _table(o) for o in ("cvd", "non_cvd")}


def _yearly_risks(outcome, sx, reg, x, hr):
    beta, scales = _TABLES[outcome]
    lp = np.einsum("...k,...tk->...t", beta[sx], x)
    s0 = _S0[:, 0 if outcome == "cvd" else 1, :].T[np.where(sx == 1, 0, 1)]  # (..., age)
    risk = 1 - s0 ** np.exp(lp)
    sc = scales[sx, reg]
    risk = recalibrate(risk, sc[..., :1], sc[..., 1:])
    return 1 - (1 - risk) ** hr[..., None]


def life_cvd2(age, sex, smoker, sbp, total_chol, hdl_chol, region, *, outcome="lifetime_risk",
              hr_cvd=1.0, hr_non_cvd=1.0) -> np.ndarray:
    """LIFE-CVD2 lifetime CVD risk to age 80 (fraction) or median CVD-free life expectancy (years).

    sbp mmHg; total_chol, hdl_chol mmol/L; region: ESC risk region, one of REGION_NAMES.
    hr_cvd, hr_non_cvd: lifelong treatment hazard ratios (default 1, no treatment).
    Source: coefficients.SOURCE.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    last = 79 if outcome == "lifetime_risk" else 99
    a, sx, sm, s, tc, hdl, reg, hc, hn = np.broadcast_arrays(
        as_float(age), as_sex(sex), as_bool(smoker), as_float(sbp), as_float(total_chol), as_float(hdl_chol),
        as_category(region, REGION_NAMES), as_float(hr_cvd), as_float(hr_non_cvd))
    ok = ((a == np.floor(a)) & (a >= 35) & (a <= last) & (sx >= 0) & (reg >= 0) & (sm >= 0)
          & (s > 0) & (tc > 0) & (hdl > 0) & (hc > 0) & (hn > 0)
          & np.isfinite(s) & np.isfinite(tc) & np.isfinite(hdl) & np.isfinite(hc) & np.isfinite(hn))
    z = lambda v, fill=0.0: np.where(ok, v, fill)
    sx, reg = z(sx, 0).astype(int), z(reg, 0).astype(int)

    main = np.stack([z(sm), (z(s, 120) - 120) / 20, z(tc, 6) - 6, (z(hdl, 1.3) - 1.3) / 0.5], axis=-1)
    tt = np.broadcast_to(_T, sx.shape + _T.shape)[..., None]
    mm = np.broadcast_to(main[..., None, :], sx.shape + _T.shape + (4,))
    x = np.concatenate([tt, mm, mm * tt], axis=-1)  # TERMS order

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h = _yearly_risks("cvd", sx, reg, x, z(hc, 1.0))
        i = _yearly_risks("non_cvd", sx, reg, x, z(hn, 1.0))
    start = z(a, 35.0)[..., None]
    ages, g = survival(h, i, _AGES, start)
    if outcome == "lifetime_risk":
        out = lifetime_risk(h, g, _AGES, start, 80)
    else:
        out = median_age(g, ages, start)
    return np.where(ok, out, np.nan)
