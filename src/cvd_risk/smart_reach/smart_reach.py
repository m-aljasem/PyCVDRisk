"""SMART-REACH: 10-year risk of recurrent MI, stroke or CVD death, and life expectancy free of
these events, for people with established atherosclerotic disease (CAD, cerebrovascular
disease and/or PAD). Kaasenbrood et al., J Am Heart Assoc 2018.

A yearly life table from the person's age to 90 with two competing outcomes, as in the
authors' calculator: each year the event-free survivors G either have a CVD event (risk H) or
die of another cause (risk I); 10-year risk is the CVD share of the losses over 10 years, and
life expectancy free of events is the age at which G crosses 0.5 (linear within the year).

Age must be a whole number of years, as in the calculator's table (45-80 for the 10-year
risk, 45-89 for life expectancy). Life expectancy beyond the table is returned as inf (the
calculator shows ">90"). A row is NaN when an input is invalid or no disease location is given.

When a person's two yearly risks sum to more than 1 at some age, G turns negative from that age
(the two models were fitted separately); the 10-year risk follows the calculator's arithmetic there.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex
from cvd_risk.smart_reach.coefficients import BASELINE, CVD, NON_CVD, REGIONS

REGION_NAMES = tuple(REGIONS)
OUTCOMES = ("risk10", "life_expectancy")
_AGES = np.arange(45, 91)  # life-table rows; row 90 only closes the table
_S = np.array([BASELINE.get(a, (0.0, 0.0)) for a in _AGES])  # 90 has no baseline (unused)


def _lp(c, x, age):
    return (c["male"] * x["male"] + c["smoker"] * x["smoker"] + c["diabetes"] * x["dm"]
            + c["sbp"] * x["sbp"] + c["sbp_sq"] * x["sbp"] ** 2 + c["tc"] * x["tc"] + c["tc_sq"] * x["tc"] ** 2
            + c["creatinine"] * x["cr"] + c["creatinine_sq"] * x["cr"] ** 2
            + c["two_locations"] * (x["n"] == 2) + c["three_locations"] * (x["n"] == 3)
            + c["af"] * x["af"] + c["chf"] * x["chf"] + c["smoker_age"] * x["smoker"] * age)


def _life_table(age, sex, smoker, diabetes, sbp, total_chol, creatinine, cad, cerebrovascular, pad,
                af, heart_failure, region, max_age):
    """(ok, start age, H, I, G): yearly CVD risk H, non-CVD death risk I and event-free survival G
    at the start of each year of age 45-90 (0 before the person's age), one row per person."""
    reg = as_category(region, REGION_NAMES)
    flags = [as_bool(v) for v in (smoker, diabetes, cad, cerebrovascular, pad, af, heart_failure)]
    a, sx, s, tc, cr, reg, sm, dm, l1, l2, l3, af_, chf = np.broadcast_arrays(
        as_float(age), as_sex(sex), as_float(sbp), as_float(total_chol), as_float(creatinine), reg, *flags)
    n = l1 + l2 + l3
    ok = ((a == np.floor(a)) & (a >= 45) & (a <= max_age) & (sx >= 0) & (reg >= 0) & (n >= 1)
          & np.isfinite(s) & np.isfinite(tc) & np.isfinite(cr) & (s > 0) & (tc > 0) & (cr > 0)
          & np.all([f >= 0 for f in (sm, dm, af_, chf, l1, l2, l3)], axis=0))

    x = {k: np.where(ok, v, 0.0)[..., None] for k, v in
         dict(male=(sx == 1) * 1.0, smoker=sm, dm=dm, sbp=s, tc=tc, cr=cr, n=n, af=af_, chf=chf).items()}
    shift = np.array(list(REGIONS.values()))[np.where(ok, reg, 0)]
    ages = _AGES.astype(float)
    with np.errstate(over="ignore", invalid="ignore"):
        h = 1 - _S[:, 0] ** np.exp(_lp(CVD, x, ages) + shift[..., :1])
        i = 1 - _S[:, 1] ** np.exp(_lp(NON_CVD, x, ages) + shift[..., 1:])
    start = np.where(ok, a, 45.0)[..., None]
    step = np.where(ages > start, 1 - np.roll(h + i, 1, axis=-1), 1.0)
    g = np.where(ages >= start, np.cumprod(step, axis=-1), 0.0)
    return ok, start, h, i, g


def smart_reach(age, sex, smoker, diabetes, sbp, total_chol, creatinine, cad, cerebrovascular, pad,
                *, af=0, heart_failure=0, region="western_europe", outcome="risk10") -> np.ndarray:
    """SMART-REACH 10-year risk (fraction) or CVD-free life expectancy (years). Source: coefficients.SOURCE.

    cad, cerebrovascular, pad: 0/1 disease locations (at least one). region: one of REGION_NAMES.
    outcome: "risk10" or "life_expectancy".
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}")
    ok, start, h, i, g = _life_table(age, sex, smoker, diabetes, sbp, total_chol, creatinine, cad, cerebrovascular,
                                     pad, af, heart_failure, region, 80 if outcome == "risk10" else 89)
    ages = _AGES.astype(float)
    if outcome == "risk10":
        g_next = np.concatenate([g[..., 1:], np.zeros(g.shape[:-1] + (1,))], axis=-1)
        share = np.divide(h, h + i, out=np.zeros_like(h), where=(h + i) != 0)
        window = (ages >= start) & (ages < start + 10)
        out = np.sum(np.where(window, share * (g - g_next), 0.0), axis=-1)
    else:
        # Last age with G >= 0.5: the first crossing of 0.5. (The calculator counts every row
        # not in (0, 0.5), which misplaces the median when G later turns negative; see REPORT.)
        m = np.argmax((ages >= start) & (g < 0.5), axis=-1) + 44
        m = np.where(((ages >= start) & (g < 0.5)).any(axis=-1), m, 90)
        k = np.clip(m - 45, 0, len(_AGES) - 2)[..., None]
        gm, gm1 = np.take_along_axis(g, k, -1)[..., 0], np.take_along_axis(g, k + 1, -1)[..., 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(m < 90, m + (gm - 0.5) / (gm - gm1), np.inf)
    return np.where(ok, out, np.nan)
