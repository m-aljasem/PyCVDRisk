"""SCORE2 / SCORE2-OP with CKD add-ons (eGFR, eGFR + ACR, eGFR + dipstick).

Same input conventions as cvd_risk.score2. The base model is SCORE2 for ages 40-69
and SCORE2-OP for 70-89; the add-on rescales it on the cumulative-hazard scale:
risk_new = 1 - (1 - risk) ** exp(delta). eGFR mL/min/1.73m²; ACR mg/g (must be > 0);
dipstick one of "negative", "trace", "1+", "2+", "3+" (3+ means "3+ or more").
Source: coefficients.SCORE2_CKD_SOURCE.
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex
from cvd_risk.score2 import coefficients as C
from cvd_risk.score2.score2 import score2, score2_op


def _lin(c, female, a, tc, hdl, sbp, dm, sm):
    return (c["constant"] + c["age"] * a + c["female"] * female + c["tc"] * tc + c["hdl"] * hdl
            + c["sbp"] * sbp + c["diabetes"] * dm + c["smoker"] * sm + c["age_tc"] * a * tc
            + c["age_hdl"] * a * hdl + c["age_sbp"] * a * sbp + c["age_diabetes"] * a * dm
            + c["age_smoker"] * a * sm)


def _spline(egfr):
    """eGFR segments per -15 mL/min: < 60, 60-90, >= 90."""
    return (np.minimum(egfr, 60) / -15, np.clip(egfr - 60, 0, 30) / -15, np.maximum(egfr - 90, 0) / -15)


def score2_ckd(
    age, sex, smoker, sbp, total_chol, hdl_chol, diabetes, egfr, region,
    *, acr=None, dipstick=None, addon: str = "egfr",
) -> np.ndarray:
    """SCORE2/SCORE2-OP + CKD add-on. addon: "egfr", "acr" (eGFR + ACR) or "dipstick" (eGFR + dipstick)."""
    if addon not in ("egfr", "acr", "dipstick"):
        raise ValueError('addon must be "egfr", "acr" or "dipstick"')
    base_args = (age, sex, smoker, sbp, total_chol, hdl_chol, diabetes, region)
    old = score2_op(*base_args)
    base = np.where(np.isnan(old), score2(*base_args), old)
    return ckd_addon(base, age, sex, smoker, sbp, total_chol, hdl_chol, diabetes, egfr,
                     acr=acr, dipstick=dipstick, addon=addon)


def ckd_addon(
    base_risk, age, sex, smoker, sbp, total_chol, hdl_chol, diabetes, egfr,
    *, acr=None, dipstick=None, addon: str = "egfr",
) -> np.ndarray:
    """Apply a CKD add-on to a given SCORE2 (age < 70) or SCORE2-OP (age >= 70) risk."""
    base = as_float(base_risk)
    a_, sx, sm, sb, tc, hdl, dm, gfr = np.broadcast_arrays(
        as_float(age), as_sex(sex), as_bool(smoker), as_float(sbp), as_float(total_chol),
        as_float(hdl_chol), as_bool(diabetes), as_float(egfr))
    base = np.broadcast_to(base, a_.shape)
    female = (sx == 0) * 1.0
    a = (a_ - 60) / 5
    t = (a, tc - 6, (hdl - 1.3) / 0.5, (sb - 120) / 20, dm, sm)
    ex_egfr = _lin(C.CKD_EXPECTED_EGFR, female, *t)
    obs, exp_ = _spline(gfr), _spline(ex_egfr)
    d = [o - e for o, e in zip(obs, exp_)]
    op = a_ >= 70
    k = lambda key: np.where(op, C.CKD_EGFR_ADDON["score2_op"][key], C.CKD_EGFR_ADDON["score2"][key])  # noqa: E731
    age_int = np.where(op, a_ - 73, a)
    delta = sum(k(s) * di + k(f"age_{s}") * age_int * di for s, di in zip(("lt60", "60_90", "ge90"), d))
    valid = gfr > 0

    if addon == "acr":
        x = as_float(np.nan if acr is None else acr)
        c = C.CKD_EXPECTED_LOG8_ACR
        log8_ex = 1 + _lin(c, female, *t) + sum(
            c[f"egfr_{s}"] * v for s, v in zip(("lt60", "60_90", "ge90"),
                                                ((np.minimum(gfr, 60) - 60) / -15, *obs[1:])))
        with np.errstate(divide="ignore", invalid="ignore"):
            log8_acr = np.log(x) / np.log(8)
        beta = np.where(op, C.CKD_ACR_ADDON["score2_op"], C.CKD_ACR_ADDON["score2"])
        delta = delta + beta * (log8_acr - log8_ex)
        valid = valid & (x > 0)
    elif addon == "dipstick":
        di = as_category(np.nan if dipstick is None else dipstick, C.DIPSTICK)
        table = np.array([C.CKD_DIPSTICK_ADDON["score2"] + (np.nan,),
                          C.CKD_DIPSTICK_ADDON["score2_op"] + (np.nan,)])
        delta = delta + table[op.astype(int), di]

    with np.errstate(invalid="ignore"):
        risk = -np.expm1(np.exp(delta) * np.log1p(-base))  # 1 - (1 - base) ** exp(delta)
    return np.where(valid, risk, np.nan)
