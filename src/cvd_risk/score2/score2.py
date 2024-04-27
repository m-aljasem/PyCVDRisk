"""SCORE2, SCORE2-OP, SCORE2-Diabetes and SCORE2 Asia-Pacific: 10-year risk of fatal and non-fatal CVD.

All functions accept scalars or array-likes (broadcast like numpy) and return a
float64 array of risks as fractions in [0, 1]. A row is NaN when any input is
missing or invalid, sex/region is unrecognised, or age is outside the variant's
valid range (see coefficients.AGE_RANGE). Values are never clamped.

Units: SBP mmHg; cholesterol mmol/L; HbA1c mmol/mol; eGFR mL/min/1.73m².
Regions: "low", "moderate", "high", "very_high" (ESC risk regions).
"""

from __future__ import annotations

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex
from cvd_risk.score2 import coefficients as C


def recalibrate(uncal: np.ndarray, scale1: np.ndarray, scale2: np.ndarray) -> np.ndarray:
    """Region recalibration: 1 - exp(-exp(scale1 + scale2 * ln(-ln(1 - uncal))))."""
    return -np.expm1(-np.exp(scale1 + scale2 * np.log(-np.log1p(-uncal))))


def _by_sex(male: np.ndarray, coefs: dict, key: str) -> np.ndarray:
    return np.where(male, coefs["male"][key], coefs["female"][key])


def _scales(sex: np.ndarray, region: np.ndarray, table: dict) -> tuple[np.ndarray, np.ndarray]:
    lookup = np.full((2, len(C.REGIONS) + 1, 2), np.nan)  # last slot = unknown region
    for s, name in ((1, "male"), (0, "female")):
        lookup[s, : len(C.REGIONS)] = table[name]
    picked = lookup[np.where(sex == 1, 1, 0), region]  # region -1 → unknown slot
    picked = np.where((sex < 0)[..., None], np.nan, picked)
    return picked[..., 0], picked[..., 1]


def _prepare(age, sex, smoker, sbp, total_chol, hdl_chol, diabetes, region, variant):
    a, sb, tc, hdl = (as_float(v) for v in (age, sbp, total_chol, hdl_chol))
    sm, dm = as_bool(smoker), as_bool(diabetes)
    sx, rg = as_sex(sex), as_category(region, C.REGIONS)
    a, sb, tc, hdl, sm, dm, sx, rg = np.broadcast_arrays(a, sb, tc, hdl, sm, dm, sx, rg)
    lo, hi = C.AGE_RANGE[variant]
    a = np.where((a >= lo) & (a < hi), a, np.nan)
    return a, sx, sm, sb, tc, hdl, dm, rg


def _score2_lp(c, male, age, sm, sbp, tc, hdl, dm):
    """SCORE2 core terms (shared with SCORE2-Diabetes), with SCORE2 centring."""
    age_t, sbp_t, tc_t, hdl_t = (age - 60) / 5, (sbp - 120) / 20, tc - 6, (hdl - 1.3) / 0.5
    k = lambda key: _by_sex(male, c, key)  # noqa: E731
    lp = (
        k("age") * age_t + k("smoker") * sm + k("sbp") * sbp_t + k("diabetes") * dm
        + k("tc") * tc_t + k("hdl") * hdl_t
        + k("age_smoker") * age_t * sm + k("age_sbp") * age_t * sbp_t
        + k("age_tc") * age_t * tc_t + k("age_hdl") * age_t * hdl_t
        + k("age_diabetes") * age_t * dm
    )
    return lp, age_t


def score2(age, sex, smoker, sbp, total_chol, hdl_chol, diabetes, region) -> np.ndarray:
    """SCORE2 (ages 40-69). Source: coefficients.SCORE2_SOURCE."""
    age, sx, sm, sbp, tc, hdl, dm, rg = _prepare(
        age, sex, smoker, sbp, total_chol, hdl_chol, diabetes, region, "score2"
    )
    male = sx == 1
    lp, _ = _score2_lp(C.SCORE2, male, age, sm, sbp, tc, hdl, dm)
    uncal = -np.expm1(np.exp(lp) * np.log(_by_sex(male, C.SCORE2, "s0")))
    return recalibrate(uncal, *_scales(sx, rg, C.SCORE2_SCALES))


def score2_asia_pacific(age, sex, smoker, sbp, total_chol, hdl_chol, diabetes, region) -> np.ndarray:
    """SCORE2 Asia-Pacific (ages 40-69). Source: coefficients.SCORE2_ASIA_PACIFIC_SOURCE.

    `region` is the Asia-Pacific risk region ("low", "moderate", "high",
    "very_high") as assigned per country in the source's Supplementary Table 4.
    """
    age, sx, sm, sbp, tc, hdl, dm, rg = _prepare(
        age, sex, smoker, sbp, total_chol, hdl_chol, diabetes, region, "score2_asia_pacific"
    )
    male = sx == 1
    lp, _ = _score2_lp(C.SCORE2, male, age, sm, sbp, tc, hdl, dm)
    uncal = -np.expm1(np.exp(lp) * np.log(_by_sex(male, C.SCORE2, "s0")))
    return recalibrate(uncal, *_scales(sx, rg, C.SCORE2_ASIA_PACIFIC_SCALES))


def score2_op(age, sex, smoker, sbp, total_chol, hdl_chol, diabetes, region) -> np.ndarray:
    """SCORE2-OP (ages 70-89). Source: coefficients.SCORE2_OP_SOURCE."""
    age, sx, sm, sbp, tc, hdl, dm, rg = _prepare(
        age, sex, smoker, sbp, total_chol, hdl_chol, diabetes, region, "score2_op"
    )
    male = sx == 1
    k = lambda key: _by_sex(male, C.SCORE2_OP, key)  # noqa: E731
    age_t, sbp_t, tc_t, hdl_t = age - 73, sbp - 150, tc - 6, hdl - 1.4
    lp = (
        k("age") * age_t + k("diabetes") * dm + k("smoker") * sm + k("sbp") * sbp_t
        + k("tc") * tc_t + k("hdl") * hdl_t
        + k("age_diabetes") * age_t * dm + k("age_smoker") * age_t * sm
        + k("age_sbp") * age_t * sbp_t + k("age_tc") * age_t * tc_t
        + k("age_hdl") * age_t * hdl_t
    )
    uncal = -np.expm1(np.exp(lp - k("mean_lp")) * np.log(k("s0")))
    return recalibrate(uncal, *_scales(sx, rg, C.SCORE2_OP_SCALES))


def score2_diabetes(
    age, sex, smoker, sbp, total_chol, hdl_chol, diabetes,
    age_at_diabetes_diagnosis, hba1c, egfr, region,
) -> np.ndarray:
    """SCORE2-Diabetes (ages 40-69). Source: coefficients.SCORE2_DIABETES_SOURCE.

    `age_at_diabetes_diagnosis` is ignored (may be NaN) when diabetes == 0,
    because its term is multiplied by diabetes. HbA1c in mmol/mol.
    """
    age, sx, sm, sbp, tc, hdl, dm, rg = _prepare(
        age, sex, smoker, sbp, total_chol, hdl_chol, diabetes, region, "score2_diabetes"
    )
    agediag, a1c, gfr = np.broadcast_arrays(
        as_float(age_at_diabetes_diagnosis), as_float(hba1c), as_float(egfr), age
    )[:3]
    male = sx == 1
    c = C.SCORE2_DIABETES
    k = lambda key: _by_sex(male, c, key)  # noqa: E731
    lp, age_t = _score2_lp(c, male, age, sm, sbp, tc, hdl, dm)
    agediag_t = np.where(dm == 0, 0.0, (agediag - 50) / 5)
    a1c_t = (a1c - 31) / 9.34
    with np.errstate(divide="ignore", invalid="ignore"):
        egfr_t = np.where(gfr > 0, (np.log(gfr) - 4.5) / 0.15, np.nan)
    lp = lp + (
        k("agediag") * dm * agediag_t + k("hba1c") * a1c_t + k("egfr") * egfr_t
        + k("egfr2") * egfr_t**2 + k("age_hba1c") * a1c_t * age_t
        + k("age_egfr") * egfr_t * age_t
    )
    uncal = -np.expm1(np.exp(lp) * np.log(k("s0")))
    return recalibrate(uncal, *_scales(sx, rg, C.SCORE2_SCALES))
