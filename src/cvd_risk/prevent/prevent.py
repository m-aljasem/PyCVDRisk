"""AHA PREVENT equations: 10- and 30-year risk of total CVD, ASCVD, heart failure, CHD and stroke.

Accepts scalars or array-likes (broadcast like numpy) and returns a float64 array
of risks as fractions in [0, 1]. A row is NaN when a required input is missing or
invalid, sex is unrecognised, or a value is outside the source's allowable range.
Values are never clamped.

Units: cholesterol mmol/L; SBP mmHg; BMI kg/m²; eGFR mL/min/1.73m²; UACR mg/g;
HbA1c %; SDI decile 1-10 (integer).

Optional predictors (uacr, hba1c, sdi) are used only by the models that contain
them. In those models a missing (NaN/None) value sets the source's "missing"
indicator term instead of making the row NaN, as the published equations specify.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from cvd_risk._inputs import as_bool, as_float, as_sex

SOURCE = (
    "Khan SS, Matsushita K, Sang Y, et al. Development and validation of the American Heart "
    "Association's PREVENT equations. Circulation 2024;149:430-449. "
    "doi:10.1161/CIRCULATIONAHA.123.067626, Supplemental Tables S12.A-J; "
    "HbA1c coefficients as corrected by the erratum of 2026-10-14, doi:10.1161/CIR.0000000000001491"
)
MODELS = ("base", "uacr", "hba1c", "sdi", "full")
OUTCOMES = ("total_cvd", "ascvd", "heart_failure", "chd", "stroke")
MG_PER_MMOL = 1 / 0.02586  # the source's cholesterol conversion factor (mg/dL -> mmol/L: x 0.02586)

# Allowable ranges, Supplemental Tables S12.A-J "Allowable range". Closed unless
# noted. Age 30-79 years is half-open [30, 80) so fractional ages are covered.
RANGES = {
    "age": (30, 80), "total_chol_mgdl": (130, 320), "hdl_chol_mgdl": (20, 100),
    "sbp": (90, 200), "bmi": (18.5, 40), "egfr": (15, 150),
    "uacr": (0.1, 25000), "hba1c": (3, 15), "sdi": (1, 10),
}
HALF_OPEN = {"age", "bmi"}  # "30-79 years", "18.5 to <40 kg/m2"


@lru_cache(maxsize=None)
def _table() -> dict[tuple[str, int], tuple[list[str], np.ndarray]]:
    """(model, years) -> (terms, coefs[term, outcome, sex]) with sex 0 = female, 1 = male."""
    out: dict = {}
    with (Path(__file__).parent / "coefficients.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            terms, rows = out.setdefault((r["model"], int(r["years"])), ([], []))
            terms.append(r["term"])
            rows.append([[float(r[f"{o}_{s}"] or 0) for s in ("female", "male")] for o in OUTCOMES])
    return {k: (t, np.array(v)) for k, (t, v) in out.items()}


def _in_range(x: np.ndarray, name: str) -> np.ndarray:
    lo, hi = RANGES[name]
    return (x >= lo) & ((x < hi) if name in HALF_OPEN else (x <= hi))


def prevent(
    age, sex, sbp, bp_treated, total_chol, hdl_chol, statin, diabetes, smoker, egfr, bmi,
    *, uacr=None, hba1c=None, sdi=None, model: str = "base", years: int = 10,
    outcome: str = "total_cvd",
) -> np.ndarray:
    """PREVENT risk. Source: prevent.SOURCE.

    model: "base", "uacr", "hba1c", "sdi" or "full" (all three optional predictors).
    years: 10 or 30. outcome: one of OUTCOMES.
    """
    if model not in MODELS or years not in (10, 30) or outcome not in OUTCOMES:
        raise ValueError(f"model must be one of {MODELS}, years 10 or 30, outcome one of {OUTCOMES}")
    none = lambda v: np.nan if v is None else v  # noqa: E731
    a, sb, tc, hdl, gfr, bm, ua, a1c, sd = (
        as_float(v) for v in (age, sbp, total_chol, hdl_chol, egfr, bmi, none(uacr), none(hba1c), none(sdi))
    )
    tx, st, dm, sm, sx = as_bool(bp_treated), as_bool(statin), as_bool(diabetes), as_bool(smoker), as_sex(sex)
    a, sb, tc, hdl, gfr, bm, ua, a1c, sd, tx, st, dm, sm, sx = np.broadcast_arrays(
        a, sb, tc, hdl, gfr, bm, ua, a1c, sd, tx, st, dm, sm, sx)

    valid = (
        _in_range(a, "age") & _in_range(sb, "sbp") & _in_range(bm, "bmi") & _in_range(gfr, "egfr")
        & _in_range(tc * MG_PER_MMOL, "total_chol_mgdl") & _in_range(hdl * MG_PER_MMOL, "hdl_chol_mgdl")
        & ~np.isnan(tx + st + dm + sm) & (sx >= 0)
    )
    # Optional predictors: missing -> indicator; present but out of range -> invalid row.
    uses = {"uacr": model in ("uacr", "full"), "hba1c": model in ("hba1c", "full"),
            "sdi": model in ("sdi", "full")}
    for name, x in (("uacr", ua), ("hba1c", a1c), ("sdi", sd)):
        if uses[name]:
            ok = _in_range(x, name) & ((x == np.round(x)) if name == "sdi" else True)
            valid &= np.isnan(x) | ok

    age_t = (a - 55) / 10
    non_hdl = tc - hdl - 3.5
    hdl_t = (hdl - 1.3) / 0.3
    sbp_ge = (np.maximum(sb, 110) - 130) / 20
    bmi_ge = (np.maximum(bm, 30) - 30) / 5
    egfr_lt = (np.minimum(gfr, 60) - 60) / -15
    ua_miss, a1c_miss, sdi_miss = np.isnan(ua), np.isnan(a1c), np.isnan(sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_uacr = np.where(ua_miss, 0.0, np.log(ua))
    a1c_c = np.where(a1c_miss, 0.0, a1c - 5.3)
    x = {
        "age": age_t, "age_squared": age_t**2, "non_hdl_c": non_hdl, "hdl_c": hdl_t,
        "sbp_lt_110": (np.minimum(sb, 110) - 110) / 20, "sbp_gte_110": sbp_ge,
        "dm": dm, "smoking": sm,
        "bmi_lt_30": (np.minimum(bm, 30) - 25) / 5, "bmi_gte_30": bmi_ge,
        "egfr_lt_60": egfr_lt, "egfr_gte_60": (np.maximum(gfr, 60) - 90) / -15,
        "bp_tx": tx, "statin": st,
        "bp_tx_sbp_gte_110": tx * sbp_ge, "statin_non_hdl_c": st * non_hdl,
        "age_non_hdl_c": age_t * non_hdl, "age_hdl_c": age_t * hdl_t,
        "age_sbp_gte_110": age_t * sbp_ge, "age_dm": age_t * dm, "age_smoking": age_t * sm,
        "age_bmi_gte_30": age_t * bmi_ge, "age_egfr_lt_60": age_t * egfr_lt,
        "ln_uacr": ln_uacr, "missing_uacr": ua_miss * 1.0,
        "hba1c_dm": a1c_c * dm, "hba1c_no_dm": a1c_c * (1 - dm), "missing_hba1c": a1c_miss * 1.0,
        "sdi_4_to_6": ((sd >= 4) & (sd <= 6)) * 1.0, "sdi_7_to_10": ((sd >= 7) & (sd <= 10)) * 1.0,
        "missing_sdi": sdi_miss * 1.0, "constant": np.ones_like(a),
    }
    terms, coefs = _table()[(model, years)]
    beta = coefs[:, OUTCOMES.index(outcome), :][:, np.where(sx == 1, 1, 0)]  # [term, row]
    with np.errstate(invalid="ignore"):
        log_odds = sum(beta[i] * x[t] for i, t in enumerate(terms))
    risk = np.exp(log_odds) / (1 + np.exp(log_odds))  # as written in the source
    return np.where(valid, risk, np.nan)
