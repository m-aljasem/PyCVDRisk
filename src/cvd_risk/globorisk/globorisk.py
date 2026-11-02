"""Globorisk: country-specific 10-year risk of CVD (laboratory, office, fatal).

Accepts scalars or array-likes (broadcast like numpy), returns a float64 array
of risks as fractions in [0, 1]. A row is NaN when an input it needs is missing
or invalid, the country has no Globorisk data, or age/year/horizon is out of
range (see coefficients.py). Values are never clamped.

Units: SBP mmHg; total cholesterol mmol/L; BMI kg/m^2.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex
from cvd_risk.globorisk import coefficients as C

VERSIONS = ("lab", "office", "fatal")


@cache
def _data():
    with np.load(Path(__file__).with_name("data.npz")) as d:
        return tuple(str(i) for i in d["isos"]), d["rates"], d["means"]


def globorisk(
    age, sex, sbp, smoker, country, year,
    total_chol=None, diabetes=None, bmi=None, version="lab", horizon=10,
) -> np.ndarray:
    """Globorisk risk over `horizon` years (1-10) from baseline `year` (2000-2020).

    version "lab" and "fatal" need total_chol and diabetes; "office" needs bmi.
    `country` is an ISO 3166-1 alpha-3 code (e.g. "USA", "SDN").
    """
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}")
    if not (isinstance(horizon, (int, np.integer)) and 1 <= horizon <= 10):
        raise ValueError("horizon must be a whole number of years from 1 to 10")
    office = version == "office"
    if (bmi is None) if office else (total_chol is None or diabetes is None):
        needs = "bmi" if office else "total_chol and diabetes"
        raise ValueError(f"version '{version}' needs {needs}")

    isos, rates, means = _data()
    age, sbp, yr = as_float(age), as_float(sbp), as_float(year)
    smk, female = as_bool(smoker), as_sex(sex)
    iso = as_category(np.char.upper(np.asarray(country, dtype=str)), tuple(i.lower() for i in isos))
    x3 = as_float(bmi) / 5 if office else as_float(total_chol)
    dm = np.float64(0) if office else as_bool(diabetes)
    age, sbp, yr, smk, female, iso, x3, dm = np.broadcast_arrays(age, sbp, yr, smk, female, iso, x3, dm)
    female = np.where(female < 0, -1, 1 - female)  # model codes sex as 1 = female

    a = np.trunc(age)
    ok = (
        (a >= C.AGE_RANGE[0]) & (a < C.AGE_RANGE[1]) & (yr >= C.YEAR_RANGE[0])
        & (yr <= C.YEAR_RANGE[1]) & (yr == np.trunc(yr)) & (female >= 0) & (iso >= 0)
    )
    # Index arrays only for valid rows; others point at slot 0 and are masked at the end.
    ai = np.where(ok, a, 40).astype(int)
    yi = np.where(ok, yr, 2000).astype(int)
    si = np.where(ok, female, 0)
    ci = np.where(ok, iso, 0)
    m = means[ci, si, np.minimum(ai // 5 - 8, 8)]  # age group 40-44 -> 0 ... 80-84 -> 8

    sbp_c, smk_c = sbp / 10 - m[..., 0], smk - m[..., 3]
    if office:
        c, x3_c = C.OFFICE, x3 - m[..., 4]
        main = c["main_sbp"] * sbp_c + c["main_bmi"] * x3_c + c["main_smk"] * smk_c \
            + c["main_female_smk"] * si * smk_c
        tvc = c["tvc_sbp"] * sbp_c + c["tvc_smk"] * smk_c + c["tvc_bmi"] * x3_c
    else:
        c = C.FATAL if version == "fatal" else C.LAB
        tc_c, dm_c = x3 - m[..., 1], dm - m[..., 2]
        main = c["main_sbp"] * sbp_c + c["main_tc"] * tc_c + c["main_dm"] * dm_c \
            + c["main_smk"] * smk_c + c["main_female_dm"] * si * dm_c \
            + c["main_female_smk"] * si * smk_c
        tvc = c["tvc_sbp"] * sbp_c + c["tvc_tc"] * tc_c + c["tvc_dm"] * dm_c + c["tvc_smk"] * smk_c

    outcome = 1 if version == "fatal" else 0
    cum_hazard = np.zeros(np.shape(age))
    for t in range(horizon):
        base = rates[outcome, ci, si, yi + t - 2000, np.minimum(ai + t - 40, rates.shape[-1] - 1)]
        base = np.where(ai + t - 40 < rates.shape[-1], base, np.nan)
        cum_hazard = cum_hazard + np.exp(main + tvc * (ai + t)) * base
    return np.where(ok, -np.expm1(-cum_hazard), np.nan)
