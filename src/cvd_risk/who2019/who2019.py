"""2019 WHO CVD risk models: 10-year risk of fatal and non-fatal MI and stroke.

Accepts scalars or array-likes (broadcast like numpy) and returns a float64 array
of risks as fractions in [0, 1]. A row is NaN when an input is missing or invalid,
sex/region/country is unrecognised, or age/year is outside the valid range.

version="lab":    total_chol (mmol/L) and diabetes required.
version="nonlab": bmi (kg/m²) required.
Location: `region` (one of coefficients.REGIONS) or `country` (ISO3; mapped to its
region). recalibration="country" uses country-level parameters where the authors
provide them and falls back to the region otherwise (5 territories, e.g. SSD).
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from cvd_risk._inputs import as_bool, as_category, as_float, as_sex
from cvd_risk.who2019 import coefficients as C

HERE = Path(__file__).parent
YEARS = (1990, 1995, 2000, 2005, 2010, 2015, 2016, 2017)
PARAMS = ("chd_cons", "chd_slope", "stroke_cons", "stroke_slope")


@lru_cache(maxsize=None)
def _data():
    """countries, country->region index, params[level][code, sex, year, version*4 + param]."""
    with (HERE / "countries.csv").open(newline="") as f:
        mapping = list(csv.DictReader(f))
    countries = tuple(r["iso3"] for r in mapping)
    region_of = np.array([C.REGIONS.index(r["region"]) for r in mapping] + [-1])
    cols = [f"{v}_{p}" for v in ("lab", "nonlab") for p in PARAMS]
    tables = {"region": np.full((len(C.REGIONS) + 1, 2, len(YEARS), 8), np.nan),
              "country": np.full((len(countries) + 1, 2, len(YEARS), 8), np.nan)}
    codes = {"region": C.REGIONS, "country": countries}
    with (HERE / "recalibration.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            i = codes[r["level"]].index(r["code"])
            tables[r["level"]][i, int(r["sex"] == "male"), YEARS.index(int(r["year"]))] = [
                float(r[c]) for c in cols]
    return countries, region_of, tables


def _risk(coefs, sex, age_c, sbp_c, x_c, x_name, sm, dm=None):
    k = lambda key: np.where(sex == 1, coefs["male"][key], coefs["female"][key])  # noqa: E731
    lp = (k("age") * age_c + k(x_name) * x_c + k("sbp") * sbp_c + k("smoker") * sm
          + k(f"{x_name}_age") * x_c * age_c + k("sbp_age") * sbp_c * age_c + k("smoker_age") * sm * age_c)
    if dm is not None:
        lp = lp + k("diabetes") * dm + k("diabetes_age") * dm * age_c
    return -np.expm1(np.exp(lp) * np.log(k("s0")))  # 1 - s0 ** exp(lp)


def _recalibrate(p, cons, slope):
    return -np.expm1(-np.exp(cons + slope * np.log(-np.log1p(-p))))


def who2019(
    age, sex, sbp, smoker, *, total_chol=None, diabetes=None, bmi=None,
    region=None, country=None, version: str = "lab", recalibration: str = "region", year=2017,
) -> np.ndarray:
    """2019 WHO CVD risk. Source: coefficients.SOURCE."""
    if version not in ("lab", "nonlab") or recalibration not in ("region", "country"):
        raise ValueError('version must be "lab" or "nonlab"; recalibration "region" or "country"')
    if region is None and country is None:
        raise ValueError("give region or country")
    if recalibration == "country" and country is None:
        raise ValueError('recalibration="country" needs country')
    if version == "lab" and (total_chol is None or diabetes is None):
        raise ValueError('version="lab" needs total_chol and diabetes')
    if version == "nonlab" and bmi is None:
        raise ValueError('version="nonlab" needs bmi')

    countries, region_of, tables = _data()
    ci = as_category(np.nan if country is None else country, tuple(c.lower() for c in countries))
    ri = region_of[ci] if region is None else as_category(region, C.REGIONS)
    x = as_float(total_chol if version == "lab" else bmi)
    dm = as_bool(diabetes) if version == "lab" else np.float64(0)
    a, sx, sb, sm, x, dm, ci, ri, yr = np.broadcast_arrays(
        as_float(age), as_sex(sex), as_float(sbp), as_bool(smoker), x, dm, ci, ri, as_float(year))

    lo, hi = C.AGE_RANGE
    a = np.where((a >= lo) & (a < hi), a, np.nan)
    y = np.floor(yr)
    y = np.where(y <= 2015, 5 * np.floor(y / 5), y)
    valid_year = (yr >= C.YEAR_RANGE[0]) & (yr <= C.YEAR_RANGE[1])
    yi = np.searchsorted(YEARS, np.where(valid_year, y, YEARS[0]))
    s = np.where(sx == 1, 1, 0)

    v = 0 if version == "lab" else 4
    cal = tables["region"][ri, s, yi]  # ri == -1 -> NaN row
    if recalibration == "country":
        cc = tables["country"][ci, s, yi]
        cal = np.where(np.isnan(cc[..., :1]), cal, cc)
    cal = cal[..., v:v + 4]

    coefs = C.LAB if version == "lab" else C.NONLAB
    name = "chol" if version == "lab" else "bmi"
    x_c = x - (6 if version == "lab" else 25)
    args = (sx, a - 60, sb - 120, x_c, name, sm) + ((dm,) if version == "lab" else ())
    with np.errstate(invalid="ignore"):
        p_mi = _recalibrate(_risk(coefs["mi"], *args), cal[..., 0], cal[..., 1])
        p_st = _recalibrate(_risk(coefs["stroke"], *args), cal[..., 2], cal[..., 3])
        risk = 1 - (1 - p_mi) * (1 - p_st)
    return np.where(valid_year & (sx >= 0), risk, np.nan)
