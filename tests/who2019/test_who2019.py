"""Verification of the 2019 WHO CVD risk models.

1. Every cell of the official WHO risk charts (21 regions, lab + non-lab; 44 100 cells).
2. Agreement with the R reference (WHORiskCalculator) on 30 000 synthetic patients,
   lab/non-lab x region/country recalibration.
3. Input handling.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk import who2019

FX = Path(__file__).parents[2] / "reference" / "who2019" / "fixtures"
# Chart cells are computed at band midpoints (found by search; each value is the only
# one in its neighbourhood that reproduces every printed cell).
AGE = {"40-44": 42.5, "45-49": 47.5, "50-54": 52.5, "55-59": 57.5, "60-64": 62.5, "65-69": 67.5, "70-74": 72.5}
SBP = {"<120": 110, "120-139": 130, "140-159": 150, "160-179": 170, ">=180": 190}
X = {"<4": 3.5, "4-4.9": 4.5, "5-5.9": 5.5, "6-6.9": 6.5, ">=7": 7.5,
     "<20": 17.5, "20-24": 22.5, "25-29": 27.5, "30-35": 32.5, ">=35": 37.5}


@pytest.mark.parametrize("version", ["lab", "nonlab"])
def test_reproduces_every_official_chart_cell(version):
    c = pd.read_csv(FX / "charts.csv")
    c = c[c.version == version]
    x = c.x_band.map(X)
    kw = dict(total_chol=x, diabetes=c.diabetes) if version == "lab" else dict(bmi=x)
    r = who2019(c.age_band.map(AGE), c.sex, c.sbp_band.map(SBP), c.smoker, **kw,
                region=c.region, version=version)
    assert len(c) == (29_400 if version == "lab" else 14_700)
    np.testing.assert_array_equal(np.round(100 * r).astype(int), c.risk_pct)


@pytest.fixture(scope="module")
def ref():
    p = pd.read_csv(FX / "patients.csv", keep_default_na=False, na_values=[""])  # "NAM" = Namibia
    return p, pd.read_csv(FX / "reference.csv")


@pytest.mark.parametrize("version", ["lab", "nonlab"])
@pytest.mark.parametrize("recal", ["region", "country"])
def test_matches_r_reference(ref, version, recal):
    p, r = ref
    kw = dict(total_chol=p.total_chol, diabetes=p.diabetes) if version == "lab" else dict(bmi=p.bmi)
    got = who2019(p.age, p.sex, p.sbp, p.smoker, **kw, country=p.country, version=version,
                  recalibration=recal, year=p.year)
    assert not np.isnan(got).any()
    np.testing.assert_allclose(got, r[f"{version}_{recal}"], rtol=1e-12, atol=1e-15)


LAB = dict(age=55, sex="male", sbp=150, smoker=1, total_chol=6, diabetes=0)


def test_country_maps_to_region():
    assert who2019(**LAB, country="nga") == who2019(**LAB, region="western_subsaharan_africa")


def test_country_without_own_parameters_falls_back_to_region():
    a = who2019(**LAB, country="SSD", recalibration="country")
    assert a == who2019(**LAB, region="eastern_subsaharan_africa") and not np.isnan(a)


def test_country_recalibration_differs_from_region():
    assert who2019(**LAB, country="GBR", recalibration="country") != who2019(**LAB, country="GBR")


@pytest.mark.parametrize("kw", [
    {"country": "XXX"}, {"region": "atlantis"}, {"country": "GBR", "year": 1989},
    {"country": "GBR", "year": 2017.5}, {"country": "GBR", "age": 39.9}, {"country": "GBR", "age": 81},
    {"country": "GBR", "sex": "x"}, {"country": "GBR", "total_chol": np.nan}, {"country": "GBR", "smoker": 3},
])
def test_invalid_rows_are_nan(kw):
    assert np.isnan(who2019(**{**LAB, **kw}))


def test_year_periods():
    # 1990-2015 use the 5-year period they fall in; 2016 and 2017 their own year.
    f = lambda y: float(who2019(**LAB, country="IND", year=y))  # noqa: E731
    assert f(1997) == f(1995) != f(2000) and f(2016) != f(2015) and f(2017) != f(2016)


def test_bad_arguments_raise():
    for kw in ({}, {"country": "GBR", "version": "x"}, {"region": "east_asia", "recalibration": "country"}):
        with pytest.raises(ValueError):
            who2019(**LAB, **kw)
    with pytest.raises(ValueError):
        who2019(55, "male", 150, 1, region="east_asia", version="nonlab")
