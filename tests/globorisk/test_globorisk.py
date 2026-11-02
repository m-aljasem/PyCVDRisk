"""Verification of Globorisk against the authors' R package (30 000 patients x 3 versions)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cvd_risk import globorisk

FX = Path(__file__).parents[2] / "reference" / "globorisk" / "fixtures"


@pytest.fixture(scope="module")
def data():
    p = pd.read_csv(FX / "patients.csv", keep_default_na=False, na_values=[""])
    r = pd.read_csv(FX / "reference.csv", keep_default_na=False, na_values=[""])
    assert (p.id == r.id).all()
    return p, r


def run(p, version):
    out = np.full(len(p), np.nan)
    for h, g in p.groupby("horizon"):
        out[g.index] = globorisk(
            g.age, g.sex, g.sbp, g.smoker, g.country, g.year, total_chol=g.total_chol,
            diabetes=g.diabetes, bmi=g.bmi, version=version, horizon=int(h))
    return out


@pytest.mark.parametrize("version", ["lab", "office", "fatal"])
def test_matches_authors_r_package(data, version):
    p, r = data
    got, ref = run(p, version), r[version].to_numpy()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)


@settings(max_examples=300)
@given(st.floats(40, 69.99), st.sampled_from(["male", "female"]), st.floats(90, 200),
       st.integers(0, 1), st.sampled_from(["USA", "GBR", "SDN", "CHN", "BRA", "IND"]),
       st.integers(2000, 2020), st.floats(3, 9), st.integers(0, 1))
def test_lab_in_unit_interval_and_monotone(age, sex, sbp, smk, iso, year, tc, dm):
    r = globorisk(age, sex, sbp, smk, iso, year, total_chol=tc, diabetes=dm)
    assert 0 < r < 1
    assert globorisk(age, sex, sbp + 10, smk, iso, year, total_chol=tc, diabetes=dm) > r
    assert globorisk(age, sex, sbp, 1, iso, year, total_chol=tc, diabetes=dm) > \
        globorisk(age, sex, sbp, 0, iso, year, total_chol=tc, diabetes=dm)
    shorter = globorisk(age, sex, sbp, smk, iso, year, total_chol=tc, diabetes=dm, horizon=5)
    assert shorter < r


def test_invalid_rows_nan_and_bad_arguments_raise():
    got = globorisk([55, 39, 81, 55, 55, 55], ["male", "male", "male", "x", "male", "male"],
                    140, 1, ["USA", "USA", "USA", "USA", "ZZZ", "usa"],
                    [2015, 2015, 2015, 2015, 2015, 2021], total_chol=5, diabetes=0)
    assert not np.isnan(got[0]) and np.isnan(got[1:]).all()
    with pytest.raises(ValueError):
        globorisk(55, "male", 140, 1, "USA", 2015, version="office")  # needs bmi
    with pytest.raises(ValueError):
        globorisk(55, "male", 140, 1, "USA", 2015, total_chol=5, diabetes=0, horizon=11)


def test_country_code_case_insensitive():
    a = globorisk(55, "male", 140, 1, ["usa", "nam"], 2015, total_chol=5, diabetes=0)
    assert a[0] == globorisk(55, "male", 140, 1, "USA", 2015, total_chol=5, diabetes=0)
    assert not np.isnan(a[1])


# ---- findings (see reference/globorisk/REPORT.md) --------------------------

def test_finding_smoking_lowers_lab_risk_in_oldest():
    # Smoking log-HR 1.8467 - 0.02205 * attained age is negative after ~83.7 years.
    rest = dict(total_chol=3.0, diabetes=0)
    assert globorisk(79, "male", 90, 1, "USA", 2000, **rest) < globorisk(79, "male", 90, 0, "USA", 2000, **rest)
    assert globorisk(60, "male", 90, 1, "USA", 2000, **rest) > globorisk(60, "male", 90, 0, "USA", 2000, **rest)


def test_finding_reference_returns_values_over_80_despite_documenting_na():
    # The R package warns ages > 80 "will be returned as NAs" but returns a value
    # when the horizon is short enough for its rate table. We follow the
    # documented rule: NaN at baseline age >= 81.
    assert np.isnan(globorisk(81, "male", 140, 1, "USA", 2015, total_chol=5, diabetes=0, horizon=5))


def test_south_sudan_has_no_data():
    # Present in the rate table but without population means in the source.
    assert np.isnan(globorisk(55, "male", 140, 1, "SSD", 2015, total_chol=5, diabetes=0))
