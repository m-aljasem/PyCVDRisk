"""Verification of the SCORE2 family.

1. Agreement with the R reference (RiskScorescvd, unrounded) on 60 000 synthetic patients.
2. Properties every valid implementation must satisfy.
3. Input handling and range edges.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cvd_risk import score2, score2_diabetes, score2_op
from cvd_risk.score2 import REGIONS

FIXTURES = Path(__file__).parents[2] / "reference" / "score2" / "fixtures"
BASE_COLS = ["age", "sex", "smoker", "sbp", "total_chol", "hdl_chol", "diabetes"]
DM_COLS = BASE_COLS + ["age_at_diabetes_diagnosis", "hba1c", "egfr"]


def load(variant):
    p = pd.read_csv(FIXTURES / f"{variant}_patients.csv")
    r = pd.read_csv(FIXTURES / f"{variant}_reference.csv")
    assert (p.id == r.id).all()
    return p, r.risk.to_numpy()


@pytest.mark.parametrize(
    "variant, fn, cols",
    [("score2", score2, BASE_COLS), ("score2_op", score2_op, BASE_COLS),
     ("score2_diabetes", score2_diabetes, DM_COLS)],
)
def test_matches_r_reference(variant, fn, cols):
    p, expected = load(variant)
    got = fn(**{c: p[c] for c in cols}, region=p.region)
    assert not np.isnan(got).any()
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-15)


# ---- properties -----------------------------------------------------------

sexes = st.sampled_from(["male", "female"])
regions = st.sampled_from(REGIONS)
binary = st.integers(0, 1)
sbp_s = st.floats(90, 200)
tc_s = st.floats(3, 9)
hdl_s = st.floats(0.7, 2.5)


@settings(max_examples=300)
@given(st.floats(40, 69.9), sexes, binary, sbp_s, tc_s, hdl_s, binary, regions)
def test_score2_in_unit_interval_and_monotone(age, sex, sm, sbp, tc, hdl, dm, rg):
    r = score2(age, sex, sm, sbp, tc, hdl, dm, rg)
    assert 0 < r < 1
    if sm == 0 and dm == 0:  # provably monotone here; see test_finding_score2_risk_can_fall_with_age
        assert score2(min(age + 1, 69.9), sex, sm, sbp, tc, hdl, dm, rg) >= r
    assert score2(age, sex, sm, sbp + 10, tc, hdl, dm, rg) > r
    assert score2(age, sex, 1, sbp, tc, hdl, dm, rg) >= score2(age, sex, 0, sbp, tc, hdl, dm, rg)
    assert score2(age, sex, sm, sbp, tc, hdl, 1, rg) >= score2(age, sex, sm, sbp, tc, hdl, 0, rg)


@settings(max_examples=300)
@given(st.floats(70, 89.9), sexes, binary, sbp_s, tc_s, hdl_s, binary, regions)
def test_score2_op_in_unit_interval_and_monotone(age, sex, sm, sbp, tc, hdl, dm, rg):
    r = score2_op(age, sex, sm, sbp, tc, hdl, dm, rg)
    assert 0 < r < 1
    assert score2_op(age, sex, sm, sbp + 10, tc, hdl, dm, rg) > r
    if age < 87:  # see test_finding_score2_op_smoking_lowers_risk_in_oldest_men
        assert score2_op(age, sex, 1, sbp, tc, hdl, dm, rg) >= score2_op(age, sex, 0, sbp, tc, hdl, dm, rg)


@settings(max_examples=300)
@given(st.floats(40, 69.9), sexes, binary, sbp_s, tc_s, hdl_s, binary)
def test_very_high_region_never_below_high(age, sex, sm, sbp, tc, hdl, dm):
    hi = score2(age, sex, sm, sbp, tc, hdl, dm, "high")
    assert score2(age, sex, sm, sbp, tc, hdl, dm, "very_high") >= hi


# ---- findings: behaviour of the published models (see reference/score2/REPORT.md)

def test_finding_moderate_region_can_exceed_high_region():
    # Recalibration slopes differ by region, so at low risk the "moderate"
    # region gives a higher risk than "high". Same in the R reference.
    args = (40, "male", 0, 90, 3.0, 1.0, 0)
    assert score2(*args, "moderate") > score2(*args, "high")


def test_finding_score2_op_risk_can_fall_with_age():
    # Age interaction terms can outweigh the age main effect.
    args = ("male", 0, 191, 3.0, 1.0, 1, "low")
    assert score2_op(71, *args) < score2_op(70, *args)


def test_batch_equals_row_by_row():
    p, _ = load("score2")
    p = p.head(200)
    batch = score2(**{c: p[c] for c in BASE_COLS}, region=p.region)
    rows = [score2(**{c: row[c] for c in BASE_COLS}, region=row.region) for _, row in p.iterrows()]
    np.testing.assert_array_equal(batch, rows)


# ---- input handling and range edges ----------------------------------------

def test_age_boundary_exactly_one_variant_applies():
    args = ("male", 1, 140, 5.5, 1.3, 0, "low")
    for age in (40, 69, 69.99, 70, 89, 89.99, 90, 39.99):
        n_valid = sum(not np.isnan(f(age, *args)) for f in (score2, score2_op))
        assert n_valid == (1 if 40 <= age < 90 else 0), age


def test_invalid_rows_are_nan_others_unaffected():
    got = score2(
        age=[50, 50, 50, 50, 50, 30],
        sex=["Male", "other", "female", "male", "male", "male"],
        smoker=[1, 1, 2, 1, 1, 1],
        sbp=[140, 140, 140, np.nan, 140, 140],
        total_chol=5.5, hdl_chol=1.3, diabetes=0,
        region=["Very High", "low", "low", "low", "nowhere", "low"],
    )
    assert not np.isnan(got[0])
    assert np.isnan(got[1:]).all()


def test_scalar_broadcast_and_series_input():
    s = pd.Series([140.0, 160.0], index=[7, 3])
    got = score2(55, "female", 0, s, 6.0, 1.5, 0, "moderate")
    assert got.shape == (2,) and got[1] > got[0]


def test_diabetes_agediag_ignored_without_diabetes():
    a = score2_diabetes(55, "male", 0, 130, 5, 1.2, 0, np.nan, 40, 90, "low")
    b = score2_diabetes(55, "male", 0, 130, 5, 1.2, 0, 30, 40, 90, "low")
    assert a == b and not np.isnan(a)


def test_diabetes_nonpositive_egfr_is_nan():
    assert np.isnan(score2_diabetes(55, "male", 0, 130, 5, 1.2, 1, 50, 40, 0, "low"))


def test_finding_score2_op_smoking_lowers_risk_in_oldest_men():
    # Male smoking effect 0.3524 - 0.0247 * (age - 73) turns negative after ~87.3 years.
    args = (88, "male")
    rest = (90, 3.0, 1.0, 0, "low")
    assert score2_op(*args, 1, *rest) < score2_op(*args, 0, *rest)
    assert score2_op(80, "male", 1, *rest) > score2_op(80, "male", 0, *rest)


def test_finding_score2_risk_can_fall_with_age():
    # Male age effect per 5 y: 0.3742 - 0.0755*smoker - 0.0983*diabetes - 0.0255*sbp_t
    # - 0.0281*tc_t + 0.0426*hdl_t; negative for extreme diabetic smokers.
    rest = ("male", 1, 175, 9.0, 0.75, 1, "low")
    assert score2(41, *rest) < score2(40, *rest)
