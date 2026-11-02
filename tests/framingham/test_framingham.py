"""Verification of the Framingham 2008 general CVD risk profile (lab and BMI models).

1. Every printed coefficient equals the paper / supplement (check_coefficients.py).
2. The official Framingham Heart Study calculator spreadsheets: every cached result.
3. The paper's two worked examples (Score Sheet, cases 1 and 2).
4. Agreement with R CVrisk (rounding, floor and cap removed) on 20 000 synthetic patients.
5. Input handling.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk import framingham
from cvd_risk.prevent.prevent import MG_PER_MMOL as K

REF = Path(__file__).parents[2] / "reference" / "framingham"


def _kw(model, d):
    if model == "lab":
        return dict(total_chol=d.total_chol_mgdl / K, hdl_chol=d.hdl_chol_mgdl / K)
    return dict(bmi=d.bmi)


def test_coefficients_match_the_publication():
    spec = importlib.util.spec_from_file_location("chk", REF / "check_coefficients.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.main() == 0


@pytest.mark.parametrize("model", ["lab", "bmi"])
def test_reproduces_official_spreadsheet(model):
    e = pd.read_csv(REF / "fixtures" / "spreadsheet_examples.csv")
    e = e[e.model == model]
    r = framingham(e.age, e.sex, e.sbp, e.bp_treated, e.smoker, e.diabetes, **_kw(model, e), model=model)
    np.testing.assert_allclose(r, e.risk, rtol=1e-12)


def test_paper_worked_examples():
    r = framingham([61, 53], ["female", "male"], [124, 125], [0, 1], [1, 0], [0, 1],
                   total_chol=np.array([180, 161]) / K, hdl_chol=np.array([47, 55]) / K)
    np.testing.assert_array_equal(np.round(100 * r, 1), [10.5, 15.6])


@pytest.mark.parametrize("model", ["lab", "bmi"])
def test_matches_cvrisk(model):
    p = pd.read_csv(REF / "fixtures" / "patients.csv")
    r = pd.read_csv(REF / "fixtures" / "reference.csv")[model]
    g = framingham(p.age, p.sex, p.sbp, p.bp_treated, p.smoker, p.diabetes, **_kw(model, p), model=model)
    na = r.isna()
    assert ((p.age[na] > 74) & (p.age[na] < 75)).all()  # CVrisk rejects 74.01-74.99
    np.testing.assert_allclose(g[~na], r[~na], rtol=1e-12, atol=1e-15)


BASE = dict(age=50, sex="male", sbp=130, bp_treated=0, smoker=0, diabetes=0, total_chol=5, hdl_chol=1.2)


@pytest.mark.parametrize("kw", [{"age": 29.9}, {"age": 75}, {"sex": "x"}, {"sbp": 0}, {"hdl_chol": np.nan},
                                {"smoker": 3}])
def test_invalid_rows_are_nan(kw):
    assert np.isnan(framingham(**{**BASE, **kw}))


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        framingham(50, "male", 130, 0, 0, 0, model="bmi")
    with pytest.raises(ValueError):
        framingham(50, "male", 130, 0, 0, 0, bmi=25, model="x")
