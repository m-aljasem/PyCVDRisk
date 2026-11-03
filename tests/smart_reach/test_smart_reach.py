"""Verification of SMART-REACH (Kaasenbrood et al., JAHA 2018).

1. Coefficients equal the authors' calculator, and round to the supplement's printed values.
2. Agreement with the authors' calculator, recalculated by LibreOffice, on 600 synthetic patients:
   10-year risk everywhere; life expectancy wherever the life table is well formed, and the
   calculator's own median rule reproduces its output on every row (REPORT.md section 3).
3. Input handling.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk import smart_reach

REF = Path(__file__).parents[2] / "reference" / "smart_reach"


def _load(name):
    sys.path.insert(0, str(REF))
    spec = importlib.util.spec_from_file_location(name, REF / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load("report")
P = pd.read_csv(REF / "fixtures" / "patients.csv")
REF10, REF_LE = R.reference()


@pytest.mark.skipif(not (REF / "source" / "kaasenbrood2018_supplement.pdf").exists(), reason="publisher document not in the repository (see reference/README.md)")
def test_coefficients_match_calculator_and_supplement():
    assert _load("check_coefficients").main() == 0


def test_risk10_matches_calculator():
    got = smart_reach(**R.inputs(P))
    np.testing.assert_array_equal(np.isnan(got), np.isnan(REF10))
    np.testing.assert_allclose(got, REF10, rtol=1e-12, atol=0)


def test_life_expectancy_matches_calculator():
    got = smart_reach(**R.inputs(P), outcome="life_expectancy")
    calc_rule, negative = R.calculator_life_expectancy(P)
    np.testing.assert_allclose(calc_rule, REF_LE, rtol=1e-12, atol=0)  # the calculator, emulated exactly
    np.testing.assert_allclose(got[~negative], REF_LE[~negative], rtol=1e-12, atol=0)
    assert (got[negative] != REF_LE[negative]).all()  # only where the calculator's rule misfires
    assert (got >= P.age).all()


BASE = dict(age=60, sex="male", smoker=1, diabetes=0, sbp=140, total_chol=5, creatinine=90,
            cad=1, cerebrovascular=0, pad=0)


def test_calculator_example():
    # Patient 0 of the fixtures: the calculator prints 0.383325159456817 and 57.4687622901329.
    row = dict(age=45, sex="female", smoker=0, diabetes=1, sbp=125, total_chol=8, creatinine=132,
               cad=1, cerebrovascular=1, pad=1)
    np.testing.assert_allclose(smart_reach(**row), 0.383325159456817, rtol=1e-13)
    np.testing.assert_allclose(smart_reach(**row, outcome="life_expectancy"), 57.4687622901329, rtol=1e-13)


@pytest.mark.parametrize("kw", [{"age": 44}, {"age": 81}, {"age": 60.5}, {"sex": "x"}, {"cad": 0},
                                {"region": "mars"}, {"sbp": np.nan}, {"creatinine": 0}, {"af": 2}])
def test_invalid_rows_are_nan(kw):
    assert np.isnan(smart_reach(**{**BASE, **kw}))


def test_life_expectancy_age_range_and_cap():
    assert np.isfinite(smart_reach(**{**BASE, "age": 85}, outcome="life_expectancy"))
    assert np.isnan(smart_reach(**{**BASE, "age": 90}, outcome="life_expectancy"))
    young = dict(BASE, age=45, smoker=0, sbp=120, cad=1, region="netherlands")
    assert smart_reach(**young, outcome="life_expectancy") > 60


def test_bad_outcome_raises():
    with pytest.raises(ValueError):
        smart_reach(**BASE, outcome="lifetime")
