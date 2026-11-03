"""Verification of LIFE-CVD2 (Hageman et al., Eur J Prev Cardiol 2024).

1. Coefficients equal Supplementary Tables 3-5.
2. Agreement with the authors' 16 published charts (Appendix 1): lifetime risk to 80 and
   CVD-free life years gained from SBP lowering, LDL lowering and smoking cessation
   (REPORT.md section 2).
3. Input handling.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from cvd_risk import life_cvd2

REF = Path(__file__).parents[2] / "reference" / "life_cvd2"


def _load(name):
    sys.path.insert(0, str(REF))
    spec = importlib.util.spec_from_file_location(name, REF / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(not (REF / "source" / "hageman2024_supplement.pdf").exists(), reason="publisher document not in the repository (see reference/README.md)")
def test_coefficients_match_supplement():
    assert _load("check_coefficients").main() == 0


@pytest.mark.skipif(not (REF / "source" / "hageman2024_appendix1_charts.pdf").exists(), reason="publisher document not in the repository (see reference/README.md)")
def test_published_charts():
    R = _load("report")
    outliers, worst = [], [0.0]
    res = R.compare(defect_age=62, outliers=outliers, worst=worst)
    assert {k: v.tolist() for k, v in res.items()} == {
        "lifetime": [2048, 2043, 2048, 0], "sbp": [1280, 1279, 1280, 1280],
        "ldl": [2560, 2557, 2560, 0], "smoking": [1280, 1277, 1279, 1280]}
    assert len(outliers) == 1 and worst[0] < 0.52  # one tie-break cell, 0.512 of 0.1 years


BASE = {"age": 50, "sex": "male", "smoker": 1, "sbp": 150, "total_chol": 6, "hdl_chol": 1.2, "region": "low"}


def test_chart_cells():
    # Low-risk lifetime chart: man, smoking, 40-44, SBP 160-179, non-HDL 6.0-6.9 prints 40%;
    # woman, non-smoking, 75-79, SBP 160-179, non-HDL 3.0-3.9 prints 4%.
    got = life_cvd2([42, 77], ["male", "female"], [1, 0], 170, [6.5 + 1.2, 3.5 + 1.4], [1.2, 1.4], "low")
    np.testing.assert_array_equal(np.round(100 * got), [40, 4])


@pytest.mark.parametrize("kw", [{"age": 34}, {"age": 80}, {"age": 50.5}, {"sex": "x"}, {"region": "mars"},
                                {"smoker": 2}, {"sbp": np.nan}, {"hdl_chol": 0}, {"hr_cvd": 0}])
def test_invalid_rows_are_nan(kw):
    assert np.isnan(life_cvd2(**{**BASE, **kw}))


def test_life_expectancy():
    le = life_cvd2(**{**BASE, "age": [35, 99]}, outcome="life_expectancy")
    assert 70 < le[0] < 90 and 99 < le[1] < 101
    assert np.isnan(life_cvd2(**{**BASE, "age": 100}, outcome="life_expectancy"))
    treated = life_cvd2(**BASE, outcome="life_expectancy", hr_cvd=0.6, hr_non_cvd=0.73)
    assert treated > life_cvd2(**BASE, outcome="life_expectancy")


def test_broadcasting_and_defaults():
    got = life_cvd2(**{**BASE, "age": np.arange(40, 80).reshape(8, 5)})
    assert got.shape == (8, 5) and np.all(np.diff(got.ravel()) < 0)  # less time left to 80
    assert life_cvd2(**BASE) == life_cvd2(**BASE, hr_cvd=1, hr_non_cvd=1)


def test_bad_outcome_raises():
    with pytest.raises(ValueError):
        life_cvd2(**BASE, outcome="risk10")
