"""Verification of DIAL2 (Østergaard et al., Eur J Prev Cardiol 2023), core model.

1. Coefficients equal Supplementary Tables 3-5 (Word and PDF versions).
2. The paper's worked examples (Figure 3), example B with eGFR 90 (REPORT.md section 2).
3. Input handling.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from cvd_risk import dial2

REF = Path(__file__).parents[2] / "reference" / "dial2"


def _load(name):
    sys.path.insert(0, str(REF))
    spec = importlib.util.spec_from_file_location(name, REF / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_coefficients_match_supplement():
    assert _load("check_coefficients").main() == 0


def test_figure3_examples():
    R = _load("report")
    assert R.figure3(R.EXAMPLES["A"][0])[:3] == (70.5, 74.1, 3.6)
    assert R.figure3({**R.EXAMPLES["B"][0], "egfr": 90})[:3] == (76.0, 79.1, 3.1)
    assert R.single_changes() == {"egfr": [90]}


BASE = {"age": 50, "sex": "male", "smoker": 0, "sbp": 140, "total_chol": 5.5, "hdl_chol": 1.3, "hba1c": 75, "egfr": 70,
            "age_at_diagnosis": 40, "region": "moderate"}


@pytest.mark.parametrize("kw", [{"age": 29}, {"age": 95}, {"age": 50.5}, {"sex": "x"}, {"region": "high"},
                                {"smoker": 2}, {"egfr": 0}, {"hba1c": np.nan}, {"age_at_diagnosis": 51},
                                {"hr_cvd": -1}])
def test_invalid_rows_are_nan(kw):
    assert np.isnan(dial2(**{**BASE, **kw}))


def test_outcomes():
    r10, life = dial2(**BASE, outcome="risk10"), dial2(**BASE)
    assert 0 < r10 < life < 1
    assert np.isnan(dial2(**{**BASE, "age": 86}, outcome="risk10"))
    assert np.isfinite(dial2(**{**BASE, "age": 94}))
    le = dial2(**{**BASE, "age": [30, 94], "age_at_diagnosis": 30}, outcome="life_expectancy")
    assert 50 < le[0] < 95 and le[1] > 94  # inf when survival stays >= 0.5 to 95


def test_broadcasting():
    got = dial2(**{**BASE, "age": np.arange(40, 80).reshape(8, 5)})
    assert got.shape == (8, 5) and np.all(np.diff(got.ravel()) < 0)


def test_bad_outcome_raises():
    with pytest.raises(ValueError):
        dial2(**BASE, outcome="risk5")
