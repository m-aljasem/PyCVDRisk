"""Verification of SCORE (Conroy 2003), total-cholesterol model.

1. Every printed coefficient equals Appendix A of the paper (check_coefficients.py).
2. 798/800 cells of the official charts (Figs 1-2) lie within the risk interval the printed
   coefficients allow; the other two are documented chart anomalies.
3. Input handling.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from cvd_risk import score

REF = Path(__file__).parents[2] / "reference" / "score"


def _load(name):
    sys.path.insert(0, str(REF))
    spec = importlib.util.spec_from_file_location(name, REF / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_coefficients_match_the_publication():
    assert _load("check_coefficients").main() == 0


def test_official_charts_within_printed_precision():
    cc = _load("chart_check")
    reg, sex, sm, age, sbp, chol, _ = cc.cells()
    bad = {(r, s, m, a, b, c) for r, s, m, a, b, c, ok in zip(reg, sex, sm, age, sbp, chol, cc.consistent()) if not ok}
    assert len(reg) == 800
    assert bad == cc.KNOWN_ANOMALIES


BASE = dict(age=55, sex="male", total_chol=6, sbp=140, smoker=1, region="high")


def test_monotone_and_region_order():
    hi, lo = score(**BASE), score(**{**BASE, "region": "low"})
    assert 0 < lo < hi < 1
    assert score(**{**BASE, "smoker": 0}) < hi
    assert score(**{**BASE, "sex": "female"}) < hi


@pytest.mark.parametrize("kw", [dict(age=39.9), dict(age=66), dict(region="mid"), dict(sex="x"),
                                dict(smoker=2), dict(total_chol=np.nan), dict(sbp=-1)])
def test_invalid_rows_are_nan(kw):
    assert np.isnan(score(**{**BASE, **kw}))


def test_vectorised():
    r = score([45, 60], ["male", "female"], [5, 7], [120, 160], [0, 1], ["low", "high"])
    assert r.shape == (2,) and np.isfinite(r).all()
