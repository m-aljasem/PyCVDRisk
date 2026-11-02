import csv
import sys
from pathlib import Path

import numpy as np
import pytest

from cvd_risk.predict import (
    predict_bleed,
    predict_cvd,
    predict_cvd_bmi,
    predict_cvd_policy,
    predict_post_acs,
    predict_post_cvd,
    predict_t2d,
)

REF = Path(__file__).parents[2] / "reference" / "predict"
sys.path.insert(0, str(REF))
import report

NAN = float("nan")


def test_paper_examples():
    assert round(float(predict_cvd(55, "f", "european", 3, "ex", 1, 0, 0, 135, 5, 1, 0, 0)), 4) == 0.0511
    got = predict_bleed(65, "m", "european", 3, "ex", 1, 0, 1, 1, *[0] * 9, 130, 5)
    assert round(float(got), 9) == 0.019038994


@pytest.mark.parametrize("name", list(report.MODELS))
def test_fixtures_valid_and_match_r_if_run(name):
    rows = report.load(name)
    p = report.python_risk(name, rows)
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    ref = REF / "fixtures" / f"{name}_reference.csv"
    if not ref.exists():
        pytest.skip("authors' R code not run yet (reference/predict/run_reference.R)")
    r = np.array([float(x["risk"]) for x in csv.DictReader(ref.open())])
    np.testing.assert_allclose(p, r, rtol=0, atol=1e-10)


def test_bands_are_gap_free():
    args = lambda bmi: (55, "m", "european", 3, "never", 0, 0, 0, 130, 4, bmi, 0, 0, 0)
    r = predict_cvd_bmi(*args(np.array([22.0, 24.99, 25.0, 27.3, 29.99, NAN])))
    assert r[0] == r[1] and r[2] == r[3] == r[4] and r[1] != r[2] and np.isfinite(r[5])
    post = lambda hb: predict_post_cvd(65, "m", "european", 3, 0, 0, 0, 0, 0, 1, 1, 1, 130, 4, 27, 80, hb, 400)
    h = post(np.array([39.9, 40, 64.5, 65, NAN]))
    assert h[1] == h[2] and h[2] != h[3] and h[0] != h[1] and np.isfinite(h[4])


def test_invalid_rows_are_nan():
    assert np.isnan(predict_cvd([29, 80, 50, 50, 50], ["m", "m", "x", "m", "m"], ["european"] * 3 + ["other", "european"],
                                3, "ex", 1, 0, 0, 135, 5, 1, 0, [0, 0, 0, 0, 2])).all()
    assert np.isnan(predict_cvd_policy(50, "m", "european", [0, 6], 0, 0, 0, 0, 0)).all()
    assert np.isnan(predict_t2d(55, "f", "maori", 4, 0, 0, 0, 1, 0, 1, 1, 0, 135, 4, 32, 85, -1, 64, 6))
    assert np.isnan(predict_post_acs(65, "m", "european", 3, 0, 1, 0, 0, 1, 1, 1, 135, 4, NAN, 110, 70, 100, "mi"))
    assert np.isfinite(predict_cvd_policy(50, "m", "other", 3, 0, 0, 0, 0, 0))


def test_post_cvd_other_vd_counts_as_old_event():
    f = lambda ov, d: float(predict_post_cvd(65, "m", "european", 3, 0, 0, 0, 0, ov, 1, 1, 1, 130, 4, 27, 80, 50, d))
    assert f(1, 10) == f(0, 3000) == f(0, NAN)


def test_broadcast_2d():
    assert predict_cvd(np.array([[40], [60]]), ["m", "f"], "european", 3, "ex", 0, 0, 0, 130, 4, 0, 0, 0).shape == (2, 2)
