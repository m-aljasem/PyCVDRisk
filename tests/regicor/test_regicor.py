import math

import numpy as np
import pytest

from cvd_risk import regicor
from cvd_risk.prevent.prevent import MG_PER_MMOL as K
from cvd_risk.regicor import mean_lp

MAN = {"age": 50, "sex": "male", "total_chol": 210 / K, "hdl_chol": 52 / K, "sbp": 135, "dbp": 80, "diabetes": 1, "smoker": 0}


def test_appendix_man_recomputed():
    # Appendix terms re-added (the printed sums 3.84 / 4.32 do not add up; see REPORT.md)
    assert round(mean_lp("male", "framingham"), 4) == 3.6661 and round(mean_lp("male", "gerona"), 4) == 4.1208
    assert math.isclose(regicor(**MAN, calibration="framingham"), 1 - 0.943 ** math.exp(4.02 - 3.66607))
    assert math.isclose(regicor(**MAN), 1 - 0.965 ** math.exp(4.02 - 4.12079))


def test_bp_takes_higher_category_and_edges():
    r = regicor(**{**MAN, "sbp": [119, 119, 160], "dbp": [79, 100, 60]})
    lp = np.log(np.log(1 - r) / np.log(1 - r[0]))
    np.testing.assert_allclose(lp, [0, 0.90 - 0.09, 0.90 - 0.09], atol=1e-12)
    r = regicor(**{**MAN, "total_chol": [199.99 / K, 200 / K]})
    assert r[1] > r[0]


def test_invalid():
    got = regicor([34, 75, 50, 50, 50], ["m", "m", "x", "f", "m"], 5, 1.3, [130, 130, 130, 130, -1], 80, [0, 0, 0, 2, 0], 0)
    assert np.isnan(got).all()
    with pytest.raises(ValueError):
        regicor(**MAN, calibration="spain")


def test_broadcast_2d():
    assert regicor(np.array([[40], [60]]), ["m", "f"], 5, 1.3, 130, 80, 0, 1).shape == (2, 2)
