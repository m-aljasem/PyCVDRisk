import math

import numpy as np
import pytest

from cvd_risk import finrisk


def test_figure1_bars():
    # Vartiainen 2016 Fig 1: age 50, inferred HDL 1.3 (men) / 1.64 (women); see REPORT.md
    assert round(100 * float(finrisk(50, "m", 0, 4.5, 1.3, 120, 0, 0, 0, outcome="chd")), 2) == 1.10
    assert round(100 * float(finrisk(50, "m", 1, 4.5, 1.3, 160, 1, 0, 0, outcome="chd")), 2) == 7.85
    assert round(100 * float(finrisk(50, "f", 1, 7, 1.64, 160, 1, 0, 0, outcome="chd")), 2) == 6.64


def test_formula_and_cvd_is_sum():
    lp = -11.213 + 0.0802 * 60 + 0.6260 + 0.3293 * 6 + 0.0166 * 140 - 0.5893 * 1.2 + 0.3138
    assert math.isclose(finrisk(60, "m", 1, 6, 1.2, 140, 0, 1, 0, outcome="chd"), 1 / (1 + math.exp(-lp)))
    args = (55, ["m", "f"], 1, 6, 1.2, 140, 1, 1, 1)
    np.testing.assert_allclose(finrisk(*args), finrisk(*args, outcome="chd") + finrisk(*args, outcome="stroke"))


def test_invalid():
    assert np.isnan(finrisk([29, 65, 50, 50], ["m", "m", "x", "m"], [0, 0, 0, 2], 5, 1.3, 130, 0, 0, 0)).all()
    assert np.isnan(finrisk(50, "f", 0, -1, 1.3, 130, 0, 0, 0, outcome="stroke"))
    with pytest.raises(ValueError):
        finrisk(50, "m", 0, 5, 1.3, 130, 0, 0, 0, outcome="mi")
