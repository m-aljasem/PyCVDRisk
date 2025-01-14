import numpy as np

from cvd_risk import reynolds
from cvd_risk.prevent.prevent import MG_PER_MMOL as K


def test_women_table6_and_men_example():
    # Ridker 2007 Table 6 (printed 4.9 ... 18.4) and Ridker 2008 Clinical Examples (printed 20.4)
    w = reynolds(50, "female", 155, 240 / K, 35 / K, [0.1, 20.0], 1, [0, 1])
    np.testing.assert_allclose(100 * w, [4.932, 18.404], atol=1e-3)
    np.testing.assert_allclose(100 * reynolds(65, "male", 130, 205 / K, 45 / K, 4.0, 0, 1), 20.490, atol=1e-3)


def test_diabetes():
    base = reynolds(60, "female", 130, 5, 1.3, 1, 0, 0)
    assert reynolds(60, "female", 130, 5, 1.3, 1, 0, 0, diabetes=1, hba1c=7) > base
    assert np.isnan(reynolds(60, "female", 130, 5, 1.3, 1, 0, 0, diabetes=1))  # HbA1c needed
    assert np.isnan(reynolds(60, "male", 130, 5, 1.3, 1, 0, 0, diabetes=1))  # men's model: non-diabetic


def test_invalid_rows_are_nan():
    got = reynolds([60, 60, 60, 60], ["x", "m", "m", "m"], 130, [5, 0, 5, 5], 1.3, [1, 1, 0, 1], [0, 0, 0, 2], 0)
    assert np.isnan(got).all()
