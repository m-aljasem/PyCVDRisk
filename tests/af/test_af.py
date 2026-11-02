import numpy as np

from cvd_risk import cha2ds2_vasc, chads2, has_bled


def test_chads2_worked_example_and_range():
    # Gage 2001: an 82-year-old with hypertension and a prior stroke scores 4
    assert chads2(82, 0, 1, 0, 1) == 4
    np.testing.assert_array_equal(chads2([74.9, 75], 0, 0, 0, 0), [0, 1])
    assert chads2(80, 1, 1, 1, 1) == 6


def test_cha2ds2_vasc_age_and_sex():
    np.testing.assert_array_equal(cha2ds2_vasc([64.9, 65, 74.9, 75], "male", 0, 0, 0, 0, 0), [0, 1, 1, 2])
    assert cha2ds2_vasc(50, "female", 0, 0, 0, 0, 0) == 1
    assert cha2ds2_vasc(80, "f", 1, 1, 1, 1, 1) == 9


def test_has_bled_elderly_is_strictly_over_65():
    np.testing.assert_array_equal(has_bled([65, 65.1], 0, 0, 0, 0, 0, 0, 0, 0), [0, 1])
    assert has_bled(70, 1, 1, 1, 1, 1, 1, 1, 1) == 9


def test_invalid_rows_are_nan():
    assert np.isnan(chads2([-1, 70], 0, [0, 2], 0, 0)).all()
    assert np.isnan(cha2ds2_vasc(70, ["x", "male"], 0, 0, 0, 0, [0, 0.5])).all()
    assert np.isnan(has_bled([np.nan, 70], 0, 0, 0, 0, 0, 0, [0, 3], 0)).all()


def test_broadcast_2d():
    assert chads2(np.array([[60], [80]]), [0, 1], 1, 0, 0).shape == (2, 2)
