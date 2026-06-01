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


def test_every_combination_matches_paper_tables():
    # independent of af.py: Gage, Lip Table 2 and Pisters Table 2 point tables, at every age cut point
    import itertools
    for age in [64.9, 65, 65.1, 74.9, 75]:
        for f in itertools.product((0, 1), repeat=4):
            assert chads2(age, *f) == f[0] + f[1] + f[2] + 2 * f[3] + (age >= 75)
        for f in itertools.product((0, 1), repeat=6):
            want = f[0] + f[1] + f[2] + 2 * f[3] + f[4] + f[5] + (2 if age >= 75 else age >= 65)
            assert cha2ds2_vasc(age, "female" if f[5] else "male", *f[:5]) == want
        for f in itertools.product((0, 1), repeat=8):
            assert has_bled(age, *f) == sum(f) + (age > 65)
