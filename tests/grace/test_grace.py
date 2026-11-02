import numpy as np

from cvd_risk import grace_6month, grace_hospital


def test_hospital_worked_examples_follow_the_table():
    # Granger 2003 Fig 4 prints 196 and 103; its terms use SBP 100 -> 53 and creatinine 0.4 -> 1,
    # against the printed table (43 and 4). See reference/grace/REPORT.md.
    assert grace_hospital(2, 100, 100, 65, 1, 0, 1, 1) == 185
    assert grace_hospital(1, 80, 60, 55, 0.4, 0, 0, 0) == 106


def test_hospital_edges_and_maximum():
    np.testing.assert_array_equal(grace_hospital(1, [80, 80.1, 200], 50, 30, 4.0, 0, 0, 0),
                                  [58 + 28, 53 + 28, 0 + 28])
    assert grace_hospital(4, 250, 250, 95, 5, 1, 1, 1) == 314


def test_six_month_items_and_edges():
    base = grace_6month(20, 0, 0, 40, 250, 0, 0.2, 0, 1)
    assert base == 1
    assert grace_6month(20, 1, 1, 40, 250, 1, 0.2, 1, 0) == base + 24 + 12 + 11 + 15 + 14
    np.testing.assert_array_equal(grace_6month([39.9, 40, 90], 0, 0, [49.9, 50, 200], [79.9, 80, 200], 0, 4, 0, 1),
                                  [0 + 0 + 24 + 20, 18 + 3 + 22 + 20, 100 + 43 + 0 + 20])


def test_invalid_rows_are_nan():
    assert np.isnan(grace_hospital([5, 1, 1], [120, -1, 120], 80, 60, [1, 1, np.nan], 0, 0, 0)).all()
    assert np.isnan(grace_6month([-1, 60], 0, 0, 80, 120, 0, 1, 0, [1, 2])).all()
