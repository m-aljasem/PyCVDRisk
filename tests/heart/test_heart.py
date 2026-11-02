import csv
from pathlib import Path

import numpy as np

from cvd_risk import heart

FX = Path(__file__).parents[2] / "reference" / "heart" / "fixtures"


def test_matches_riskscorescvd():
    p = list(csv.DictReader((FX / "patients.csv").open()))
    ref = [float(r["score"]) for r in csv.DictReader((FX / "reference.csv").open())]
    col = lambda k: np.array([float(x[k]) for x in p])
    got = heart(col("history"), col("ecg"), col("age"), col("risk_factors"), col("atherosclerosis"),
                col("troponin_ratio"))
    np.testing.assert_array_equal(got, ref)


def test_extremes_and_cut_points():
    np.testing.assert_array_equal(heart(0, 0, 44.9, 0, 0, 0.999), 0)
    np.testing.assert_array_equal(heart(2, 2, 65, 3, 0, 3.001), 10)
    np.testing.assert_array_equal(heart(0, 0, [45, 64.9, 65], 0, 0, 0), [1, 1, 2])
    np.testing.assert_array_equal(heart(0, 0, 30, 0, 0, [1, 3, 3.01]), [1, 1, 2])
    np.testing.assert_array_equal(heart(0, 0, 30, [0, 2, 3, 0], [0, 0, 0, 1], 0), [0, 1, 2, 2])


def test_invalid_rows_are_nan():
    got = heart([3, 0, 0, 0, 0, 0, 1.5], 0, [30, -1, 30, 30, 30, 30, 30], [0, 0, 1.5, 0, 0, -1, 0],
                [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, -0.1, 0, 0])
    assert np.isnan(got).all()


def test_broadcast_2d():
    assert heart(np.array([[0], [2]]), [0, 1], 50, 1, 0, 2).shape == (2, 2)
