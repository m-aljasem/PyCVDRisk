import csv
from pathlib import Path

import numpy as np

from cvd_risk import timi

FX = Path(__file__).parents[2] / "reference" / "timi" / "fixtures"


def test_matches_riskscorescvd():
    p = list(csv.DictReader((FX / "patients.csv").open()))
    ref = [float(r["score"]) for r in csv.DictReader((FX / "reference.csv").open())]
    col = lambda k: np.array([float(x[k]) for x in p])
    np.testing.assert_array_equal(timi(col("age"), col("risk_factors"), col("coronary_stenosis"),
                                       col("st_deviation"), col("episodes"), col("aspirin"),
                                       col("elevated_markers")), ref)


def test_cut_points():
    np.testing.assert_array_equal(timi([64, 65], [2, 3], 0, 0, [1, 2], 0, 0), [0, 3])
    assert timi(70, 5, 1, 1, 3, 1, 1) == 7


def test_invalid_rows_are_nan():
    assert np.isnan(timi([-1, 50, 50, 50], [0, 1.5, 0, 0], [0, 0, 2, 0], 0, [0, 0, 0, -1], 0, 0)).all()
