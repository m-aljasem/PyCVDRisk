import csv
from pathlib import Path

import numpy as np

from cvd_risk import edacs

FX = Path(__file__).parents[2] / "reference" / "edacs" / "fixtures"


def test_matches_riskscorescvd():
    p = list(csv.DictReader((FX / "patients.csv").open()))
    ref = [float(r["score"]) for r in csv.DictReader((FX / "reference.csv").open())]
    c = lambda k: np.array([x[k] for x in p])
    f = lambda k: c(k).astype(float)
    np.testing.assert_array_equal(edacs(f("age"), c("sex"), f("known_cad"), f("risk_factors"), f("diaphoresis"),
                                        f("radiates"), f("pleuritic"), f("palpation")), ref)


def test_age_bands_and_young_rule():
    np.testing.assert_array_equal(edacs([18, 45, 46, 50, 51, 85, 86, 99], "f", 0, 0, 0, 0, 0, 0),
                                  [2, 2, 4, 4, 6, 18, 20, 20])
    np.testing.assert_array_equal(edacs([50, 51], "m", 1, 0, 0, 0, 0, 0), [14, 12])


def test_invalid_rows_are_nan():
    assert np.isnan(edacs([17, 40, 40], ["m", "x", "m"], [0, 0, 2], 0, 0, 0, 0, 0)).all()
