import numpy as np
import pytest

from cvd_risk import who_ish
from cvd_risk.who_ish import BANDS, REGIONS


def test_bands_and_regions():
    assert len(BANDS) == 5 and len(REGIONS) == 14


# Cells read from the guideline's Annex 3 (SEAR-D) / Annex 4 (EMR-B); see reference/who_ish/REPORT.md.
def test_guideline_corrections_over_whoishrisk():
    # women 70, no diabetes, SBP 120: printed 10-<20% at chol 5 and 6, >=40% for a smoker at chol 8
    got = who_ish(70, "female", [0, 0, 1], 120, 0, "sear_d", total_chol=[5, 6, 8])
    np.testing.assert_array_equal(got, [1, 1, 4])


def test_every_printed_cell_round_trips():
    import csv
    import sys
    from pathlib import Path

    path = Path(sys.modules["cvd_risk.who_ish.who_ish"].__file__).with_name("charts.csv")
    rows = list(csv.DictReader(path.open()))
    r = {k: np.array([float(x[k]) for x in rows]) for k in ("diabetes", "male", "smoker", "age", "sbp", "chol")}
    sex = np.where(r["male"] == 1, "male", "female")
    lab = r["chol"] > 0
    for reg in REGIONS:
        want = np.array([float(x[reg]) for x in rows])
        got = np.empty(len(rows))
        got[lab] = who_ish(r["age"][lab], sex[lab], r["smoker"][lab], r["sbp"][lab], r["diabetes"][lab],
                           reg, total_chol=r["chol"][lab])
        got[~lab] = who_ish(r["age"][~lab], sex[~lab], r["smoker"][~lab], r["sbp"][~lab],
                            r["diabetes"][~lab], reg)
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("age,sbp,chol,cell", [
    (49.99, 139.99, 4.49, (40, 120, 4)), (50, 140, 4.5, (50, 140, 5)), (79.9, 180, 7.5, (70, 180, 8)),
    (65, 250, 12, (60, 180, 8)), (40, 90, 2.0, (40, 120, 4)), (59, 179.9, 6.5, (50, 160, 7)),
])
def test_cell_selection(age, sbp, chol, cell):
    a, b, c = cell
    np.testing.assert_array_equal(
        who_ish(age, "male", 1, sbp, 1, "eur_c", total_chol=chol),
        who_ish(a, "male", 1, b, 1, "eur_c", total_chol=c))


def test_invalid_rows_are_nan():
    got = who_ish([39.9, 80, 50, 50, 50, 50, 50, 50], ["m", "m", "x", "m", "m", "m", "m", "m"],
                  [1, 1, 1, 2, 1, 1, 1, 1], [150, 150, 150, 150, 0, 150, 150, 150],
                  [0, 0, 0, 0, 0, 0.5, 0, 0], ["eur_a"] * 6 + ["eur_z", "eur_a"],
                  total_chol=[5, 5, 5, 5, 5, 5, 5, 0])
    assert np.isnan(got).all()


def test_region_spelling_and_nonlab_chart():
    got = who_ish(62, "male", 1, 185, 1, ["EMR-B", "emr b", "emr_b"])
    assert np.all(got == got[0]) and got[0] == 4


def test_broadcast_2d():
    got = who_ish(np.array([[45], [65]]), "female", [0, 1], 150, 0, "wpr_b", total_chol=6)
    assert got.shape == (2, 2)
