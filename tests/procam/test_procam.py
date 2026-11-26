import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cvd_risk import procam, procam_stroke
from cvd_risk.prevent.prevent import MG_PER_MMOL as K
from cvd_risk.procam import CATEGORIES, STROKE_RISK


REF = Path(__file__).parents[2] / "reference" / "procam"


def _report():
    spec = importlib.util.spec_from_file_location("procam_report", REF / "report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rep():
    return _report()


@pytest.mark.skipif(not (Path(__file__).parents[2] / "reference/procam/source/assmann2007.pdf").exists(), reason="publisher document not in the repository (see reference/*/source/SOURCES.md)")
def test_table2_bands_against_pdf(rep):
    rows = rep.table2_rows(rep.flat(rep.pages()[3]))
    assert all(found and lo == p == hi for _, _, p, found, lo, hi in rows) and len(rows) == 50


@pytest.mark.skipif(not (Path(__file__).parents[2] / "reference/procam/source/assmann2007.pdf").exists(), reason="publisher document not in the repository (see reference/*/source/SOURCES.md)")
def test_tables3_4_against_pdf(rep):
    parsed, slips = rep.risk_rows(rep.pages()[4])
    for sex, rows in parsed.items():
        for age, (bounds, _) in rows.items():
            assert bounds == rep.CORONARY_TABLES[sex].get(age, ()), (sex, age)
    assert slips == [("female", "46 < 44 45–53 54–62 63–69 = 68")]  # printed overlap, see REPORT.md


@pytest.mark.skipif(not (Path(__file__).parents[2] / "reference/procam/source/assmann2007.pdf").exists(), reason="publisher document not in the repository (see reference/*/source/SOURCES.md)")
def test_table7_against_pdf(rep):
    assert rep.table7_rows(rep.flat(rep.pages()[5])) == STROKE_RISK


def test_points_and_categories():
    zero = dict(ldl_chol=90 / K, hdl_chol=60 / K, triglycerides=90 / 88.57, sbp=100, smoker=0, diabetes=0, family_history=0)
    worst = dict(ldl_chol=200 / K, hdl_chol=30 / K, triglycerides=250 / 88.57, sbp=190, smoker=1, diabetes=1, family_history=1)
    assert procam(50, "m", **zero) == 0
    np.testing.assert_array_equal(procam(50, ["m", "f"], **worst), [69, 71])
    # boundary points read as "<=": men 35 has 0-4% up to 43 points, 5-9% from 44
    assert CATEGORIES[int(procam(35.9, "m", **{**worst, "smoker": 0, "diabetes": 0, "sbp": 140}, output="category"))] == "5-9%"
    pts = procam(35, "m", **{**worst, "smoker": 0, "diabetes": 0, "sbp": 140})
    assert pts == 20 + 11 + 4 + 4 + 5
    assert procam(20, "f", **worst, output="category") == 0  # 20-33 row: <= 71 is 0-4%
    assert procam(75, "m", **worst, output="category") == 4


def test_stroke():
    assert procam_stroke(50, "m", 1, 1, 150) == 50 + 6 + 9 + 7 + 3
    assert procam_stroke(35, "f", 0, 0, 139.9) == 35 and procam_stroke(35, "f", 0, 0, 180) == 44
    assert STROKE_RISK[59] == 1.0 and STROKE_RISK[80] == 10.0


def test_invalid():
    assert np.isnan(procam([19, 76, 50, 50], ["m", "m", "x", "m"], 3, 1.2, 1.5, [120, 120, 120, -1], 0, 0, 0)).all()
    assert np.isnan(procam_stroke([34, 66], "m", 0, 0, 120)).all()
    with pytest.raises(ValueError):
        procam(50, "m", 3, 1.2, 1.5, 120, 0, 0, 0, output="risk")


def test_assmann_foundation_online_tests():
    # coronary: Tables 3-4 category = the tool's Weibull risk category 167/193 with inputs in closed bands, none 2 apart;
    # stroke: the tool departs from Tables 6-7 (male sex 4 points, higher table). Asserted inside tool_section().
    pytest.importorskip("pypdf")
    lines = _report().tool_section()
    assert any("| no input in an open-ended band | 193 | 167 | 14 | 12 | 0 |" in ln for ln in lines)
