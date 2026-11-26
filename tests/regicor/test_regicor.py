import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from cvd_risk import regicor
from cvd_risk.prevent.prevent import MG_PER_MMOL as K
from cvd_risk.regicor import mean_lp


REF = Path(__file__).parents[2] / "reference" / "regicor"


def _report():
    spec = importlib.util.spec_from_file_location("regicor_report", REF / "report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not (Path(__file__).parents[2] / "reference/regicor/source/marrugat2003_rev_esp_cardiol.pdf").exists() or not (Path(__file__).parents[2] / "reference/regicor/source/wilson1998.pdf").exists(), reason="publisher document not in the repository (see reference/*/source/SOURCES.md)")
def test_tables_against_pdfs():
    rep = _report()
    assert rep.wilson_table6() in rep.text("wilson1998.pdf")
    rows = rep.rev_esp_rows(rep.text("marrugat2003_rev_esp_cardiol.pdf", dash=True))
    assert [n for n, f in rows if not f] == ["bp Óptima (PAS < 120)/(PAD < 80)"]  # printed +0.0023, Wilson -0.00226


def test_bands():
    _, ok, n = _report().bands()
    assert ok == n == 40


def test_worked_examples():
    # Wilson 1998 Appendix (Framingham) and Rev Esp Cardiol 2003 Table 3 (Girona)
    assert round(float(regicor(55, "m", 250 / K, 39 / K, 146, 88, 0, 1, calibration="framingham")), 4) == 0.3336
    sum_bx = 0.04826 * 56 + 0.50539 + 0.24310 + 0.52168 + 0.52337
    assert round(sum_bx, 4) == 4.4961
    assert math.isclose(regicor(56, "m", 256 / K, 37 / K, 146, 92, 0, 1),
                        1 - 0.951 ** math.exp(sum_bx - mean_lp("male", "gerona")))
    assert round(mean_lp("male", "gerona"), 4) == 3.4881 and round(mean_lp("female", "gerona"), 4) == 10.2973


def test_official_charts():
    import sys

    sys.path.insert(0, str(REF))
    import chart_check

    ok, _, miss, _ = chart_check.stated(chart_check.package, "male")
    assert ok >= 198 and miss < 0.01
    assert chart_check.stated(chart_check.package, "female")[2] < 1


def test_bp_takes_higher_category_and_edges():
    man = dict(age=50, sex="male", total_chol=210 / K, hdl_chol=52 / K, diabetes=1, smoker=0)
    r = regicor(**man, sbp=[119, 119, 160], dbp=[79, 100, 60])
    lp = np.log(np.log(1 - r) / np.log(1 - r[0]))
    np.testing.assert_allclose(lp, [0, 0.61859 + 0.00226, 0.61859 + 0.00226], atol=1e-12)
    r = regicor(**{**man, "total_chol": [199.99 / K, 200 / K]}, sbp=135, dbp=80)
    assert r[1] > r[0]


def test_invalid():
    got = regicor([34, 75, 50, 50, 50], ["m", "m", "x", "f", "m"], 5, 1.3, [130, 130, 130, 130, -1], 80, [0, 0, 0, 2, 0], 0)
    assert np.isnan(got).all()
    with pytest.raises(ValueError):
        regicor(50, "m", 5, 1.3, 130, 80, 0, 0, calibration="spain")


def test_broadcast_2d():
    assert regicor(np.array([[40], [60]]), ["m", "f"], 5, 1.3, 130, 80, 0, 1).shape == (2, 2)
