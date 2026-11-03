"""uniscore routes each row to its country's guideline score and returns exactly that score."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cvd_risk import (guideline_score, prevent, qrisk3, score2, score2_asia_pacific, score2_op,
                      uniscore, who2019)

ROOT = Path(__file__).parents[2]
P = dict(sex="female", sbp=135, total_chol=5.8, hdl_chol=1.4, smoker=0, diabetes=0, bp_treated=1,
         statin=0, egfr=85, bmi=29, ethnicity="indian", smoking="ex", sbp_sd=6)
S2 = dict(sex="female", smoker=0, sbp=135, total_chol=5.8, hdl_chol=1.4, diabetes=0)


@pytest.mark.skipif(not (ROOT / "reference/score2/source/score2_asia_pacific_ehae609_supplement.docx").exists(), reason="publisher document not in the repository (see reference/README.md)")
def test_routing_table_is_current():
    spec = importlib.util.spec_from_file_location("b", ROOT / "reference/uniscore/build_routing.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    rows = b.rows()
    assert sum(v[0] == "score2" for v in rows.values()) == 54  # 55 in Fig 5, minus GBR -> QRISK3
    lines = (ROOT / "src/cvd_risk/uniscore/routing.csv").read_text().splitlines()
    assert len(lines) - 1 == len(rows)


@pytest.mark.parametrize("country,expected", [
    ("USA", lambda: prevent(60, "female", 135, 1, 5.8, 1.4, 0, 0, 0, 85, 29)),
    ("GBR", lambda: qrisk3(60, "female", "indian", "ex", 29, 5.8 / 1.4, 135, 6, bp_treated=1)),
    ("FRA", lambda: score2(60, **S2, region="low")),
    ("LTU", lambda: score2(60, **S2, region="very_high")),
    ("JPN", lambda: score2_asia_pacific(60, **S2, region="low")),
    ("IND", lambda: score2_asia_pacific(60, **S2, region="high")),
    ("BRA", lambda: who2019(60, "female", 135, 0, total_chol=5.8, diabetes=0, country="BRA",
                            recalibration="country")),
])
def test_each_route_equals_the_direct_call(country, expected):
    assert uniscore(country, age=60, **P) == expected()


def test_score2_op_from_age_70():
    assert uniscore("DEU", age=75, **P) == score2_op(75, **S2, region="moderate")


def test_vectorised_mixed_countries():
    c = np.array(["USA", "DEU", "BRA", "XXX", "jpn"])
    r = uniscore(c, age=[50, 60, 70, 50, 55], **P)
    singles = [uniscore(x, age=a, **P) for x, a in zip(c, [50, 60, 70, 50, 55])]
    np.testing.assert_array_equal(r, singles)
    assert np.isnan(r[3]) and not np.isnan(r[[0, 1, 2, 4]]).any()
    assert list(guideline_score(c)) == ["prevent", "score2", "who2019", "", "score2_asia_pacific"]


def test_qrisk3_diabetes_type_bridge():
    a = uniscore("GBR", age=60, **{**P, "diabetes": 1}, diabetes_type="type2")
    assert a == qrisk3(60, "female", "indian", "ex", 29, 5.8 / 1.4, 135, 6, bp_treated=1, diabetes="type2")
    with pytest.raises(ValueError, match="diabetes_type"):
        uniscore("GBR", age=60, **{**P, "diabetes": 1})


def test_missing_and_unknown_inputs_raise():
    with pytest.raises(ValueError, match="egfr"):
        uniscore("USA", age=60, **{k: v for k, v in P.items() if k != "egfr"})
    with pytest.raises(TypeError, match="cholesterol"):
        uniscore("USA", age=60, cholesterol=5, **P)
