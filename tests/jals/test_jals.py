import csv
import math
from pathlib import Path

import numpy as np
import pytest

from cvd_risk.jals import jals, jals_risk
from cvd_risk.jals.jals import POINTS

REF = Path(__file__).parents[2] / "reference"

# Tables 2-5 incidence rate ratios: item -> (model without AF, model with AF), categories as in POINTS
IRR = {
    "stroke": dict(af=(None, 4.18), bp=((1, 1.64, 1.64, 2.65, 4.23, 6.32), (1, 1.65, 1.64, 2.69, 4.25, 6.50)),
                   bp_treated=((2.33, 2.61, 2.52, 3.44, 3.80, 3.80), (2.22, 2.57, 2.52, 3.41, 3.84, 3.73)),
                   age=((1, 1.32, 2.45, 5.10, 8.68), (1, 1.31, 2.41, 4.93, 8.33)), diabetes=(1.51, 1.49),
                   men=(1.38, 1.31), smoker=(1.60, 1.64)),
    "ami": dict(af=(None, 2.41), hdl=((2.12, 1.64, 1), (2.11, 1.64, 1)),
                bp=((1, 1.23, 1.47, 1.65, 1.82, 3.18), (1, 1.23, 1.47, 1.66, 1.83, 3.21)),
                bp_treated=((1.18, 2.64, 3.15, 2.14, 1.56, 1.88), (1.15, 2.61, 3.17, 2.14, 1.56, 1.86)),
                age=((1, 2.19, 2.55, 4.38, 7.14), (1, 2.18, 2.53, 4.31, 6.99)),
                non_hdl=((1, 1.94, 2.41, 3.22), (1, 1.96, 2.43, 3.24)), diabetes=(1.87, 1.85), men=(3.21, 3.16),
                smoker=(2.04, 2.05)),
    "composite": dict(af=(None, 3.98), hdl=((1.30, 1.13, 1), (1.29, 1.13, 1)),
                      bp=((1, 1.55, 1.67, 2.54, 3.82, 5.95), (1, 1.57, 1.68, 2.58, 3.84, 6.12)),
                      bp_treated=((2.11, 2.68, 2.65, 3.20, 3.35, 3.46), (2.02, 2.64, 2.67, 3.18, 3.38, 3.41)),
                      age=((1, 1.46, 2.52, 5.04, 8.35), (1, 1.45, 2.47, 4.88, 8.03)), diabetes=(1.56, 1.54),
                      men=(1.51, 1.44), smoker=(1.66, 1.70)),
    "cvd_death": dict(af=(None, 3.53), bmi=((1.55, 1, 0.91), (1.57, 1, 0.90)), hdl=((1.24, 0.93, 1), (1.22, 0.94, 1)),
                      bp=((1, 1.02, 1.29, 1.32, 2.06, 2.39), (1, 1.03, 1.30, 1.35, 2.11, 2.51)),
                      bp_treated=((1.64, 1.78, 1.25, 1.53, 1.76, 1.43), (1.56, 1.79, 1.28, 1.55, 1.81, 1.41)),
                      age=((1, 1.52, 3.06, 11.32, 47.97), (1, 1.51, 3.02, 11.01, 46.20)),
                      egfr=((2.15, 1.18, 1, 1.10), (2.10, 1.17, 1, 1.11)), diabetes=(1.30, 1.28), men=(1.57, 1.51),
                      smoker=(1.86, 1.89)),
}


def _pts(irr, d=0.0):
    return None if irr is None else math.floor(10 * math.log2(irr + d) + 0.5)


@pytest.mark.parametrize("outcome", list(IRR))
def test_points_are_10_log2_irr(outcome):
    """Table footnote: score = 10 x log2(IRR), rounded. 174 of the 176 non-reference points follow from the printed IRR;
    the other 2 (stroke AF model, treated optimal BP: 2.22 -> 11.51, printed 11; composite AF model, normal BP:
    1.57 -> 6.51, printed 6) from an IRR within its 2-decimal rounding."""
    n = edge = 0
    for k, (no_af, af) in IRR[outcome].items():
        for m, v in enumerate((no_af, af)):
            for i, x in enumerate(v if isinstance(v, tuple) else (v,)):
                if x is None or x == 1:
                    continue
                want = POINTS[outcome][k][m][i] if isinstance(v, tuple) else POINTS[outcome][k][m]
                n += 1
                if _pts(x) != want:
                    assert want in (_pts(x, -0.005), _pts(x, 0.005)), (outcome, k, m, i)
                    edge += 1
    assert (n, edge) == {"stroke": (37, 1), "ami": (47, 0), "composite": (41, 1), "cvd_death": (51, 0)}[outcome]


def test_table6():
    """Table 6 (AF models), 4 outcomes x 13 scores x 5 age groups: 257/260 printed 5-year and 235/260 10-year values
    equal, every value within 0.01 point."""
    rows = list(csv.DictReader(open(REF / "jals/fixtures/table6.csv")))
    name = dict(stroke="stroke", ami="ami", composite="composite", cvd="cvd_death")
    age = {"40-49": 45, "50-59": 55, "60-69": 65, "70-79": 75, "80-89": 85}
    hits = {5: 0, 10: 0}
    for r in rows:
        for y in (5, 10):
            ours = round(100 * float(jals_risk(int(r["score"]), age[r["age"]], name[r["outcome"]], y)), 2)
            assert abs(ours - float(r[f"risk{y}"])) < 0.0101
            hits[y] += ours == float(r[f"risk{y}"])
    assert hits == {5: 257, 10: 235}


def test_worked_example():
    """Table 6 footnote: man 75, grade II hypertension untreated, no diabetes, non-smoker: stroke score 25 (total 48),
    5-year 3.08%, 10-year 8.07%."""
    s = jals(75, "male", 165, 85, 0, 0, 0, outcome="stroke", af=0)
    assert s == 25
    assert round(100 * float(jals_risk(s, 75, "stroke")), 2) == 3.08
    assert round(100 * float(jals_risk(s, 75, "stroke", 10)), 2) == 8.07


def test_blood_pressure_takes_the_higher_category():
    f = lambda sbp, dbp, t=0: float(jals(50, "female", sbp, dbp, t, 0, 0, outcome="stroke", af=0))  # noqa: E731
    assert [f(119, 79), f(120, 79), f(119, 80), f(119, 85), f(130, 70), f(110, 92), f(150, 105), f(119, 110)] == \
        [0, 7, 7, 7, 7, 14, 21, 27]
    assert [f(119, 79, 1), f(185, 70, 1)] == [11, 19]


def test_inputs():
    kw = dict(hdl=50, non_hdl=150, bmi=22, egfr=70)
    assert jals(60, "male", 130, 80, 0, 1, 1, outcome="cvd_death", **kw) == 4 + 9 + 4 + 7 - 1
    assert jals(60, "male", 130, 80, 0, 1, 1, outcome="cvd_death", af=1, **kw) == 4 + 9 + 4 + 6 - 1 + 18
    assert np.isnan(jals([39, 90], "male", 130, 80, 0, 0, 0, outcome="stroke")).all()
    assert np.isnan(jals_risk(10, [39, 90], "stroke")).all()
    with pytest.raises(ValueError):
        jals(60, "male", 130, 80, 0, 0, 0, outcome="ami", hdl=50)
    with pytest.raises(ValueError):
        jals_risk(10, 60, "stroke", years=7)
