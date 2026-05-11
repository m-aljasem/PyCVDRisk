import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk.hisayama import hisayama, hisayama_points, hisayama_points_risk
from cvd_risk.hisayama.hisayama import AGE, COEF, POINT_CONST, POINT_SLOPE

REF = Path(__file__).parents[2] / "reference"
WEB = json.loads((REF / "hisayama/fixtures/web_calculator_and_fig1.json").read_text())

# Supplemental Table 3: (points, weight in regression units C = beta x distance from the reference value)
TABLE_S3 = [(5, 0.771), (11, 1.543), (16, 2.314), (20, 2.854), (7, 0.984), (1, 0.155), (2, 0.259), (3, 0.414),
            (4, 0.621), (3, 0.459), (1, 0.178), (2, 0.356), (1, 0.144), (2, 0.240), (3, 0.384), (4, 0.632), (2, 0.336),
            (2, 0.339)]


def test_points_are_weights_over_0144():
    """Supplemental Table 3: point = C / 0.144, rounded."""
    assert [round(c / POINT_SLOPE) for _, c in TABLE_S3] == [p for p, _ in TABLE_S3]


def test_point_constant():
    """Supplemental Methods: 2.4767 = 6.7963 - sum(beta x reference value) = 6.7963 - 4.3196. Betas from the C column
    (age 2.854/37, SBP 0.621/60, HDL-C 0.356/30, LDL-C 0.384/80) at the reference values 45, 110, 65, 100."""
    ref = 45 * 2.854 / 37 + 110 * 0.621 / 60 - 65 * 0.356 / 30 + 100 * 0.384 / 80
    assert ref == pytest.approx(4.3196, abs=0.002)
    assert COEF["ldl"][1] - 4.3196 == pytest.approx(POINT_CONST, abs=1e-9)


@pytest.mark.parametrize("kind,mean", [("ldl", dict(lipid=134.0)), ("non_hdl", dict(lipid=156.8))])
def test_equation_constant_near_table1_means(kind, mean):
    """The constant is sum(beta x mean); with Table 1 means and the 3-decimal betas it comes within 0.05."""
    b, const = COEF[kind]
    m = dict(age=58.2, male=0.418, sbp=132.4, diabetes=0.118, hdl=50.6, proteinuria=0.057, smoker=0.25,
             no_exercise=1 - 0.101, **mean)
    assert sum(b[k] * m[k] for k in b) == pytest.approx(const, abs=0.05)


def test_fig1_points_and_age():
    assert hisayama_points("male", 165, 1, 35, 1, 1, 0, ldl=170) == 27
    assert hisayama_points("female", 119, 0, 60, 0, 0, 1, ldl=119) == 0
    assert hisayama_points("female", [120, 130, 140, 160], 0, 59, 0, 0, 1, ldl=139).tolist() == [3, 4, 5, 6]
    assert hisayama_points("female", 110, 0, 70, 0, 0, 1, non_hdl=[149, 150, 170, 190]).tolist() == [0, 1, 2, 3]
    assert list(AGE[1]) == [0, 5, 11, 16, 20]


def test_fig1_table_does_not_follow_the_formula():
    """Finding: Fig. 1's risk table is one sequence in score + age points (checked against its corner values), but it
    runs above the stated formula: 15.7% at score 27, ages 40-49, against 11.9%."""
    seq = WEB["fig1_sequence"]["values"]
    assert (seq[9], seq[11], seq[16], seq[20], seq[27], seq[36]) == (1.0, 1.4, 3.1, 5.6, 15.7, 49.7)
    assert round(100 * float(hisayama_points_risk(27, 45)), 1) == 11.9
    formula = [round(100 * float(hisayama_points_risk(k, 45)), 1) for k in range(28)]
    assert sum(f == s for f, s in zip(formula, seq)) < 10


def test_jas2022_web_calculator_is_the_formula_with_exercise_points():
    """ecg.watson.jp (JAS 2022 chart: no proteinuria or exercise items) = the formula at score + 2 (no regular
    exercise), shown to 1 decimal with a floor of 1.0: 79/80; the page shows 41.1 at score 19, ages 70-79 (41.4)."""
    tab = np.array(WEB["risk_by_score"])
    ours = np.array([[max(round(100 * float(hisayama_points_risk(s + 2, a)), 1), 1.0) for a in (45, 55, 65, 75)]
                     for s in range(20)])
    assert list(zip(*np.nonzero(ours != tab))) == [(19, 3)]
    assert (ours[19, 3], tab[19, 3]) == (41.4, 41.1)


def test_jcvrisk_points_risk():
    """R package Jcvrisk 0.1.3 hisayama() (vendored) on 2,702 patients aged 40-84: equal to 1e-12."""
    d = pd.read_csv(REF / "jcvrisk/fixtures/patients.csv")
    r = pd.read_csv(REF / "jcvrisk/fixtures/jcvrisk_output.csv")
    ok = d.age < 85
    sex = np.where(d.male == 1, "male", "female")
    ours = hisayama_points_risk(hisayama_points(sex, d.sbp, d.t2dm, d.hdl, d.urineprotein, d.smoking, d.exercise,
                                                ldl=d.ldl), d.age)
    assert ok.sum() == 2702
    assert np.abs(ours[ok] - r.hisayama[ok]).max() < 1e-12


def test_equation_and_ranges():
    r = hisayama([39, 60, 85], "male", 140, 0, 50, 0, 1, 0, ldl=130)
    assert np.isnan(r[[0, 2]]).all()
    lp = 0.077 * 60 + 0.984 + 1.4 - 0.6 + 0.65 + 0.336 + 0.339
    assert r[1] == pytest.approx(1 - 0.9696 ** np.exp(lp - 6.7963), rel=1e-12)
    assert np.isnan(hisayama_points_risk(10, [39, 85])).all()
    with pytest.raises(ValueError):
        hisayama(60, "male", 140, 0, 50, 0, 1, 0)
    with pytest.raises(ValueError):
        hisayama_points("male", 140, 0, 50, 0, 1, 0, ldl=130, non_hdl=160)


def test_jas2022_official_app():
    """JAS's official 2022 app (j-athero.org gl2022app, browser JS run unmodified in Node by
    reference/hisayama/run_jas2022_app.js) on 3,920 patients: shown 10-year risk (1 decimal, '1.0% or less' at total
    <= 7), risk category (<2, 2-<10, >=10%) and the ratio's reference risk all equal Fig. 1
    points (age, sex, SBP, HDL-C, LDL-C, smoking) + 1 for impaired glucose tolerance, put through the paper's formula
    at score + 2 (the no-exercise points; the chart has no exercise or proteinuria item)."""
    d = pd.DataFrame(json.loads((REF / "hisayama/fixtures/jas2022_app_reference.json").read_text())["rows"])
    assert len(d) == 3920 and "error" not in d
    age = d.old.to_numpy()
    sex = np.where(d.sex == 7, "male", "female")
    ldl = d.ldl.fillna(np.trunc(d.tc - d.hdl - d.tg / 5)).to_numpy()  # app's Friedewald, parseInt truncation
    assert (ldl == d.ldl_used).all()
    score = hisayama_points(sex, d.systolicBloodPressure, 0, d.hdl, 0, d.smoking == 2, 1, ldl=ldl) + d.diabeteMellitus
    total = score + np.select([age >= 70, age >= 60, age >= 50], [16, 11, 5], 0)
    risk = 100 * hisayama_points_risk(score + 2, age)
    shown = np.where(total <= 7, 1.0, np.where(total >= 37, 50.0, np.round(risk, 1)))
    assert (shown == d.rate).all()
    assert (np.digitize(risk, [2, 10]) == d.risk).all()
    assert (np.maximum(np.round(100 * hisayama_points_risk(np.where(d.sex == 7, 9, 2), age), 1), 1.0) == d.rateMin).all()
    assert set(d.risk) == {0, 1, 2} and total.min() == 0 and total.max() == 35  # 36 and ">= 37" rows unreachable
