import importlib
import math

import numpy as np
import pytest

from cvd_risk.suita import bp_category, suita, suita_risk
mod = importlib.import_module("cvd_risk.suita.suita")

# Nishimura 2014 Table 3a (TC Suita Score with CKD) and Table 4a (LDL Suita Score with CKD): beta
TABLE3A = {"age": 0.0766382, "female": -0.5866078, "smoker": 0.4865127, "diabetes": 0.4557071,
           "bp": (-0.7183575, 0, 0, 0.3330895, 0.59332684), "chol": (-1.112393, 0, 0.5110573, 0.8511397),
           "hdl": (0.6173452, 0, -0.5096169, -0.4322771), "ckd": (1.315004, 0.3278965, 0)}
TABLE4A = {"age": 0.0760078, "female": -0.6619839, "smoker": 0.5031949, "diabetes": 0.5533678,
           "bp": (-0.6763825, 0, 0, 0.4189501, 0.5935986), "chol": (0, 0.5319015, 0.6837867, 1.021015, 1.128479),
           "hdl": (0, -0.4730423, -0.5822414), "ckd": (1.388216, 0.2893668, 0)}
R = lambda x: math.floor(10 * x + 0.5)


def test_ldl_sheet_points_are_ten_times_the_coefficients():
    """Paper: categorical points = 10 x beta, rounded; age points = 10 x beta x band midpoint."""
    s, b = mod.SHEETS["ldl"], TABLE4A
    for k in ("female", "smoker", "diabetes"):
        assert s[k] == R(b[k]), k
    for k in ("bp", "chol", "hdl", "ckd"):
        assert list(s[k][1] if isinstance(s[k][0], tuple) else s[k]) == [R(x) for x in b[k]], k
    midpoints = (39.5, 49.5, 59.5, 67)  # 35-44, 45-54, 55-64, 65-69
    assert [R(b["age"] * m) for m in midpoints] == list(s["age"][1][1:5])


def test_tc_sheet_points_and_where_they_depart_from_table3a():
    """Most TC-sheet points are 10 x beta; three are not (REPORT.md, findings)."""
    s, b = mod.SHEETS["tc"], TABLE3A
    for k in ("female", "smoker", "diabetes"):
        assert s[k] == R(b[k]), k
    assert list(s["bp"]) == [R(x) for x in b["bp"]] and list(s["chol"][1]) == [R(x) for x in b["chol"]]
    assert [R(x) for x in b["hdl"]] == [6, 0, -5, -4] and list(s["hdl"][1]) == [6, 0, -5, -5]  # HDL >= 60
    assert [R(x) for x in b["ckd"]] == [13, 3, 0] and list(s["ckd"]) == [15, 3, 0]  # CKD stage 4-5
    # age 36-45 prints 30, which is 10 x beta x 39.5 (the 35-44 midpoint), not x 40.5 (31)
    assert [R(b["age"] * m) for m in (39.5, 40.5, 50.5, 60.5)] == [30, 31, 39, 46]
    assert list(s["age"][1][1:4]) == [30, 39, 46]


@pytest.mark.parametrize("sheet", ["tc", "ldl"])
def test_probability_table_fits_one_baseline_survival(sheet):
    """P = 1 - S ** exp(points / 10) (paper); S is unpublished, but one S reproduces every bounded row of the
    printed table at the row's middle score, rounded to whole percent."""
    rows, lo = mod.RISK[sheet], {"tc": 31, "ldl": 36}[sheet]
    feasible = (0.0, 1.0)
    for hi, label in rows[1:-1]:
        p, mid = float(label) / 100, (lo + hi) / 2
        s_hi = (1 - max(p - 0.005, 0)) ** math.exp(-mid / 10)  # P >= p - 0.5 point
        s_lo = (1 - (p + 0.005)) ** math.exp(-mid / 10)
        feasible = (max(feasible[0], s_lo), min(feasible[1], s_hi))
        lo = hi + 1
    assert feasible[0] < feasible[1], feasible


def test_bands_and_worked_profile():
    # 60-year-old man, smoker, no diabetes, 145/85 (stage 1), TC 250, HDL 45, eGFR 50 (stage 3)
    assert suita(60, "male", 1, 0, 145, 85, 45, 50, tc=250) == 46 + 5 + 3 + 5 + 0 + 3
    assert suita(60, "male", 1, 0, 145, 85, 45, 50, ldl=150) == 45 + 5 + 4 + 7 - 5 + 3  # HDL 40-59: -5 on this sheet
    assert suita_risk(62, "tc") == "24" and suita_risk(62, "ldl") == "14" and suita_risk(30, "tc") == "<1"
    np.testing.assert_array_equal(bp_category([119, 120, 135, 125, 160], [79, 79, 70, 95, 60]), [0, 1, 2, 3, 4])
    hdl = suita(50, "female", 0, 0, 125, 75, [34.9, 35, 49.9, 50, 60], 90, tc=200)
    np.testing.assert_array_equal(np.diff(hdl), [-6, 0, -5, 0])
    ckd = suita(50, "female", 0, 0, 125, 75, 55, [29.9, 30, 59.9, 60], ldl=120)
    np.testing.assert_array_equal(ckd - ckd[-1], [14, 3, 3, 0])


def test_ages_outside_sheets_and_arguments():
    r = suita([35, 36, 79, 80], "male", 0, 0, 125, 75, 55, 90, tc=200)
    assert np.isnan(r[0]) and np.isfinite(r[1:3]).all() and np.isnan(r[3])
    assert np.isfinite(suita(35, "male", 0, 0, 125, 75, 55, 90, ldl=120))
    with pytest.raises(ValueError):
        suita(50, "male", 0, 0, 125, 75, 55, 90)
    with pytest.raises(ValueError):
        suita(50, "male", 0, 0, 125, 75, 55, 90, tc=200, ldl=120)
    assert suita_risk(np.nan, "ldl") == ""


def test_jas2017_web_calculators_share_the_ldl_sheet():
    """calculator.jp and dataclock.jp (JAS 2017 adaptation of the LDL sheet): every item it shares with Table 5b, and
    the probability rows, equal ours; diabetes 6 -> glucose intolerance 5, CKD dropped, family history +5 added."""
    import json
    from pathlib import Path
    web = json.loads((Path(__file__).parents[2] / "reference" / "suita" / "fixtures" / "jas2017_web_calculators.json")
                     .read_text(encoding="utf-8"))
    p, s = web["points"], mod.SHEETS["ldl"]
    assert p["age (35-44, 45-54, 55-64, 65-69, >=70)"] == list(s["age"][1][1:6])
    assert (p["female"], p["smoker"]) == (s["female"], s["smoker"])
    assert p["bp (optimal, normal, high normal, stage 1, stage 2)"] == list(s["bp"])
    assert p["hdl (<40, 40-59, >=60)"] == list(s["hdl"][1])
    assert p["ldl (<100, 100-139, 140-159, 160-179, >=180)"] == list(s["chol"][1])
    assert [lab for _, lab in web["risk_by_total (dataclock)"]] == [lab for _, lab in mod.RISK["ldl"]]
    assert [hi for hi, _ in web["risk_by_total (dataclock)"][:-1]] == [hi for hi, _ in mod.RISK["ldl"][:-1]]


def test_ncvc_2014_press_release_sheet_equals_the_ldl_sheet():
    """The developers' own sheet (NCVC press release, May 2014): every point and probability row equals ours; its BP
    row 'normal' (0) carries the paper's 'normal and high normal', and it labels CKD stage 3 as eGFR 30-59."""
    import json
    from pathlib import Path
    web = json.loads((Path(__file__).parents[2] / "reference" / "suita" / "fixtures" / "ncvc_2014_press_sheet.json")
                     .read_text(encoding="utf-8"))
    p, s = list(web["points"].values()), mod.SHEETS["ldl"]
    assert p[0] == list(s["age"][1][1:6])
    assert p[1:4] == [s["female"], s["smoker"], s["diabetes"]]
    assert p[4] == [s["bp"][0], s["bp"][1], s["bp"][3], s["bp"][4]] and s["bp"][1] == s["bp"][2]
    assert p[5] == list(s["chol"][1]) and p[6] == list(s["hdl"][1]) and p[7] == list(s["ckd"])[::-1]
    assert [lab for _, lab in web["risk_by_total"]] == [lab for _, lab in mod.RISK["ldl"]]
    assert [hi for hi, _ in web["risk_by_total"][:-1]] == [hi for hi, _ in mod.RISK["ldl"][:-1]]
    ckd = suita(50, "female", 0, 0, 125, 75, 55, [29.9, 30, 59.9, 60], ldl=120)
    np.testing.assert_array_equal(ckd - ckd[-1], [14, 3, 3, 0])  # eGFR 30-59 is stage 3, as the sheet labels it
