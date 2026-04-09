import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk.auscvd import auscvd
mod = importlib.import_module("cvd_risk.auscvd.auscvd")

REF = Path(__file__).parents[2] / "reference" / "auscvd"
SMOKING = {0: "never", 1: "ex", 2: "current"}


def _ours(g, diabetes_equation):
    args = (g.age, g.sex.map({0: "female", 1: "male"}), g.smoking.map(SMOKING), g.sbp, g.tc_hdl, g.diabetes, g.af, g.bp, g.lipid, g.antithrombotic, g.quintile)
    if diabetes_equation:
        return auscvd(*args, diabetes_years=g.years, bmi=g.bmi, egfr=g.egfr, hba1c=g.hba1c, uacr=g.uacr,
                      insulin=g.insulin)
    return auscvd(*args)


def test_matches_official_calculator():
    """reference/auscvd/run_calculator.js: the calculator's four equations, cut from its bundle, on 6,000
    seeded patients (3,000 general equation, 20% of them with diabetes; 3,000 diabetes equation)."""
    c = pd.read_csv(REF / "fixtures" / "calculator_reference.csv")
    for eq, g in c.groupby("equation"):
        ours = 100 * _ours(g, eq == "diabetes")
        a = g.age.to_numpy()
        np.testing.assert_array_equal(np.where(g.sex == 1, mod._factor(a, "male"), mod._factor(a, "female")), g.factor)
        # the calculator floors the unrecalibrated risk at 1% before multiplying: max(1, p) * f = max(f, f * p)
        np.testing.assert_allclose(np.maximum(ours, g.factor), g.unrounded, rtol=1e-6)
        np.testing.assert_array_equal(np.floor(np.maximum(ours, g.factor) + 0.5), g.shown)


def test_recalibration_factors_are_mja_box5_rate_ratios():
    box5 = {"female": [1.3, 1.2, 1.2, 1.0, 0.9, 0.8, 0.7, 0.7], "male": [1.1, 1.1, 1.1, 1.0, 0.9, 0.8, 0.8, 0.8]}
    bands = np.array([30, 45, 50, 55, 60, 65, 70, 75])  # 30-44, 45-49, ..., 75-79
    for sex, ratios in box5.items():
        for lo, hi in zip(bands, [*bands[1:], 80]):
            assert set(mod._factor(np.arange(lo, hi), sex)) == {ratios[list(bands).index(lo)]}


def test_centring_values_are_the_cohort_means_in_mja_boxes_1_and_6():
    """Box 1 (AUS-PREDICT) and Box 6 (Diabetes): means, and the deprivation quintile mean from its distribution."""
    printed = {  # age, SBP, TC/HDL; deprivation quintile shares (%) 1..5
        ("general", "female"): ((58.5, 129, 3.6), (27.5, 23.3, 20.2, 17.3, 11.8)),
        ("general", "male"): ((53.8, 129, 4.3), (27.6, 23.4, 19.8, 17.1, 12.1)),
        ("diabetes", "female"): ((57.8, 132, 3.9), (17.8, 20.2, 21.1, 22.6, 18.3)),
        ("diabetes", "male"): ((56.6, 132, 4.3), (21.0, 20.7, 21.0, 21.3, 16.0)),
    }
    for (eq, sex), ((age, sbp, ratio), shares) in printed.items():
        centre = (mod.GENERAL if eq == "general" else mod.DIABETES)[sex][0]
        assert round(centre["age"], 1) == age and round(centre["sbp"]) == sbp and round(centre["tc_hdl"], 1) == ratio
        assert abs(np.dot(shares, range(1, 6)) / 100 - centre["dep"]) < 0.005  # 1 = least deprived, as 6 - SEIFA
    diabetes_means = {"female": (5.2, 30, 88.4, 56.1), "male": (5.3, 29, 88.2, 57.8)}  # years, BMI, eGFR, HbA1c
    for sex, (yrs, bmi, egfr, hba1c) in diabetes_means.items():
        c = mod.DIABETES[sex][0]
        assert (round(c["years"], 1), round(c["bmi"]), round(c["egfr"], 1), round(c["hba1c"], 1)) == (yrs, bmi, egfr,
                                                                                                       hba1c)


def test_mja_worked_example_recalibration():
    """MJA 2025: 'if the estimated 5-year risk for a 65-year-old man is 14%, the recalibrated risk is 11.2%'."""
    assert mod._factor(np.array([65.0]), "male")[0] * 14 == pytest.approx(11.2)


def test_arguments_and_invalid_rows():
    base = (60, "male", "never", 130, 4.0, 1, 0, 0, 0, 0, 3)
    with pytest.raises(ValueError):
        auscvd(*base, bmi=30)
    r = auscvd([60, 29, 80, 60, 60], "male", "never", 130, 4.0, 0, 0, 0, 0, 0, [3, 3, 3, 0, 6])
    assert np.isfinite(r[0]) and np.isnan(r[1:]).all()
    dm = dict(diabetes_years=5, bmi=30, egfr=90, hba1c=55, uacr=1, insulin=0)
    general, specific = auscvd(*base), auscvd(*base, **dm)
    assert general != specific
    assert auscvd(60, "male", "never", 130, 4.0, 0, 0, 0, 0, 0, 3, **dm) == auscvd(*base[:5], 0, *base[6:])


def test_risk_falls_with_less_disadvantage_and_rises_with_smoking():
    r = auscvd(60, "female", ["never", "current"] + ["never"] * 2, 130, 4.0, 0, 0, 0, 0, 0, [3, 3, 1, 5])
    assert r[1] > r[0] and r[2] > r[0] > r[3]
