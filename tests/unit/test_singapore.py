"""SG-FRS-2023 tests. Expected values are hand-computed from ACE Tables A1-A4."""

import pytest

from cvd_risk.core.data_classes import SingaporePatientData
from cvd_risk.models import Singapore


def _patient(**overrides) -> SingaporePatientData:
    base = dict(
        age=55, sex="male", systolic_bp=135.0, total_cholesterol=5.5, hdl_cholesterol=1.1,
        smoking=True, ethnicity="chinese", antihypertensive=False,
    )
    return SingaporePatientData(**{**base, **overrides})


@pytest.mark.parametrize(
    "overrides, points, risk, display, category",
    [
        # Men: age 8 + TC(5.2-6.1, 50-59) 3 + smoker(50-59) 3 + HDL(1.0-1.2) 1 + SBP(130-139 untreated) 1 = 16
        ({}, 16, 11.0, "11%", "intermediate"),
        ({"ethnicity": "malay"}, 16, 18.0, "18%", "intermediate"),
        ({"ethnicity": "indian"}, 16, 24.0, "24%", "high"),
        # Treated SBP 130-139 in men = 2 -> 17 points
        ({"antihypertensive": True}, 17, 14.0, "14%", "intermediate"),
        # Low-risk man: age 40-44 0, TC<4.1 0, non-smoker 0, HDL>=1.6 -1, SBP<120 0 = -1
        ({"age": 42, "total_cholesterol": 4.0, "smoking": False, "hdl_cholesterol": 1.7,
          "systolic_bp": 115.0}, -1, 0.5, "<1%", "low"),
        # Men 70-79 smoker = 1 (SG-FRS-2023 footnote a): age(75-79) 13 + TC(>=7.3) 1 + smoker 1
        # + HDL<1.0 2 + SBP>=160 treated 3 = 20 -> Chinese 28%
        ({"age": 77, "total_cholesterol": 8.0, "hdl_cholesterol": 0.9, "systolic_bp": 170.0,
          "antihypertensive": True}, 20, 28.0, "28%", "high"),
        # Same profile, Indian table at its last row (20 points)
        ({"age": 77, "total_cholesterol": 8.0, "hdl_cholesterol": 0.9, "systolic_bp": 170.0,
          "antihypertensive": True, "ethnicity": "indian"}, 20, 54.0, "54%", "high"),
        # Women: age(60-64) 10 + TC(6.2-7.2, 60-69) 3 + smoker(60-69) 2 + HDL(1.3-1.5) 0
        # + SBP(140-159 untreated) 3 = 18 -> Chinese 2%, Malay 4%, Indian 6%
        ({"sex": "female", "age": 62, "total_cholesterol": 6.5, "hdl_cholesterol": 1.4,
          "systolic_bp": 150.0}, 18, 2.0, "2%", "low"),
        ({"sex": "female", "age": 62, "total_cholesterol": 6.5, "hdl_cholesterol": 1.4,
          "systolic_bp": 150.0, "ethnicity": "indian"}, 18, 6.0, "6%", "borderline"),
        # Woman 75-79: age 16 + TC(>=7.3, 70-79) 2 + smoker 1 + HDL<1.0 2 + SBP>=160 treated 6 = 27
        ({"sex": "female", "age": 78, "total_cholesterol": 7.5, "hdl_cholesterol": 0.8,
          "systolic_bp": 165.0, "antihypertensive": True, "ethnicity": "malay"}, 27, 36.0, "36%", "high"),
    ],
)
def test_table_values(overrides, points, risk, display, category) -> None:
    result = Singapore().calculate(_patient(**overrides))
    assert result.calculation_metadata["total_points"] == points
    assert result.risk_score == risk
    assert result.calculation_metadata["risk_display"] == display
    assert result.risk_category == category


def test_points_above_table_are_capped_and_flagged() -> None:
    # Man 45-49: age 3 + TC(>=7.3, 40-49) 8 + smoker 5 + HDL<1.0 2 + SBP>=160 treated 3 = 21 > 20 (table end)
    result = Singapore().calculate(
        _patient(age=47, total_cholesterol=8.0, hdl_cholesterol=0.9, systolic_bp=170.0, antihypertensive=True)
    )
    assert result.calculation_metadata["total_points"] == 21
    assert result.risk_score == 28.0
    assert result.calculation_metadata["risk_display"] == ">=28%"


def test_twenty_percent_is_intermediate() -> None:
    # ACE: intermediate 10-20%, high >20%.
    # Woman: age(65-69) 12 + TC(6.2-7.2, 60-69) 3 + smoker 2 + HDL(1.0-1.2) 1 + SBP(140-159 treated) 5 = 23
    result = Singapore().calculate(
        _patient(sex="female", age=66, total_cholesterol=6.5, hdl_cholesterol=1.1,
                 systolic_bp=150.0, antihypertensive=True, ethnicity="indian")
    )
    assert result.calculation_metadata["total_points"] == 23
    assert result.risk_score == 20.0
    assert result.risk_category == "intermediate"


@pytest.mark.parametrize("overrides", [{"age": 19}, {"age": 80}, {"diabetes": True}])
def test_out_of_scope_patients_rejected(overrides) -> None:
    with pytest.raises(ValueError):
        Singapore().calculate(_patient(**overrides))
