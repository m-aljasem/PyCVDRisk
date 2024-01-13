"""
Tests for Batch 4 Refactor.
Models: SCORE2-OP, SCORE2-DM.
"""

import pytest
import pandas as pd
import numpy as np
from cvd_risk.models.score2op import SCORE2OP, SCORE2OPInput
from cvd_risk.models.score2dm import SCORE2DM, SCORE2DMInput
from pydantic import ValidationError


def test_score2op_calculation():
    model = SCORE2OP()
    p = SCORE2OPInput(
        age=75, sex="male", systolic_bp=150, total_cholesterol=5.5,
        hdl_cholesterol=1.2, smoking=False, diabetes=True, region="moderate"
    )
    res = model.calculate(p)
    assert "risk_score" in res
    assert 0 <= res["risk_score"] <= 100
    assert res["model"] == "SCORE2-OP"


def test_score2op_age_validation():
    with pytest.raises(ValidationError):
        SCORE2OPInput(
            age=65, sex="male", systolic_bp=150, total_cholesterol=5.5,
            hdl_cholesterol=1.2, smoking=False, diabetes=True, region="moderate"
        )


def test_score2op_batch():
    model = SCORE2OP()
    df = pd.DataFrame([
        {"age": 75, "sex": "male", "systolic_bp": 150, "total_cholesterol": 5.5, "hdl_cholesterol": 1.2, "smoking": False, "diabetes": True, "region": "moderate"},
        {"age": 80, "sex": "female", "systolic_bp": 140, "total_cholesterol": 6.0, "hdl_cholesterol": 1.5, "smoking": True, "diabetes": False, "region": "high"}
    ])
    res = model.calculate_batch(df)
    assert len(res) == 2
    assert not res["risk_score"].isna().any()


def test_score2dm_calculation():
    model = SCORE2DM()
    p = SCORE2DMInput(
        age=55, sex="male", systolic_bp=140, total_cholesterol=6.0,
        hdl_cholesterol=1.1, smoking=True, region="high",
        diabetes_age=45, hba1c=55, egfr=75
    )
    res = model.calculate(p)
    assert 0 <= res["risk_score"] <= 100
    assert res["model"] == "SCORE2-DM"


def test_score2dm_batch():
    model = SCORE2DM()
    df = pd.DataFrame([
        {"age": 55, "sex": "male", "systolic_bp": 140, "total_cholesterol": 6.0, "hdl_cholesterol": 1.1, "smoking": True, "region": "high", "diabetes_age": 45, "hba1c": 55, "egfr": 75},
        {"age": 60, "sex": "female", "systolic_bp": 130, "total_cholesterol": 5.5, "hdl_cholesterol": 1.4, "smoking": False, "region": "low", "diabetes_age": 50, "hba1c": 48, "egfr": 90}
    ])
    res = model.calculate_batch(df)
    assert len(res) == 2
    assert not res["risk_score"].isna().any()

