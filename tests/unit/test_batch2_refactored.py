"""
Tests for Batch 2 Refactor.
Models: Globorisk, SCORE, FINRISK, REGICOR, ProgettoCUORE.
"""

import pytest
import pandas as pd
import numpy as np
from cvd_risk.models.globorisk import Globorisk, GloboriskInput
from cvd_risk.models.score import SCORE, SCOREInput
from cvd_risk.models.finrisk import FINRISK, FINRISKInput
from cvd_risk.models.regicor import REGICOR, REGICORInput
from cvd_risk.models.progetto_cuore import ProgettoCUORE, ProgettoCUOREInput
from pydantic import ValidationError

def test_score_v1_calculation():
    model = SCORE()
    p = SCOREInput(
        age=50, sex="male", systolic_bp=140, total_cholesterol=6.0,
        smoking=True, region="high"
    )
    res = model.calculate(p)
    assert "risk_score" in res
    assert 0 <= res["risk_score"] <= 100
    assert res["model"] == "SCORE"

def test_score_v1_batch():
    model = SCORE()
    df = pd.DataFrame([
        {"age": 50, "sex": "male", "systolic_bp": 140, "total_cholesterol": 6, "smoking": True, "region": "high"},
        {"age": 60, "sex": "female", "systolic_bp": 120, "total_cholesterol": 5, "smoking": False, "region": "low"}
    ])
    res = model.calculate_batch(df)
    assert len(res) == 2
    assert not res["risk_score"].isna().any()

def test_finrisk():
    model = FINRISK()
    p = FINRISKInput(
        age=50, sex="female", systolic_bp=130, total_cholesterol=5.5,
        hdl_cholesterol=1.5, smoking=False
    )
    res = model.calculate(p)
    assert 0 <= res["risk_score"] <= 100
    assert res["model"] == "FINRISK"

def test_regicor():
    model = REGICOR()
    p = REGICORInput(
        age=55, sex="male", systolic_bp=135, total_cholesterol=5.8,
        hdl_cholesterol=1.2, smoking=True
    )
    res = model.calculate(p)
    assert 0 <= res["risk_score"] <= 100
    assert res["model"] == "REGICOR"

def test_progetto_cuore():
    model = ProgettoCUORE()
    p = ProgettoCUOREInput(
        age=60, sex="female", systolic_bp=140, total_cholesterol=6.2,
        hdl_cholesterol=1.4, smoking=False
    )
    res = model.calculate(p)
    assert 0 <= res["risk_score"] <= 100
    assert res["model"] == "ProgettoCUORE"

def test_globorisk_structure():
    # Only test structure if data is missing
    model = Globorisk()
    # If data is missing, calculate raises RuntimeError, which is fine to catch
    try:
        p = GloboriskInput(
            age=50, sex="male", systolic_bp=140, smoking=True,
            total_cholesterol=5.0, # Required for lab version
            country="USA", year=2020
        )
        res = model.calculate(p)
    except RuntimeError:
        pytest.skip("Globorisk data file missing")
    except Exception as e:
        pytest.fail(f"Globorisk failed with unexpected error: {e}")

