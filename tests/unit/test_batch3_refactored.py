"""
Tests for Batch 3 Refactor.
Models: QRISK2, QRISK3, ASSIGN, SMART2, SMART-REACH.
"""

import pytest
import pandas as pd
import numpy as np
from cvd_risk.models.qrisk2 import QRISK2, QRISK2Input
from cvd_risk.models.qrisk3 import QRISK3, QRISK3Input
from cvd_risk.models.assign import ASSIGN, ASSIGNInput
from cvd_risk.models.smart2 import SMART2, SMART2Input
from cvd_risk.models.smartreach import SMART_REACH, SMARTREACHInput
from pydantic import ValidationError


def test_qrisk2_calculation():
    model = QRISK2()
    p = QRISK2Input(
        age=55, sex="male", systolic_bp=140, total_cholesterol=6.0,
        hdl_cholesterol=1.2, bmi=28, smoking=True
    )
    res = model.calculate(p)
    assert "risk_score" in res
    assert 0 <= res["risk_score"] <= 100
    assert res["model"] == "QRISK2"


def test_qrisk2_batch():
    model = QRISK2()
    df = pd.DataFrame([
        {"age": 55, "sex": "male", "systolic_bp": 140, "total_cholesterol": 6, "hdl_cholesterol": 1.2, "bmi": 28, "smoking": True},
        {"age": 60, "sex": "female", "systolic_bp": 120, "total_cholesterol": 5.5, "hdl_cholesterol": 1.5, "bmi": 25, "smoking": False}
    ])
    res = model.calculate_batch(df)
    assert len(res) == 2
    assert not res["risk_score"].isna().any()


def test_qrisk3_calculation():
    model = QRISK3()
    p = QRISK3Input(
        age=55, sex="female", systolic_bp=130, total_cholesterol=5.5,
        hdl_cholesterol=1.4, bmi=26, smoking=False
    )
    res = model.calculate(p)
    assert 0 <= res["risk_score"] <= 100
    assert res["model"] == "QRISK3"


def test_assign_calculation():
    model = ASSIGN()
    p = ASSIGNInput(
        age=55, sex="male", systolic_bp=140, total_cholesterol=6.0,
        hdl_cholesterol=1.2, smoking=True
    )
    res = model.calculate(p)
    assert 0 <= res["risk_score"] <= 100
    assert res["model"] == "ASSIGN"


def test_smart2_calculation():
    model = SMART2()
    p = SMART2Input(
        age=60, sex="male", systolic_bp=145, total_cholesterol=5.5,
        hdl_cholesterol=1.1, smoking=True, diabetes=True, region="europe_low",
        hs_crp=2.0, egfr=75, years_since_diagnosis=3,
        has_cad=True, has_cevd=False, has_pad=False, has_aaa=False
    )
    res = model.calculate(p)
    assert 0 <= res["risk_score"] <= 100
    assert res["model"] == "SMART2"


def test_smartreach_calculation():
    model = SMART_REACH()
    p = SMARTREACHInput(
        age=65, sex="male", systolic_bp=140, total_cholesterol=5.0,
        hdl_cholesterol=1.2, smoking=False, diabetes=True, region="western_europe",
        creatinine=100, disease_locations_count=2,
        has_atrial_fibrillation=True, has_heart_failure=False
    )
    res = model.calculate(p)
    assert 0 <= res["risk_score"] <= 100
    assert res["model"] == "SMART-REACH"

