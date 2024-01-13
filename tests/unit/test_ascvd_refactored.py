"""
Tests for the refactored ASCVD model.
"""

import pytest
import pandas as pd
import numpy as np
from cvd_risk.models.ascvd import ASCVD2013, ASCVDInput
from pydantic import ValidationError

def test_ascvd_subgroups():
    """Test calculation for different race/sex subgroups."""
    model = ASCVD2013()
    
    # Test cases representing each major equation branch
    data = [
        # White Male 55 optimized scenario
        {"age": 55, "sex": "male", "race": "white", "total_cholesterol": 5.5, "hdl_cholesterol": 1.2, "systolic_bp": 140, "treated_bp": False, "smoking": False, "diabetes": False},
        # Black Female 60 (complex equation check)
        {"age": 60, "sex": "female", "race": "black", "total_cholesterol": 6.0, "hdl_cholesterol": 1.5, "systolic_bp": 150, "treated_bp": True, "smoking": True, "diabetes": True},
        # Other (should map to White)
        {"age": 50, "sex": "male", "race": "other", "total_cholesterol": 5.0, "hdl_cholesterol": 1.3, "systolic_bp": 120, "treated_bp": False, "smoking": False, "diabetes": False},
    ]
    
    df = pd.DataFrame(data)
    results = model.calculate_batch(df)
    
    assert len(results) == 3
    assert not results["risk_score"].isna().any()
    assert (results["risk_score"] > 0).all() and (results["risk_score"] < 100).all()
    
    # Check that race="other" produced a result
    assert not np.isnan(results.iloc[2]["risk_score"])

def test_ascvd_validation():
    """Test input validation."""
    # Invalid Race
    with pytest.raises(ValidationError):
        ASCVDInput(
            age=55, sex="male", race="mars_colonist", # Invalid
            total_cholesterol=5.0, hdl_cholesterol=1.2, systolic_bp=120
        )

def test_calculate_single():
    """Test single patient calculation wrapper."""
    model = ASCVD2013()
    patient = ASCVDInput(
        age=55, sex="male", race="white",
        total_cholesterol=5.0, hdl_cholesterol=1.2, systolic_bp=120
    )
    res = model.calculate(patient)
    assert "risk_score" in res
    assert res['model'] == "ASCVD 2013"
