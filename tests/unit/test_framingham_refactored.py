"""
Tests for the refactored Framingham model.
"""

import pytest
import pandas as pd
import numpy as np
from cvd_risk.models.framingham import Framingham2008, FraminghamInput

def test_framingham_basics():
    """Test basic instantiation and calculation."""
    model = Framingham2008()
    
    # Test Case: Male, 55, Smoker, Treated BP
    # Values roughly based on paper example or standard calculator
    input_data = FraminghamInput(
        age=55,
        sex="male",
        total_cholesterol=5.52,  # ~213 mg/dL
        hdl_cholesterol=1.29,    # ~50 mg/dL
        systolic_bp=140,
        treated_bp=True,
        smoking=True,
        diabetes=False
    )
    
    result = model.calculate(input_data)
    
    # Just check structure and sanity constraints for now
    assert "risk_score" in result
    assert 0 <= result["risk_score"] <= 100
    assert result["model"] == "Framingham 2008"
    print(f"Risk Score: {result['risk_score']}%")


def test_framingham_batch():
    """Test vectorized batch calculation."""
    model = Framingham2008()
    
    df = pd.DataFrame({
        "age": [55, 60],
        "sex": ["male", "female"],
        "total_cholesterol": [5.5, 6.0],
        "hdl_cholesterol": [1.2, 1.5],
        "systolic_bp": [140, 130],
        "treated_bp": [True, False],
        "smoking": [True, False],
        "diabetes": [False, True]
    })
    
    results = model.calculate_batch(df)
    
    assert "risk_score" in results.columns
    assert len(results) == 2
    assert not results["risk_score"].isna().any()
    
    # Check that female risk is calculated (different pathway)
    print("Batch Results:")
    print(results[["age", "sex", "risk_score"]])

from pydantic import ValidationError

def test_validation_error():
    """Test that invalid input raises error."""
    with pytest.raises(ValidationError):
        FraminghamInput(
            age=10, 
            sex="male",
            total_cholesterol=0.5, # Invalid: < 1.0
            hdl_cholesterol=1.0,
            systolic_bp=120
        )
