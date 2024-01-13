"""
Tests for Refactored SCORE2.
"""

import pytest
import pandas as pd
from cvd_risk.models.score2 import SCORE2, SCORE2Input
from pydantic import ValidationError

def test_score2_regions():
    """Test calculation across different risk regions."""
    model = SCORE2()
    
    # Test cases: 50yo Male Smoker
    # Low Risk Region vs Very High Risk Region
    df = pd.DataFrame([
        {
            "age": 50, "sex": "male", "smoking": True,
            "systolic_bp": 140, "total_cholesterol": 5.0, "hdl_cholesterol": 1.0,
            "region": "low"
        },
        {
            "age": 50, "sex": "male", "smoking": True,
            "systolic_bp": 140, "total_cholesterol": 5.0, "hdl_cholesterol": 1.0,
            "region": "very_high"
        }
    ])
    
    res = model.calculate_batch(df)
    
    # Very high risk region should strictly be higher risk for same inputs
    assert res.iloc[1]["risk_score"] > res.iloc[0]["risk_score"]
    
def test_score2_validation():
    with pytest.raises(ValidationError):
        SCORE2Input(
            age=50, sex="male", systolic_bp=120, total_cholesterol=4, hdl_cholesterol=1,
            region="mars", # Invalid region
            smoking=False
        )

def test_age_adjusted_categories():
    """Test that risk categories follow age-specific thresholds."""
    model = SCORE2()
    
    # 5% risk
    # For <50yo (Cutoff 2.5/7.5) -> Moderate
    # For >70yo (Cutoff 7.5/15) -> Low
    
    cat_young = model._get_risk_class_string(5.0, 45)
    cat_old = model._get_risk_class_string(5.0, 75)
    
    assert "Moderate" in cat_young or "High" in cat_young # 2.5 < 5.0 < 7.5 -> Moderate (actually it is High in my implementation logic? Wait.)
    # Logic: < 2.5 Low, < 7.5 High. 5.0 is < 7.5 so High.
    # Logic Update: I named it "High" for the middle bucket in the code because SCORE2 is weird. 
    # Let's check my code: 
    # if score < cuts[0]: "Low-to-Moderate"
    # if score < cuts[1]: "High"
    
    # 5.0 vs (2.5, 7.5): 5.0 > 2.5 and < 7.5 -> "High"
    assert cat_young == "High"
    
    # 5.0 vs (7.5, 15.0): 5.0 < 7.5 -> "Low-to-Moderate"
    assert cat_old == "Low-to-Moderate"
