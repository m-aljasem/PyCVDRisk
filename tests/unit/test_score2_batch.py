"""Additional tests for SCORE2 batch processing and edge cases."""

import pandas as pd
import pytest

from cvd_risk_calculator.core.validation import PatientData
from cvd_risk_calculator.models.score2 import SCORE2


class TestSCORE2Batch:
    """Test SCORE2 batch processing."""

    def test_batch_processing_success(self) -> None:
        """Test successful batch processing."""
        model = SCORE2()
        df = pd.DataFrame(
            {
                "age": [55, 60, 45],
                "sex": ["male", "female", "male"],
                "systolic_bp": [140.0, 130.0, 150.0],
                "total_cholesterol": [6.0, 5.5, 7.0],
                "hdl_cholesterol": [1.2, 1.5, 1.0],
                "smoking": [True, False, True],
                "region": ["moderate", "low", "high"],
            }
        )

        results = model.calculate_batch(df)

        assert len(results) == 3
        assert "risk_score" in results.columns
        assert "risk_category" in results.columns
        assert "model_name" in results.columns
        assert all(results["model_name"] == "SCORE2")
        assert all(results["risk_score"] >= 0)
        assert all(results["risk_score"] <= 100)

    def test_batch_missing_columns(self) -> None:
        """Test batch processing with missing columns."""
        model = SCORE2()
        df = pd.DataFrame(
            {
                "age": [55, 60],
                "sex": ["male", "female"],
                # Missing required columns
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            model.calculate_batch(df)

    def test_batch_with_invalid_data(self) -> None:
        """Test batch processing handles invalid data gracefully."""
        model = SCORE2()
        df = pd.DataFrame(
            {
                "age": [55, 15],  # Invalid age
                "sex": ["male", "female"],
                "systolic_bp": [140.0, 130.0],
                "total_cholesterol": [6.0, 5.5],
                "hdl_cholesterol": [1.2, 1.5],
                "smoking": [True, False],
                "region": ["moderate", "low"],
            }
        )

        results = model.calculate_batch(df)

        # First row should succeed
        assert not pd.isna(results.iloc[0]["risk_score"])
        # Second row should have error
        assert pd.isna(results.iloc[1]["risk_score"])
        assert results.iloc[1]["risk_category"] == "error"

    def test_risk_categorization_boundary_values(self) -> None:
        """Test risk categorization at boundary values."""
        model = SCORE2()

        # Test low boundary (< 5%)
        assert model._categorize_risk(4.9) == "low"
        assert model._categorize_risk(0.0) == "low"

        # Test moderate boundary (5-10%)
        assert model._categorize_risk(5.0) == "moderate"
        assert model._categorize_risk(7.5) == "moderate"
        assert model._categorize_risk(9.9) == "moderate"

        # Test high boundary (10-20%)
        assert model._categorize_risk(10.0) == "high"
        assert model._categorize_risk(15.0) == "high"
        assert model._categorize_risk(19.9) == "high"

        # Test very_high boundary (>= 20%)
        assert model._categorize_risk(20.0) == "very_high"
        assert model._categorize_risk(50.0) == "very_high"
        assert model._categorize_risk(100.0) == "very_high"

    def test_all_regions(self) -> None:
        """Test calculation works for all supported regions."""
        model = SCORE2()
        patient_base = {
            "age": 55,
            "sex": "male",
            "systolic_bp": 140.0,
            "total_cholesterol": 6.0,
            "hdl_cholesterol": 1.2,
            "smoking": True,
        }

        for region in ["low", "moderate", "high", "very_high"]:
            patient = PatientData(**patient_base, region=region)
            result = model.calculate(patient)
            assert result.risk_score >= 0
            assert result.risk_score <= 100
            assert result.risk_category in ["low", "moderate", "high", "very_high"]

    def test_extreme_values(self) -> None:
        """Test calculation with extreme but valid values."""
        model = SCORE2()

        # Very old patient
        patient_old = PatientData(
            age=69,  # Upper limit of optimal range
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region="moderate",
        )
        result_old = model.calculate(patient_old)
        assert result_old.risk_score >= 0

        # Very young patient
        patient_young = PatientData(
            age=40,  # Lower limit of optimal range
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region="moderate",
        )
        result_young = model.calculate(patient_young)
        assert result_young.risk_score >= 0
        assert result_young.risk_score < result_old.risk_score  # Younger should have lower risk

