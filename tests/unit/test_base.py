"""Unit tests for base RiskModel class."""

import pandas as pd
import pytest

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult


class MockRiskModel(RiskModel):
    """Mock risk model for testing base class functionality."""

    model_name = "MockModel"
    model_version = "1.0"

    def calculate(self, patient: PatientData) -> RiskResult:
        """Mock calculate implementation."""
        return RiskResult(
            risk_score=5.0,
            risk_category="low",
            model_name=self.model_name,
            model_version=self.model_version,
        )


class TestRiskModelBase:
    """Test base RiskModel class."""

    def test_model_attributes(self) -> None:
        """Test that model has required attributes."""
        model = MockRiskModel()
        assert model.model_name == "MockModel"
        assert model.model_version == "1.0"

    def test_calculate_batch_success(self) -> None:
        """Test batch processing with valid data."""
        model = MockRiskModel()
        df = pd.DataFrame(
            {
                "age": [55, 60],
                "sex": ["male", "female"],
                "systolic_bp": [140.0, 130.0],
                "total_cholesterol": [6.0, 5.5],
                "hdl_cholesterol": [1.2, 1.5],
                "smoking": [True, False],
            }
        )

        results = model.calculate_batch(df)

        assert len(results) == 2
        assert "risk_score" in results.columns
        assert "risk_category" in results.columns
        assert "model_name" in results.columns
        assert all(results["risk_score"] == 5.0)
        assert all(results["risk_category"] == "low")

    def test_calculate_batch_with_errors(self) -> None:
        """Test batch processing with invalid data."""
        model = MockRiskModel()
        df = pd.DataFrame(
            {
                "age": [55, 15],  # Second row has invalid age
                "sex": ["male", "female"],
                "systolic_bp": [140.0, 130.0],
                "total_cholesterol": [6.0, 5.5],
                "hdl_cholesterol": [1.2, 1.5],
                "smoking": [True, False],
            }
        )

        results = model.calculate_batch(df)

        assert len(results) == 2
        # First row should succeed
        assert results.iloc[0]["risk_score"] == 5.0
        # Second row should have error
        assert pd.isna(results.iloc[1]["risk_score"]) or results.iloc[1]["risk_score"] is None
        assert results.iloc[1]["risk_category"] == "error"
        assert "error" in results.iloc[1] or "error" in results.columns

    def test_validate_input_base(self) -> None:
        """Test base validate_input method."""
        model = MockRiskModel()

        # Valid patient should pass
        valid_patient = PatientData(
            age=55,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
        )
        model.validate_input(valid_patient)  # Should not raise

        # Test with edge case age (18, which is the minimum)
        edge_age_patient = PatientData(
            age=18,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
        )
        model.validate_input(edge_age_patient)  # Should not raise

        # Invalid BP should raise (BP is validated in base class, not Pydantic)
        # We need to create a patient with valid Pydantic values but invalid for the model
        # Actually, BP is validated by Pydantic too, so we test with valid but edge values
        low_bp_patient = PatientData(
            age=55,
            sex="male",
            systolic_bp=50.0,  # Minimum valid BP
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
        )
        model.validate_input(low_bp_patient)  # Should pass (50 is valid)

        # Test that validate_input works with valid ranges
        high_bp_patient = PatientData(
            age=55,
            sex="male",
            systolic_bp=250.0,  # Maximum valid BP
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
        )
        model.validate_input(high_bp_patient)  # Should pass (250 is valid)

    def test_get_metadata(self) -> None:
        """Test _get_metadata method."""
        model = MockRiskModel()
        metadata = model._get_metadata()

        assert metadata["model_name"] == "MockModel"
        assert metadata["model_version"] == "1.0"
        assert metadata["supported_regions"] is None

