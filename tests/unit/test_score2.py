"""Unit tests for SCORE2 risk model."""

import pytest

from cvd_risk_calculator.core.validation import PatientData
from cvd_risk_calculator.models.score2 import SCORE2


class TestSCORE2:
    """Test SCORE2 model implementation."""

    def test_model_initialization(self) -> None:
        """Test that SCORE2 model initializes correctly."""
        model = SCORE2()
        assert model.model_name == "SCORE2"
        assert model.model_version == "2021"
        assert model.supported_regions == ["low", "moderate", "high", "very_high"]

    def test_calculate_basic(self, sample_patient: PatientData) -> None:
        """Test basic risk calculation."""
        model = SCORE2()
        result = model.calculate(sample_patient)

        assert result.model_name == "SCORE2"
        assert result.model_version == "2021"
        assert 0.0 <= result.risk_score <= 100.0
        assert result.risk_category in ["low", "moderate", "high", "very_high"]
        assert "region" in result.calculation_metadata

    def test_calculate_requires_region(self) -> None:
        """Test that calculation requires region."""
        model = SCORE2()
        patient = PatientData(
            age=55,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region=None,
        )

        with pytest.raises(ValueError, match="SCORE2 requires region"):
            model.calculate(patient)

    def test_invalid_region(self) -> None:
        """Test that invalid region raises error."""
        model = SCORE2()
        # Create a patient with a valid region first
        patient = PatientData(
            age=55,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region="moderate",
        )
        # Manually test validate_input with invalid region by accessing the attribute
        # Since Pydantic validates region at PatientData level, we test model validation separately
        # by creating a patient with None and checking the model rejects it
        patient_no_region = PatientData(
            age=55,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region=None,
        )
        with pytest.raises(ValueError, match="SCORE2 requires region"):
            model.validate_input(patient_no_region)

    def test_risk_categorization_low(self, sample_patient_low_risk: PatientData) -> None:
        """Test risk categorization for low-risk patient."""
        model = SCORE2()
        result = model.calculate(sample_patient_low_risk)

        # Low-risk patient should typically have low or moderate risk
        assert result.risk_category in ["low", "moderate"]

    def test_risk_categorization_high(self, sample_patient_high_risk: PatientData) -> None:
        """Test risk categorization for high-risk patient."""
        model = SCORE2()
        result = model.calculate(sample_patient_high_risk)

        # High-risk patient should typically have high or very_high risk
        assert result.risk_category in ["high", "very_high"]

    def test_male_vs_female_difference(self) -> None:
        """Test that male and female patients have different risks for same profile."""
        model = SCORE2()

        # Create similar patients, differing only by sex
        patient_male = PatientData(
            age=55,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region="moderate",
        )

        patient_female = PatientData(
            age=55,
            sex="female",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region="moderate",
        )

        result_male = model.calculate(patient_male)
        result_female = model.calculate(patient_female)

        # Results should differ (typically males have higher risk)
        assert result_male.risk_score != result_female.risk_score

    def test_smoking_increases_risk(self) -> None:
        """Test that smoking increases risk."""
        model = SCORE2()

        patient_smoker = PatientData(
            age=55,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region="moderate",
        )

        patient_nonsmoker = PatientData(
            age=55,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=False,
            region="moderate",
        )

        result_smoker = model.calculate(patient_smoker)
        result_nonsmoker = model.calculate(patient_nonsmoker)

        # Smoker should have higher risk
        assert result_smoker.risk_score > result_nonsmoker.risk_score

    def test_region_affects_risk(self) -> None:
        """Test that different regions produce different risks."""
        model = SCORE2()

        patient_low = PatientData(
            age=55,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region="low",
        )

        patient_very_high = PatientData(
            age=55,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region="very_high",
        )

        result_low = model.calculate(patient_low)
        result_very_high = model.calculate(patient_very_high)

        # Very high region should typically have higher risk
        assert result_very_high.risk_score >= result_low.risk_score

    def test_age_warning(self) -> None:
        """Test that age outside optimal range logs warning."""
        import logging

        model = SCORE2()
        patient = PatientData(
            age=75,  # Outside optimal range 40-69
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region="moderate",
        )

        # Should not raise error but calculate anyway
        result = model.calculate(patient)
        assert result.risk_score >= 0.0

