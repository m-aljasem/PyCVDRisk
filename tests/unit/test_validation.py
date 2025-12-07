"""Unit tests for input validation."""

import pytest

from cvd_risk.core.validation import PatientData


class TestPatientData:
    """Test PatientData validation."""

    def test_valid_patient(self) -> None:
        """Test that valid patient data passes validation."""
        patient = PatientData(
            age=55,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region="moderate",
        )
        assert patient.age == 55
        assert patient.sex == "male"
        assert patient.systolic_bp == 140.0

    def test_invalid_age_too_young(self) -> None:
        """Test that age < 18 raises error."""
        with pytest.raises(ValueError):
            PatientData(
                age=15,
                sex="male",
                systolic_bp=140.0,
                total_cholesterol=6.0,
                hdl_cholesterol=1.2,
                smoking=True,
            )

    def test_invalid_age_too_old(self) -> None:
        """Test that age > 120 raises error."""
        with pytest.raises(ValueError):
            PatientData(
                age=125,
                sex="male",
                systolic_bp=140.0,
                total_cholesterol=6.0,
                hdl_cholesterol=1.2,
                smoking=True,
            )

    def test_invalid_sex(self) -> None:
        """Test that invalid sex raises error."""
        with pytest.raises(ValueError):
            PatientData(
                age=55,
                sex="other",  # type: ignore
                systolic_bp=140.0,
                total_cholesterol=6.0,
                hdl_cholesterol=1.2,
                smoking=True,
            )

    def test_invalid_bp_low(self) -> None:
        """Test that BP < 50 raises error."""
        with pytest.raises(ValueError):
            PatientData(
                age=55,
                sex="male",
                systolic_bp=40.0,
                total_cholesterol=6.0,
                hdl_cholesterol=1.2,
                smoking=True,
            )

    def test_invalid_bp_high(self) -> None:
        """Test that BP > 250 raises error."""
        with pytest.raises(ValueError):
            PatientData(
                age=55,
                sex="male",
                systolic_bp=300.0,
                total_cholesterol=6.0,
                hdl_cholesterol=1.2,
                smoking=True,
            )

    def test_invalid_cholesterol_low(self) -> None:
        """Test that cholesterol < 1.0 raises error."""
        with pytest.raises(ValueError):
            PatientData(
                age=55,
                sex="male",
                systolic_bp=140.0,
                total_cholesterol=0.5,
                hdl_cholesterol=1.2,
                smoking=True,
            )

    def test_invalid_hdl_low(self) -> None:
        """Test that HDL < 0.3 raises error."""
        with pytest.raises(ValueError):
            PatientData(
                age=55,
                sex="male",
                systolic_bp=140.0,
                total_cholesterol=6.0,
                hdl_cholesterol=0.1,
                smoking=True,
            )

    def test_optional_fields(self) -> None:
        """Test that optional fields can be None."""
        patient = PatientData(
            age=55,
            sex="male",
            systolic_bp=140.0,
            total_cholesterol=6.0,
            hdl_cholesterol=1.2,
            smoking=True,
            region=None,
            diabetes=True,
            bmi=25.5,
        )
        assert patient.diabetes is True
        assert patient.bmi == 25.5

