"""Pytest configuration and fixtures."""

import pytest

from cvd_risk_calculator.core.validation import PatientData


@pytest.fixture
def sample_patient() -> PatientData:
    """Create a sample patient for testing."""
    return PatientData(
        age=55,
        sex="male",
        systolic_bp=140.0,
        total_cholesterol=6.0,
        hdl_cholesterol=1.2,
        smoking=True,
        region="moderate",
    )


@pytest.fixture
def sample_patient_female() -> PatientData:
    """Create a sample female patient for testing."""
    return PatientData(
        age=50,
        sex="female",
        systolic_bp=130.0,
        total_cholesterol=5.5,
        hdl_cholesterol=1.5,
        smoking=False,
        region="low",
    )


@pytest.fixture
def sample_patient_high_risk() -> PatientData:
    """Create a high-risk patient for testing."""
    return PatientData(
        age=65,
        sex="male",
        systolic_bp=160.0,
        total_cholesterol=7.5,
        hdl_cholesterol=0.9,
        smoking=True,
        region="very_high",
    )


@pytest.fixture
def sample_patient_low_risk() -> PatientData:
    """Create a low-risk patient for testing."""
    return PatientData(
        age=45,
        sex="female",
        systolic_bp=110.0,
        total_cholesterol=4.5,
        hdl_cholesterol=1.8,
        smoking=False,
        region="low",
    )

