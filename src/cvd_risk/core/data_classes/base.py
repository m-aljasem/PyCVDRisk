"""
Base patient data class with core fields shared by all CVD risk models.

This module provides the foundational data structure that all model-specific
data classes inherit from.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BasePatientData(BaseModel):
    """
    Base patient data with core fields required by all CVD risk models.

    This class contains the fundamental cardiovascular risk factors that
    are common across virtually all risk prediction models.

    Attributes
    ----------
    age : int
        Patient age in years. Valid range: 18-120 years.
    sex : Literal["male", "female"]
        Patient biological sex.
    systolic_bp : float
        Systolic blood pressure in mmHg. Valid range: 50-250 mmHg.
    total_cholesterol : float
        Total cholesterol in mmol/L. Valid range: 1.0-15.0 mmol/L.
    hdl_cholesterol : float
        HDL cholesterol in mmol/L. Valid range: 0.3-5.0 mmol/L.
    smoking : bool
        Current smoking status.

    Examples
    --------
    >>> patient = BasePatientData(
    ...     age=55,
    ...     sex="male",
    ...     systolic_bp=140.0,
    ...     total_cholesterol=6.0,
    ...     hdl_cholesterol=1.2,
    ...     smoking=True,
    ... )
    """

    age: int = Field(..., ge=18, le=120, description="Patient age in years")
    sex: Literal["male", "female"] = Field(..., description="Patient biological sex")
    systolic_bp: float = Field(
        ..., ge=50.0, le=250.0, description="Systolic BP in mmHg"
    )
    total_cholesterol: float = Field(
        ..., ge=1.0, le=15.0, description="Total cholesterol in mmol/L"
    )
    hdl_cholesterol: float = Field(
        ..., ge=0.3, le=5.0, description="HDL cholesterol in mmol/L"
    )
    smoking: bool = Field(..., description="Current smoking status")

    @field_validator("total_cholesterol")
    @classmethod
    def validate_cholesterol(cls, v: float) -> float:
        """Validate total cholesterol is reasonable."""
        if v < 1.0 or v > 15.0:
            raise ValueError(
                f"Total cholesterol {v} outside clinical range [1.0, 15.0] mmol/L"
            )
        return v

    @field_validator("hdl_cholesterol")
    @classmethod
    def validate_hdl(cls, v: float) -> float:
        """Validate HDL cholesterol is reasonable."""
        if v < 0.3 or v > 5.0:
            raise ValueError(
                f"HDL cholesterol {v} outside clinical range [0.3, 5.0] mmol/L"
            )
        return v

    model_config = ConfigDict(frozen=True, validate_assignment=True)
