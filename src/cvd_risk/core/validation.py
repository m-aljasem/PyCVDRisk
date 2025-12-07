"""Input validation models for patient data using Pydantic."""

from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientData(BaseModel):
    """
    Validated patient data structure for CVD risk calculation.

    This model ensures type safety and value validation for patient
    inputs across all risk models. All values are validated against
    clinical ranges.

    Attributes
    ----------
    age : int
        Patient age in years. Typical range: 40-100 years.
    sex : Literal["male", "female"]
        Patient biological sex.
    systolic_bp : float
        Systolic blood pressure in mmHg. Typical range: 90-250 mmHg.
    total_cholesterol : float
        Total cholesterol in mmol/L. Typical range: 2.5-10.0 mmol/L.
    hdl_cholesterol : float
        HDL cholesterol in mmol/L. Typical range: 0.5-3.5 mmol/L.
    smoking : bool
        Current smoking status.
    region : Optional[Literal["low", "moderate", "high", "very_high"]]
        CVD risk region (for SCORE2). Options: low, moderate, high, very_high.
        None if not applicable to the model.
    diabetes : Optional[bool]
        Diabetes status. None if not applicable to the model.
    ethnicity : Optional[Literal["white", "south_asian", "black", "chinese", "mixed", "other"]]
        Ethnicity (for QRISK3). None if not applicable.
    bmi : Optional[float]
        Body mass index in kg/m². None if not applicable.
    family_history : Optional[bool]
        Family history of CVD. None if not applicable.
    antihypertensive : Optional[bool]
        Use of antihypertensive medication. None if not applicable.

    Examples
    --------
    >>> patient = PatientData(
    ...     age=55,
    ...     sex="male",
    ...     systolic_bp=140.0,
    ...     total_cholesterol=6.0,
    ...     hdl_cholesterol=1.2,
    ...     smoking=True,
    ...     region="moderate"
    ... )
    """

    age: int = Field(..., ge=18, le=120, description="Patient age in years")
    sex: Literal["male", "female"] = Field(..., description="Patient biological sex")
    systolic_bp: float = Field(..., ge=50.0, le=250.0, description="Systolic BP in mmHg")
    total_cholesterol: float = Field(
        ..., ge=1.0, le=15.0, description="Total cholesterol in mmol/L"
    )
    hdl_cholesterol: float = Field(
        ..., ge=0.3, le=5.0, description="HDL cholesterol in mmol/L"
    )
    smoking: bool = Field(..., description="Current smoking status")
    region: Optional[Literal["low", "moderate", "high", "very_high"]] = Field(
        default=None, description="CVD risk region for SCORE2"
    )
    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    ethnicity: Optional[
        Literal["white", "south_asian", "black", "chinese", "mixed", "other"]
    ] = Field(default=None, description="Ethnicity for QRISK3")
    bmi: Optional[float] = Field(
        default=None, ge=10.0, le=60.0, description="Body mass index in kg/m²"
    )
    family_history: Optional[bool] = Field(
        default=None, description="Family history of CVD"
    )
    antihypertensive: Optional[bool] = Field(
        default=None, description="Use of antihypertensive medication"
    )

    @field_validator("total_cholesterol")
    @classmethod
    def validate_cholesterol(cls, v: float) -> float:
        """Validate total cholesterol is reasonable."""
        if v < 1.0 or v > 15.0:
            raise ValueError(f"Total cholesterol {v} outside clinical range [1.0, 15.0] mmol/L")
        return v

    @field_validator("hdl_cholesterol")
    @classmethod
    def validate_hdl(cls, v: float) -> float:
        """Validate HDL cholesterol is reasonable."""
        if v < 0.3 or v > 5.0:
            raise ValueError(f"HDL cholesterol {v} outside clinical range [0.3, 5.0] mmol/L")
        return v

    model_config = ConfigDict(frozen=True, validate_assignment=True)


class RiskResult(BaseModel):
    """
    Risk calculation result with metadata.

    Attributes
    ----------
    risk_score : float
        Calculated risk score (typically percentage).
    risk_category : str
        Categorical risk classification (e.g., "low", "moderate", "high").
    confidence_interval_lower : Optional[float]
        Lower bound of 95% confidence interval if available.
    confidence_interval_upper : Optional[float]
        Upper bound of 95% confidence interval if available.
    model_name : str
        Name of the risk model used.
    model_version : str
        Version identifier of the model algorithm.
    calculation_metadata : dict
        Additional metadata about the calculation.
    """

    risk_score: float = Field(..., description="Calculated risk score (percentage)")
    risk_category: str = Field(..., description="Risk category classification")
    confidence_interval_lower: Optional[float] = Field(
        default=None, description="Lower 95% CI bound"
    )
    confidence_interval_upper: Optional[float] = Field(
        default=None, description="Upper 95% CI bound"
    )
    model_name: str = Field(..., description="Name of risk model")
    model_version: str = Field(..., description="Model version/algorithm identifier")
    calculation_metadata: dict = Field(
        default_factory=dict, description="Additional calculation metadata"
    )

    model_config = ConfigDict(frozen=True)


def validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """
    Validate pandas DataFrame has required columns for risk calculation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with patient data.
    required_columns : list[str]
        List of required column names.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df

