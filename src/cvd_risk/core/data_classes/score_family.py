"""
Patient data classes for SCORE family models.

Includes data classes for SCORE, SCORE2, SCORE2-OP, SCORE2-Diabetes,
SCORE2-CKD, and SCORE2-Asia models.
"""

from typing import Literal, Optional

from pydantic import Field

from cvd_risk.core.data_classes.base import BasePatientData


class SCOREPatientData(BasePatientData):
    """
    Patient data for the original SCORE model.

    SCORE (Systematic COronary Risk Evaluation) estimates 10-year fatal CVD risk.

    Additional Fields
    -----------------
    region : Literal["low", "high"]
        CVD risk region (low or high risk countries).
    """

    region: Literal["low", "high"] = Field(
        ..., description="CVD risk region: low or high"
    )


class SCORE2PatientData(BasePatientData):
    """
    Patient data for SCORE2 model (ESC 2021).

    SCORE2 estimates 10-year risk of fatal and non-fatal CVD events.
    Valid for ages 40-69.

    Additional Fields
    -----------------
    region : Literal["low", "moderate", "high", "very_high"]
        CVD risk region based on ESC classification.
    """

    region: Literal["low", "moderate", "high", "very_high"] = Field(
        ..., description="CVD risk region: low, moderate, high, or very_high"
    )


class SCORE2OPPatientData(BasePatientData):
    """
    Patient data for SCORE2-OP model (Older Persons, age 70+).

    SCORE2-OP extends SCORE2 for patients aged 70 and older.

    Additional Fields
    -----------------
    region : Literal["low", "moderate", "high", "very_high"]
        CVD risk region based on ESC classification.
    """

    region: Literal["low", "moderate", "high", "very_high"] = Field(
        ..., description="CVD risk region: low, moderate, high, or very_high"
    )
    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")


class SCORE2DMPatientData(BasePatientData):
    """
    Patient data for SCORE2-Diabetes model.

    SCORE2-Diabetes estimates CVD risk specifically for patients with diabetes.

    Additional Fields
    -----------------
    region : Literal["low", "moderate", "high", "very_high"]
        CVD risk region based on ESC classification.
    diabetes_age : Optional[int]
        Age at diabetes diagnosis in years.
    hba1c : Optional[float]
        Glycated hemoglobin (HbA1c) in mmol/mol.
    """

    region: Literal["low", "moderate", "high", "very_high"] = Field(
        ..., description="CVD risk region: low, moderate, high, or very_high"
    )
    diabetes_age: Optional[int] = Field(
        default=None, ge=1, le=120, description="Age when diabetes was diagnosed"
    )
    hba1c: Optional[float] = Field(
        default=None, ge=10.0, le=200.0, description="HbA1c in mmol/mol"
    )
    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    egfr: Optional[float] = Field(
        default=None, ge=5.0, le=200.0, description="Estimated glomerular filtration rate (eGFR) in ml/min/1.73m²"
    )


class SCORE2CKDPatientData(BasePatientData):
    """
    Patient data for SCORE2-CKD model (Chronic Kidney Disease).

    SCORE2-CKD estimates CVD risk for patients with chronic kidney disease.

    Additional Fields
    -----------------
    region : Literal["low", "moderate", "high", "very_high"]
        CVD risk region based on ESC classification.
    egfr : float
        Estimated glomerular filtration rate in ml/min/1.73m².
    acr : Optional[float]
        Albumin-to-creatinine ratio in mg/g.
    """

    region: Literal["low", "moderate", "high", "very_high"] = Field(
        ..., description="CVD risk region: low, moderate, high, or very_high"
    )
    egfr: float = Field(
        ..., ge=5.0, le=200.0, description="eGFR in ml/min/1.73m²"
    )
    acr: Optional[float] = Field(
        default=None, ge=0.1, le=2000.0, description="ACR in mg/g"
    )
    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    proteinuria_trace: Optional[Literal["negative", "trace", "1+", "2+", "3+", "4+"]] = Field(
        default=None, description="Proteinuria dipstick test result"
    )


class SCORE2AsiaPatientData(BasePatientData):
    """
    Patient data for SCORE2-Asia model.

    SCORE2-Asia is calibrated for Asian populations, including CKD assessment.

    Additional Fields
    -----------------
    region : str
        Asian region identifier.
    egfr : Optional[float]
        Estimated glomerular filtration rate in ml/min/1.73m².
    acr : Optional[float]
        Albumin-to-creatinine ratio in mg/g.
    """

    region: str = Field(..., description="Asian region identifier")
    egfr: Optional[float] = Field(
        default=None, ge=5.0, le=200.0, description="eGFR in ml/min/1.73m²"
    )
    acr: Optional[float] = Field(
        default=None, ge=0.1, le=2000.0, description="ACR in mg/g"
    )
    diabetes: Optional[bool] = Field(default=None, description="Diabetes status")
    proteinuria_trace: Optional[Literal["negative", "trace", "1+", "2+", "3+", "4+"]] = Field(
        default=None, description="Proteinuria dipstick test result"
    )
