"""
QRISK3 cardiovascular disease risk prediction model.

QRISK3 estimates 10-year risk of developing cardiovascular disease,
incorporating additional risk factors compared to earlier QRISK versions.

Reference:
    Hippisley-Cox J, Coupland C, Brindle P. Development and validation
    of QRISK3 risk prediction algorithms to estimate future risk of
    cardiovascular disease: prospective cohort study. BMJ. 2017;357:j2099.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from cvd_risk_calculator.core.base import RiskModel
from cvd_risk_calculator.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# Simplified QRISK3 coefficients (approximations)
_QRISK3_COEFFICIENTS = {
    "base": {
        "age": 0.0799,
        "age_squared": -0.0005,
        "total_cholesterol": 0.1909,
        "hdl_cholesterol": -0.2624,
        "systolic_bp": 0.0195,
        "smoking": 0.8807,
        "diabetes": 1.2226,
        "bmi": 0.0211,
        "family_history": 0.5113,
        "ethnicity_south_asian": 1.3256,
        "constant": -24.675,
    },
    "female": {
        "age_adjustment": 0.1767,
        "smoking_age": -0.0125,
    },
    "male": {
        "age_adjustment": 0.1538,
        "smoking_age": -0.0105,
    },
}

# Baseline survival at 10 years
_QRISK3_BASELINE_SURVIVAL = {"male": 0.9186, "female": 0.9653}


class QRISK3(RiskModel):
    """
    QRISK3 cardiovascular disease risk prediction model.

    QRISK3 estimates 10-year risk of cardiovascular disease with
    enhanced risk factors including ethnicity, BMI, and family history.

    Parameters
    ----------
    age : int
        Age in years (25-84 for optimal performance).
    sex : Literal["male", "female"]
        Biological sex.
    systolic_bp : float
        Systolic blood pressure in mmHg.
    total_cholesterol : float
        Total cholesterol in mmol/L.
    hdl_cholesterol : float
        HDL cholesterol in mmol/L.
    smoking : bool
        Current smoking status.
    diabetes : Optional[bool]
        Diabetes status.
    bmi : Optional[float]
        Body mass index in kg/m².
    family_history : Optional[bool]
        Family history of CVD in first-degree relative before age 60.
    ethnicity : Optional[Literal["white", "south_asian", "black", "chinese", "mixed", "other"]]
        Ethnicity category.

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year CVD risk as percentage
        - risk_category: Risk classification
        - model_name: "QRISK3"
        - model_version: Algorithm version identifier

    Examples
    --------
    >>> from cvd_risk_calculator.models.qrisk3 import QRISK3
    >>> from cvd_risk_calculator.core.validation import PatientData
    >>>
    >>> patient = PatientData(
    ...     age=55,
    ...     sex="male",
    ...     systolic_bp=140.0,
    ...     total_cholesterol=6.0,
    ...     hdl_cholesterol=1.2,
    ...     smoking=True,
    ...     diabetes=False,
    ...     bmi=25.5,
    ...     family_history=True,
    ...     ethnicity="white",
    ... )
    >>> model = QRISK3()
    >>> result = model.calculate(patient)
    >>> print(f"10-year CVD risk: {result.risk_score:.1f}%")

    Notes
    -----
    - Model validated for ages 25-84 years
    - Includes ethnicity-specific adjustments
    - Requires additional factors (BMI, family history) for optimal accuracy
    """

    model_name = "QRISK3"
    model_version = "2017"
    supported_regions = None  # UK-based but widely applicable

    def __init__(self) -> None:
        """Initialize QRISK3 model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """Validate patient input for QRISK3."""
        super().validate_input(patient)

        if patient.age < 25 or patient.age > 84:
            logger.warning(
                f"Age {patient.age} outside optimal range [25, 84] years. "
                "Results may have reduced accuracy."
            )

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate 10-year CVD risk using QRISK3."""
        self.validate_input(patient)

        coeffs = _QRISK3_COEFFICIENTS["base"]
        sex_coeffs = _QRISK3_COEFFICIENTS[patient.sex]
        baseline_surv = _QRISK3_BASELINE_SURVIVAL[patient.sex]

        # Calculate linear predictor
        linear_predictor = (
            coeffs["age"] * patient.age
            + coeffs["age_squared"] * patient.age * patient.age
            + coeffs["total_cholesterol"] * patient.total_cholesterol
            + coeffs["hdl_cholesterol"] * patient.hdl_cholesterol
            + coeffs["systolic_bp"] * patient.systolic_bp
            + coeffs["smoking"] * float(patient.smoking)
            + sex_coeffs["smoking_age"] * float(patient.smoking) * patient.age
            + coeffs["diabetes"] * float(patient.diabetes or False)
            + coeffs["bmi"] * (patient.bmi or 25.0)  # Default BMI
            + coeffs["family_history"] * float(patient.family_history or False)
            + coeffs["constant"]
        )

        # Ethnicity adjustment
        if patient.ethnicity == "south_asian":
            linear_predictor += coeffs["ethnicity_south_asian"]

        # Calculate risk
        risk_percentage = (1.0 - (baseline_surv ** np.exp(linear_predictor))) * 100.0
        risk_percentage = np.clip(risk_percentage, 0.0, 100.0)

        risk_category = self._categorize_risk(risk_percentage)

        metadata = self._get_metadata()
        metadata.update(
            {
                "age": patient.age,
                "sex": patient.sex,
                "linear_predictor": linear_predictor,
                "bmi": patient.bmi,
            }
        )

        return RiskResult(
            risk_score=float(risk_percentage),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata=metadata,
        )

    def _categorize_risk(self, risk_percentage: float) -> str:
        """Categorize risk."""
        if risk_percentage < 10:
            return "low"
        elif risk_percentage < 20:
            return "moderate"
        else:
            return "high"

