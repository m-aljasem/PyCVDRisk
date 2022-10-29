"""
SMART2 risk score for recurrent cardiovascular events.

SMART2 (Second Manifestations of ARTerial disease) estimates the risk
of recurrent cardiovascular events in patients with established
cardiovascular disease.

Reference:
    Dorresteijn JA, Visseren FL, Wassink AM, et al. Development and
    validation of a prediction rule for recurrent vascular events and
    death in patients with established cardiovascular disease.
    European Heart Journal. 2014;35(29):1925-31.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from cvd_risk_calculator.core.base import RiskModel
from cvd_risk_calculator.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# SMART2 coefficients (simplified)
_SMART2_COEFFICIENTS = {
    "age": 0.028,
    "male": 0.261,
    "systolic_bp": 0.008,
    "hdl_cholesterol": -0.458,
    "total_cholesterol": 0.152,
    "smoking": 0.371,
    "diabetes": 0.456,
    "constant": -5.432,
}

# Baseline survival at 10 years (approximation)
_SMART2_BASELINE_SURVIVAL = 0.85


class SMART2(RiskModel):
    """
    SMART2 risk model for recurrent cardiovascular events.

    SMART2 estimates the risk of recurrent cardiovascular events
    (myocardial infarction, stroke, vascular death) in patients
    with established cardiovascular disease.

    Parameters
    ----------
    age : int
        Age in years (typically 40-80).
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

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year risk of recurrent CVD events as percentage
        - risk_category: Risk classification
        - model_name: "SMART2"
        - model_version: Algorithm version identifier

    Examples
    --------
    >>> from cvd_risk_calculator.models.smart2 import SMART2
    >>> from cvd_risk_calculator.core.validation import PatientData
    >>>
    >>> patient = PatientData(
    ...     age=65,
    ...     sex="male",
    ...     systolic_bp=140.0,
    ...     total_cholesterol=5.5,
    ...     hdl_cholesterol=1.0,
    ...     smoking=False,
    ...     diabetes=True,
    ... )
    >>> model = SMART2()
    >>> result = model.calculate(patient)
    >>> print(f"10-year recurrent CVD risk: {result.risk_score:.1f}%")

    Notes
    -----
    - This model is for patients with established CVD
    - Risk estimates are for recurrent events, not primary prevention
    - Model validated in secondary prevention populations
    """

    model_name = "SMART2"
    model_version = "2014"
    supported_regions = None  # European model

    def __init__(self) -> None:
        """Initialize SMART2 model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """Validate patient input for SMART2."""
        super().validate_input(patient)

        if patient.diabetes is None:
            logger.warning(
                "Diabetes status not specified. Assuming no diabetes for calculation."
            )

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate 10-year risk of recurrent CVD events using SMART2."""
        self.validate_input(patient)

        coeffs = _SMART2_COEFFICIENTS

        # Calculate linear predictor
        linear_predictor = (
            coeffs["age"] * patient.age
            + coeffs["male"] * float(patient.sex == "male")
            + coeffs["systolic_bp"] * patient.systolic_bp
            + coeffs["hdl_cholesterol"] * patient.hdl_cholesterol
            + coeffs["total_cholesterol"] * patient.total_cholesterol
            + coeffs["smoking"] * float(patient.smoking)
            + coeffs["diabetes"] * float(patient.diabetes or False)
            + coeffs["constant"]
        )

        # Calculate risk
        risk_percentage = (
            (1.0 - (_SMART2_BASELINE_SURVIVAL ** np.exp(linear_predictor))) * 100.0
        )
        risk_percentage = np.clip(risk_percentage, 0.0, 100.0)

        risk_category = self._categorize_risk(risk_percentage)

        metadata = self._get_metadata()
        metadata.update(
            {
                "age": patient.age,
                "sex": patient.sex,
                "linear_predictor": linear_predictor,
                "diabetes": patient.diabetes,
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
        """Categorize risk for secondary prevention."""
        if risk_percentage < 20:
            return "low"
        elif risk_percentage < 30:
            return "moderate"
        else:
            return "high"

