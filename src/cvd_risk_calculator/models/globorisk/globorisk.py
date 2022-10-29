"""
Globorisk cardiovascular disease risk prediction model.

Globorisk provides country-specific cardiovascular disease risk
estimates that account for local mortality rates.

Reference:
    Ueda P, Woodward M, Lu Y, et al. Laboratory-based and
    office-based risk scores and charts to predict 10-year risk
    of cardiovascular disease in 182 countries: a pooled analysis
    of prospective cohorts and health surveys. Lancet Diabetes
    Endocrinol. 2017;5(3):196-213.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from cvd_risk_calculator.core.base import RiskModel
from cvd_risk_calculator.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# Simplified Globorisk coefficients (country-specific in full model)
_GLOBORISK_COEFFICIENTS = {
    "age": 0.065,
    "male": 0.241,
    "total_cholesterol": 0.184,
    "hdl_cholesterol": -0.231,
    "systolic_bp": 0.020,
    "smoking": 0.512,
    "diabetes": 0.445,
    "constant": -4.623,
}

# Baseline survival (country-specific in full model)
_GLOBORISK_BASELINE_SURVIVAL = 0.92


class Globorisk(RiskModel):
    """
    Globorisk cardiovascular disease risk prediction model.

    Globorisk provides country-specific 10-year CVD risk estimates
    that account for local cardiovascular mortality rates.

    Parameters
    ----------
    age : int
        Age in years (40-74 for optimal performance).
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
        - risk_score: 10-year CVD risk as percentage
        - risk_category: Risk classification
        - model_name: "Globorisk"
        - model_version: Algorithm version identifier

    Examples
    --------
    >>> from cvd_risk_calculator.models.globorisk import Globorisk
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
    ... )
    >>> model = Globorisk()
    >>> result = model.calculate(patient)
    >>> print(f"10-year CVD risk: {result.risk_score:.1f}%")

    Notes
    -----
    - Model validated for ages 40-74 years
    - Country-specific coefficients available for 182 countries
    - This is a simplified implementation
    """

    model_name = "Globorisk"
    model_version = "2017"
    supported_regions = None  # 182 countries (simplified)

    def __init__(self) -> None:
        """Initialize Globorisk model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """Validate patient input for Globorisk."""
        super().validate_input(patient)

        if patient.age < 40 or patient.age > 74:
            logger.warning(
                f"Age {patient.age} outside optimal range [40, 74] years. "
                "Results may have reduced accuracy."
            )

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate 10-year CVD risk using Globorisk."""
        self.validate_input(patient)

        coeffs = _GLOBORISK_COEFFICIENTS

        # Calculate linear predictor
        linear_predictor = (
            coeffs["age"] * patient.age
            + coeffs["male"] * float(patient.sex == "male")
            + coeffs["total_cholesterol"] * patient.total_cholesterol
            + coeffs["hdl_cholesterol"] * patient.hdl_cholesterol
            + coeffs["systolic_bp"] * patient.systolic_bp
            + coeffs["smoking"] * float(patient.smoking)
            + coeffs["diabetes"] * float(patient.diabetes or False)
            + coeffs["constant"]
        )

        # Calculate risk
        risk_percentage = (
            (1.0 - (_GLOBORISK_BASELINE_SURVIVAL ** np.exp(linear_predictor))) * 100.0
        )
        risk_percentage = np.clip(risk_percentage, 0.0, 100.0)

        risk_category = self._categorize_risk(risk_percentage)

        metadata = self._get_metadata()
        metadata.update(
            {
                "age": patient.age,
                "sex": patient.sex,
                "linear_predictor": linear_predictor,
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

