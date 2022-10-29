"""
WHO CVD Risk Chart for cardiovascular disease risk prediction.

The WHO CVD Risk Charts provide region-specific risk estimates based
on simplified risk factor profiles.

Reference:
    WHO CVD Risk Chart Working Group. World Health Organization
    cardiovascular disease risk charts: revised models to estimate
    risk in 21 global regions. Lancet Glob Health. 2019;7(10):e1332-e1345.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from cvd_risk_calculator.core.base import RiskModel
from cvd_risk_calculator.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# Simplified WHO risk charts (region-specific approximations)
_WHO_COEFFICIENTS = {
    "age": 0.063,
    "sex_female": -0.088,
    "total_cholesterol": 0.167,
    "hdl_cholesterol": -0.212,
    "systolic_bp": 0.018,
    "smoking": 0.481,
    "diabetes": 0.412,
    "constant": -3.845,
}

# Baseline survival by region (simplified - uses moderate as default)
_WHO_BASELINE_SURVIVAL = 0.95


class WHO(RiskModel):
    """
    WHO CVD Risk Chart for cardiovascular disease risk prediction.

    The WHO model provides region-specific 10-year CVD risk estimates
    based on age, sex, and major cardiovascular risk factors.

    Parameters
    ----------
    age : int
        Age in years (40-80 for optimal performance).
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
    region : Optional[str]
        WHO region (currently simplified implementation).

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year CVD risk as percentage
        - risk_category: Risk classification
        - model_name: "WHO"
        - model_version: Algorithm version identifier

    Examples
    --------
    >>> from cvd_risk_calculator.models.who import WHO
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
    >>> model = WHO()
    >>> result = model.calculate(patient)
    >>> print(f"10-year CVD risk: {result.risk_score:.1f}%")

    Notes
    -----
    - Model validated for ages 40-80 years
    - Region-specific coefficients available for 21 global regions
    - This is a simplified implementation
    """

    model_name = "WHO"
    model_version = "2019"
    supported_regions = None  # 21 global regions (simplified)

    def __init__(self) -> None:
        """Initialize WHO model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """Validate patient input for WHO model."""
        super().validate_input(patient)

        if patient.age < 40 or patient.age > 80:
            logger.warning(
                f"Age {patient.age} outside optimal range [40, 80] years. "
                "Results may have reduced accuracy."
            )

    def calculate(self, patient: PatientData) -> RiskResult:
        """Calculate 10-year CVD risk using WHO model."""
        self.validate_input(patient)

        coeffs = _WHO_COEFFICIENTS

        # Calculate linear predictor
        linear_predictor = (
            coeffs["age"] * patient.age
            + coeffs["sex_female"] * float(patient.sex == "female")
            + coeffs["total_cholesterol"] * patient.total_cholesterol
            + coeffs["hdl_cholesterol"] * patient.hdl_cholesterol
            + coeffs["systolic_bp"] * patient.systolic_bp
            + coeffs["smoking"] * float(patient.smoking)
            + coeffs["diabetes"] * float(patient.diabetes or False)
            + coeffs["constant"]
        )

        # Calculate risk
        risk_percentage = (
            (1.0 - (_WHO_BASELINE_SURVIVAL ** np.exp(linear_predictor))) * 100.0
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
        """Categorize risk using WHO categories."""
        if risk_percentage < 10:
            return "low"
        elif risk_percentage < 20:
            return "moderate"
        else:
            return "high"

