"""
ASCVD (Atherosclerotic Cardiovascular Disease) Pooled Cohort Equations.

The ASCVD Pooled Cohort Equations estimate 10-year risk of atherosclerotic
cardiovascular disease for primary prevention.

Reference:
    Goff DC Jr, Lloyd-Jones DM, Bennett G, et al. 2013 ACC/AHA Guideline
    on the Assessment of Cardiovascular Risk: A Report of the American
    College of Cardiology/American Heart Association Task Force on
    Practice Guidelines. Circulation. 2014;129(25 Suppl 2):S49-73.

Note: This is a simplified implementation. The full ASCVD model includes
race-specific equations for African American and non-African American populations.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# ASCVD coefficients (simplified, for white/non-African American population)
_ASCVD_COEFFICIENTS = {
    "male_white": {
        "age_log": 3.06117,
        "age_log_sq": -1.12370,
        "ln_chol": 1.93303,
        "age_ln_chol": -0.52551,
        "ln_hdl": -1.47906,
        "age_ln_hdl": 0.31583,
        "ln_age_sbp": 0.93338,
        "ln_age_sbp_treated": 1.99881,
        "smoking": 0.65451,
        "smoking_age": -0.57763,
        "constant": -23.9802,
    },
    "female_white": {
        "age_log": 2.32888,
        "age_log_sq": -1.12370,
        "ln_chol": 1.20904,
        "age_ln_chol": -0.30784,
        "ln_hdl": -0.70833,
        "age_ln_hdl": 0.39163,
        "ln_age_sbp": 1.34065,
        "ln_age_sbp_treated": 2.00168,
        "smoking": 0.52873,
        "smoking_age": -0.48660,
        "constant": -26.1931,
    },
}

# Baseline survival probabilities at 10 years
_ASCVD_BASELINE_SURVIVAL = {"male": 0.91436, "female": 0.96652}


class ASCVD(RiskModel):
    """
    ASCVD Pooled Cohort Equations for 10-year CVD risk prediction.

    The ASCVD model estimates 10-year risk of atherosclerotic cardiovascular
    disease using the Pooled Cohort Equations from the 2013 ACC/AHA guidelines.

    Parameters
    ----------
    age : int
        Age in years (40-79 for optimal performance).
    sex : Literal["male", "female"]
        Biological sex.
    systolic_bp : float
        Systolic blood pressure in mmHg.
    total_cholesterol : float
        Total cholesterol in mg/dL (converted from mmol/L internally).
    hdl_cholesterol : float
        HDL cholesterol in mg/dL (converted from mmol/L internally).
    smoking : bool
        Current smoking status.
    antihypertensive : Optional[bool]
        Use of antihypertensive medication (default: False).

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year ASCVD risk as percentage
        - risk_category: Risk classification
        - model_name: "ASCVD"
        - model_version: Algorithm version identifier

    Examples
    --------
    >>> from cvd_risk.models.ascvd import ASCVD
    >>> from cvd_risk.core.validation import PatientData
    >>>
    >>> patient = PatientData(
    ...     age=55,
    ...     sex="male",
    ...     systolic_bp=140.0,
    ...     total_cholesterol=6.0,
    ...     hdl_cholesterol=1.2,
    ...     smoking=True,
    ...     antihypertensive=False,
    ... )
    >>> model = ASCVD()
    >>> result = model.calculate(patient)
    >>> print(f"10-year ASCVD risk: {result.risk_score:.1f}%")

    Notes
    -----
    - Model validated for ages 40-79 years
    - This implementation uses white/non-African American coefficients
    - Full model includes race-specific equations
    - Risk estimates are for 10-year period
    """

    model_name = "ASCVD"
    model_version = "2013"
    supported_regions = None  # US-based model

    def __init__(self) -> None:
        """Initialize ASCVD model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for ASCVD requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for ASCVD calculation.
        """
        super().validate_input(patient)

        if patient.age < 40 or patient.age > 79:
            logger.warning(
                f"Age {patient.age} outside optimal range [40, 79] years. "
                "Results may have reduced accuracy."
            )

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate 10-year ASCVD risk using Pooled Cohort Equations.

        Parameters
        ----------
        patient : PatientData
            Validated patient data.

        Returns
        -------
        RiskResult
            Risk calculation result with metadata.
        """
        self.validate_input(patient)

        # Convert cholesterol from mmol/L to mg/dL
        total_chol_mgdl = patient.total_cholesterol * 38.67
        hdl_mgdl = patient.hdl_cholesterol * 38.67

        # Determine coefficient set
        sex_key = f"{patient.sex}_white"  # Simplified: use white coefficients
        coeffs = _ASCVD_COEFFICIENTS[sex_key]
        baseline_surv = _ASCVD_BASELINE_SURVIVAL[patient.sex]

        # Calculate linear predictor
        ln_age = np.log(patient.age)
        ln_chol = np.log(total_chol_mgdl)
        ln_hdl = np.log(hdl_mgdl)
        ln_sbp = np.log(patient.systolic_bp)

        # Check if on antihypertensive treatment
        treated = patient.antihypertensive if patient.antihypertensive is not None else False

        linear_predictor = (
            coeffs["age_log"] * ln_age
            + coeffs["age_log_sq"] * ln_age * ln_age
            + coeffs["ln_chol"] * ln_chol
            + coeffs["age_ln_chol"] * ln_age * ln_chol
            + coeffs["ln_hdl"] * ln_hdl
            + coeffs["age_ln_hdl"] * ln_age * ln_hdl
            + (coeffs["ln_age_sbp_treated"] if treated else coeffs["ln_age_sbp"])
            * ln_age
            * ln_sbp
            + coeffs["smoking"] * float(patient.smoking)
            + coeffs["smoking_age"] * float(patient.smoking) * ln_age
            + coeffs["constant"]
        )

        # Calculate risk
        risk_percentage = (1.0 - (baseline_surv ** np.exp(linear_predictor))) * 100.0
        risk_percentage = np.clip(risk_percentage, 0.0, 100.0)

        # Categorize risk (ASCVD categories)
        risk_category = self._categorize_risk(risk_percentage)

        metadata = self._get_metadata()
        metadata.update(
            {
                "age": patient.age,
                "sex": patient.sex,
                "linear_predictor": linear_predictor,
                "treated": treated,
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
        """Categorize risk using ASCVD categories."""
        if risk_percentage < 5:
            return "low"
        elif risk_percentage < 7.5:
            return "borderline"
        elif risk_percentage < 20:
            return "intermediate"
        else:
            return "high"

