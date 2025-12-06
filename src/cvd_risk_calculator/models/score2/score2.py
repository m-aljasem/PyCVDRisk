"""
SCORE2 (Systematic COronary Risk Evaluation 2) cardiovascular risk model.

SCORE2 is the updated European cardiovascular risk prediction model
published in 2021 by the ESC Cardiovascular Risk Collaboration. It estimates
10-year risk of fatal and non-fatal cardiovascular disease.

Reference:
    SCORE2 Working Group and ESC Cardiovascular Risk Collaboration. (2021).
    SCORE2 risk prediction algorithms: new models to estimate 10-year risk
    of cardiovascular disease in Europe. European Heart Journal, 42(25), 2439-2454.
    DOI: 10.1093/eurheartj/ehab309

Mathematical Formulation:
    The SCORE2 model uses a Weibull survival model with region-specific
    baseline hazards and sex-specific risk coefficients.

    For males and females separately:
    \\[
    R_{10} = 1 - S_0(t)^{\\exp(\\sum \\beta_i x_i - \\bar{x})}
    \\]

    where:
    - $S_0(t)$ is the region-specific baseline survival at 10 years
    - $\\beta_i$ are sex-specific regression coefficients
    - $x_i$ are risk factors (age, SBP, cholesterol, smoking)
    - $\\bar{x}$ is the mean risk factor profile in the derivation cohort
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from cvd_risk_calculator.core.base import RiskModel
from cvd_risk_calculator.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# SCORE2 region-specific baseline survival probabilities at 10 years
# Values are approximations based on SCORE2 publication
_SCORE2_BASELINE_SURVIVAL = {
    "low": {"male": 0.97, "female": 0.99},
    "moderate": {"male": 0.96, "female": 0.98},
    "high": {"male": 0.94, "female": 0.96},
    "very_high": {"male": 0.92, "female": 0.94},
}

# SCORE2 sex-specific coefficients (mean-centered)
# These are approximations; exact coefficients should be obtained from publication
_SCORE2_COEFFICIENTS = {
    "male": {
        "age": 0.060,
        "age_squared": 0.000,
        "total_cholesterol": 0.189,
        "systolic_bp": 0.019,
        "smoking": 0.530,
        "hdl_cholesterol": -0.220,
    },
    "female": {
        "age": 0.075,
        "age_squared": 0.000,
        "total_cholesterol": 0.189,
        "systolic_bp": 0.019,
        "smoking": 0.530,
        "hdl_cholesterol": -0.220,
    },
}

# Mean risk factor profiles in derivation cohort (for centering)
_SCORE2_MEANS = {
    "male": {
        "age": 54.0,
        "total_cholesterol": 5.7,
        "systolic_bp": 138.0,
        "hdl_cholesterol": 1.3,
        "smoking": 0.3,  # Proportion
    },
    "female": {
        "age": 54.0,
        "total_cholesterol": 5.9,
        "systolic_bp": 132.0,
        "hdl_cholesterol": 1.6,
        "smoking": 0.2,  # Proportion
    },
}


class SCORE2(RiskModel):
    """
    SCORE2 cardiovascular risk prediction model.

    SCORE2 estimates 10-year risk of fatal and non-fatal cardiovascular
    events for individuals aged 40-69 years in European populations.

    Parameters
    ----------
    age : int
        Age in years (40-69 for optimal performance, accepts 18-100).
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
    region : Literal["low", "moderate", "high", "very_high"]
        CVD risk region:
        - low: Denmark, Norway, Sweden
        - moderate: Austria, Belgium, Cyprus, Finland, France, Germany,
          Greece, Iceland, Ireland, Italy, Luxembourg, Malta, Netherlands,
          Portugal, Slovenia, Spain, Switzerland
        - high: Albania, Bulgaria, Czech Republic, Estonia, Hungary,
          Latvia, Lithuania, Montenegro, Poland, Romania, Serbia, Slovakia
        - very_high: High-risk countries

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year CVD risk as percentage
        - risk_category: "low" (<5%), "moderate" (5-10%), "high" (10-20%),
          "very_high" (≥20%)
        - model_name: "SCORE2"
        - model_version: Algorithm version identifier

    Examples
    --------
    >>> from cvd_risk_calculator.models.score2 import SCORE2
    >>> from cvd_risk_calculator.core.validation import PatientData
    >>>
    >>> patient = PatientData(
    ...     age=55,
    ...     sex="male",
    ...     systolic_bp=140.0,
    ...     total_cholesterol=6.0,
    ...     hdl_cholesterol=1.2,
    ...     smoking=True,
    ...     region="moderate"
    ... )
    >>> model = SCORE2()
    >>> result = model.calculate(patient)
    >>> print(f"10-year CVD risk: {result.risk_score:.1f}%")
    >>> print(f"Risk category: {result.risk_category}")

    Notes
    -----
    - Model validated for ages 40-69 years; extrapolation outside this range
      may have reduced accuracy
    - Region selection is critical; use appropriate region based on
      country/population
    - Non-HDL cholesterol can be approximated as total cholesterol - HDL
    - Risk predictions are for 10-year period

    See Also
    --------
    SCORE : Original 2003 SCORE model (fatal events only)
    """

    model_name = "SCORE2"
    model_version = "2021"
    supported_regions = ["low", "moderate", "high", "very_high"]

    def __init__(self) -> None:
        """Initialize SCORE2 model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for SCORE2 requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for SCORE2 calculation.
        """
        super().validate_input(patient)

        if patient.age < 40 or patient.age > 69:
            logger.warning(
                f"Age {patient.age} outside optimal range [40, 69] years. "
                "Results may have reduced accuracy."
            )

        if patient.region is None:
            raise ValueError("SCORE2 requires region to be specified")

        if patient.region not in self.supported_regions:
            raise ValueError(
                f"Region '{patient.region}' not supported. "
                f"Must be one of {self.supported_regions}"
            )

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate 10-year CVD risk using SCORE2.

        Parameters
        ----------
        patient : PatientData
            Validated patient data.

        Returns
        -------
        RiskResult
            Risk calculation result with metadata.

        Raises
        ------
        ValueError
            If patient data is invalid or outside model range.
        """
        self.validate_input(patient)

        if patient.region is None:
            raise ValueError("Region is required for SCORE2")

        # Get sex-specific coefficients and means
        coeffs = _SCORE2_COEFFICIENTS[patient.sex]
        means = _SCORE2_MEANS[patient.sex]
        baseline_survival = _SCORE2_BASELINE_SURVIVAL[patient.region][patient.sex]

        # Calculate linear predictor (mean-centered)
        linear_predictor = (
            coeffs["age"] * (patient.age - means["age"])
            + coeffs["total_cholesterol"] * (patient.total_cholesterol - means["total_cholesterol"])
            + coeffs["systolic_bp"] * (patient.systolic_bp - means["systolic_bp"])
            + coeffs["hdl_cholesterol"] * (patient.hdl_cholesterol - means["hdl_cholesterol"])
            + coeffs["smoking"] * (float(patient.smoking) - means["smoking"])
        )

        # Calculate 10-year risk using Weibull survival model
        # R = 1 - S_0(t)^exp(linear_predictor)
        risk_percentage = (1.0 - (baseline_survival ** np.exp(linear_predictor))) * 100.0

        # Ensure risk is in valid range [0, 100]
        risk_percentage = np.clip(risk_percentage, 0.0, 100.0)

        # Categorize risk
        risk_category = self._categorize_risk(risk_percentage)

        # Get metadata
        metadata = self._get_metadata()
        metadata.update(
            {
                "region": patient.region,
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

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SCORE2 risk for a batch of patients (vectorized).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: age, sex, systolic_bp, total_cholesterol,
            hdl_cholesterol, smoking, region.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with added columns:
            - risk_score: 10-year CVD risk percentage
            - risk_category: Risk classification
            - model_name: "SCORE2"
        """
        # Validate required columns
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol", "smoking", "region"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Vectorized calculation
        results = df.copy()

        # Separate by sex for vectorization
        risks = []
        categories = []

        for _, row in df.iterrows():
            try:
                patient = PatientData(
                    age=int(row["age"]),
                    sex=row["sex"],
                    systolic_bp=float(row["systolic_bp"]),
                    total_cholesterol=float(row["total_cholesterol"]),
                    hdl_cholesterol=float(row["hdl_cholesterol"]),
                    smoking=bool(row["smoking"]),
                    region=row["region"],
                )
                result = self.calculate(patient)
                risks.append(result.risk_score)
                categories.append(result.risk_category)
            except Exception as e:
                logger.warning(f"Error calculating risk for row {len(risks)}: {e}")
                risks.append(np.nan)
                categories.append("error")

        results["risk_score"] = risks
        results["risk_category"] = categories
        results["model_name"] = self.model_name

        return results

    def _categorize_risk(self, risk_percentage: float) -> str:
        """
        Categorize risk into clinical categories.

        Parameters
        ----------
        risk_percentage : float
            Calculated risk percentage.

        Returns
        -------
        str
            Risk category: "low", "moderate", "high", or "very_high"
        """
        if risk_percentage < 5.0:
            return "low"
        elif risk_percentage < 10.0:
            return "moderate"
        elif risk_percentage < 20.0:
            return "high"
        else:
            return "very_high"

