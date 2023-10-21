"""
Progetto CUORE Risk Score.

Progetto CUORE is the official cardiovascular risk assessment tool in Italy,
developed from Italian population-based studies. It estimates 10-year risk
of coronary heart disease and cerebrovascular disease.

Reference:
    Palmieri L, Panico S, Vanuzzo D, et al. Evaluation of the global
    cardiovascular absolute risk: the Progetto CUORE individual score.
    Italian Heart Journal Supplement. 2004;5(9):820-828.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# Progetto CUORE coefficients (adapted from Italian population data)
# From Palmieri et al. 2004
_PROGETTO_CUORE_COEFFICIENTS_MEN = {
    "age": 0.048,
    "systolic_bp": 0.011,
    "total_cholesterol": 0.009,
    "hdl_cholesterol": -0.012,
    "smoking": 0.405,
    "diabetes": 0.525,
    "bmi": 0.020,
    "constant": -8.935
}

_PROGETTO_CUORE_COEFFICIENTS_WOMEN = {
    "age": 0.053,
    "systolic_bp": 0.013,
    "total_cholesterol": 0.011,
    "hdl_cholesterol": -0.014,
    "smoking": 0.365,
    "diabetes": 0.485,
    "bmi": 0.025,
    "constant": -9.248
}


class ProgettoCUORE(RiskModel):
    """
    Progetto CUORE Risk Score for cardiovascular disease prediction.

    Progetto CUORE is Italy's official cardiovascular risk assessment tool,
    developed from Italian population studies for primary prevention.

    Parameters
    ----------
    age : int
        Age in years (35-69 for optimal performance).
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
        Diabetes mellitus.
    bmi : Optional[float]
        Body mass index in kg/m².

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year CVD risk as percentage
        - risk_category: Risk classification
        - model_name: "ProgettoCUORE"
        - model_version: "2004"
    """

    model_name = "ProgettoCUORE"
    model_version = "2004"
    supported_regions = ["Italy", "Europe"]  # Italian population

    def __init__(self) -> None:
        """Initialize Progetto CUORE model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for Progetto CUORE requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for Progetto CUORE calculation.
        """
        super().validate_input(patient)

        if patient.age < 35 or patient.age > 69:
            logger.warning(
                f"Age {patient.age} outside optimal range [35, 69] years. "
                "Results may have reduced accuracy."
            )

        # Progetto CUORE requires cholesterol measurements
        if patient.total_cholesterol is None:
            raise ValueError("Progetto CUORE requires total cholesterol measurement")

        if patient.hdl_cholesterol is None:
            raise ValueError("Progetto CUORE requires HDL cholesterol measurement")

        if patient.systolic_bp is None:
            raise ValueError("Progetto CUORE requires systolic blood pressure measurement")

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate 10-year CVD risk using Progetto CUORE score.

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

        # Get appropriate coefficients
        coeffs = _PROGETTO_CUORE_COEFFICIENTS_MEN if patient.sex == "male" else _PROGETTO_CUORE_COEFFICIENTS_WOMEN

        # Get BMI (use default if not provided)
        bmi = getattr(patient, 'bmi', 25.0)  # Default BMI 25

        # Calculate linear predictor
        linear_predictor = (
            coeffs["age"] * patient.age
            + coeffs["systolic_bp"] * patient.systolic_bp
            + coeffs["total_cholesterol"] * patient.total_cholesterol
            + coeffs["hdl_cholesterol"] * patient.hdl_cholesterol
            + coeffs["smoking"] * float(patient.smoking)
            + coeffs["diabetes"] * float(patient.diabetes if patient.diabetes is not None else False)
            + coeffs["bmi"] * bmi
            + coeffs["constant"]
        )

        # Convert to risk probability using logistic function
        risk_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        risk_percentage = risk_probability * 100.0

        # Ensure risk is within reasonable bounds
        risk_percentage = np.clip(risk_percentage, 0.0, 40.0)

        # Categorize risk
        risk_category = self._categorize_risk(risk_percentage)

        metadata = self._get_metadata()
        metadata.update({
            "age": patient.age,
            "sex": patient.sex,
            "smoking": patient.smoking,
            "systolic_bp": patient.systolic_bp,
            "total_cholesterol": patient.total_cholesterol,
            "hdl_cholesterol": patient.hdl_cholesterol,
            "diabetes": patient.diabetes,
            "bmi": bmi,
            "linear_predictor": linear_predictor,
        })

        return RiskResult(
            risk_score=float(risk_percentage),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata=metadata,
        )

    def _categorize_risk(self, risk_percentage: float) -> str:
        """Categorize risk using Progetto CUORE categories."""
        if risk_percentage < 3:
            return "low"
        elif risk_percentage < 5:
            return "low_moderate"
        elif risk_percentage < 10:
            return "moderate"
        elif risk_percentage < 20:
            return "high"
        else:
            return "very_high"

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        """
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol", "smoking"]
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing required columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        # Fill missing optional columns with defaults
        defaults = {
            "diabetes": False,
            "bmi": 25.0
        }

        for col, default in defaults.items():
            if col not in results.columns:
                results[col] = default

        # Vectorized calculations
        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any():
                continue

            coeffs = _PROGETTO_CUORE_COEFFICIENTS_MEN if sex == 'male' else _PROGETTO_CUORE_COEFFICIENTS_WOMEN

            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"]
                + coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"]
                + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"]
                + coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"]
                + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float)
                + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float)
                + coeffs["bmi"] * results.loc[mask, "bmi"]
                + coeffs["constant"]
            )

            # Convert to probability
            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        # Clip risks
        results["risk_score"] = np.clip(results["risk_score"], 0.0, 40.0)

        # Vectorized categorization
        risk_scores = results["risk_score"]
        conditions = [
            risk_scores < 3,
            (risk_scores >= 3) & (risk_scores < 5),
            (risk_scores >= 5) & (risk_scores < 10),
            (risk_scores >= 10) & (risk_scores < 20),
            risk_scores >= 20
        ]
        choices = ["low", "low_moderate", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")

        return results
