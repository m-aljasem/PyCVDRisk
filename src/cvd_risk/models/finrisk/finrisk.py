"""
FINRISK Calculator for Cardiovascular Disease Risk.

FINRISK is the official cardiovascular disease risk assessment tool used
in Finland, developed by the National Institute for Health and Welfare (THL).
Multiple versions exist, with the latest incorporating modern risk factors.

This implementation includes FINRISK 2017, which is currently recommended
for clinical use in Finland.

Reference:
    Vartiainen E, Laatikainen T, Peltonen M, et al. Thirty-five-year
    trends in cardiovascular risk factors in Finland. Int J Epidemiol.
    2010;39(2):504-518.

    FINRISK 2017: Pajunen P, Vartiainen E, Laatikainen T, et al.
    Cardiovascular risk factor changes in Finland, 2017. Helsinki:
    National Institute for Health and Welfare; 2019.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# FINRISK 2017 coefficients for men and women
# Based on Finnish population data
_FINRISK_COEFFICIENTS_MEN_2017 = {
    "age": 0.077,
    "bmi": 0.023,
    "smoking": 0.693,
    "systolic_bp": 0.013,
    "total_cholesterol": 0.015,
    "hdl_cholesterol": -0.024,
    "diabetes": 0.579,
    "family_history": 0.270,
    "constant": -8.935
}

_FINRISK_COEFFICIENTS_WOMEN_2017 = {
    "age": 0.069,
    "bmi": 0.031,
    "smoking": 0.595,
    "systolic_bp": 0.011,
    "total_cholesterol": 0.019,
    "hdl_cholesterol": -0.027,
    "diabetes": 0.728,
    "family_history": 0.205,
    "constant": -8.738
}


class FINRISK(RiskModel):
    """
    FINRISK Calculator for cardiovascular disease risk assessment.

    FINRISK is Finland's national cardiovascular risk assessment tool,
    developed and validated in the Finnish population. This implementation
    uses FINRISK 2017, the current recommended version.

    Parameters
    ----------
    age : int
        Age in years (25-74 for optimal performance).
    sex : Literal["male", "female"]
        Biological sex.
    bmi : Optional[float]
        Body mass index in kg/m².
    systolic_bp : float
        Systolic blood pressure in mmHg.
    total_cholesterol : float
        Total cholesterol in mmol/L.
    hdl_cholesterol : Optional[float]
        HDL cholesterol in mmol/L.
    smoking : bool
        Current smoking status.
    diabetes : Optional[bool]
        Diabetes mellitus.
    family_history : Optional[bool]
        Family history of CVD.

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year CVD risk as percentage
        - risk_category: Risk classification
        - model_name: "FINRISK"
        - model_version: "2017"
    """

    model_name = "FINRISK"
    model_version = "2017"
    supported_regions = ["Finland"]  # Developed for Finnish population

    def __init__(self) -> None:
        """Initialize FINRISK model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for FINRISK requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for FINRISK calculation.
        """
        super().validate_input(patient)

        if patient.age < 25 or patient.age > 74:
            logger.warning(
                f"Age {patient.age} outside optimal range [25, 74] years. "
                "Results may have reduced accuracy."
            )

        # FINRISK requires basic measurements
        if patient.total_cholesterol is None:
            raise ValueError("FINRISK requires total cholesterol measurement")

        if patient.systolic_bp is None:
            raise ValueError("FINRISK requires systolic blood pressure measurement")

        # BMI is preferred but not strictly required
        if getattr(patient, 'bmi', None) is None:
            logger.warning("BMI not provided. FINRISK performs better with BMI included.")

    def _get_coefficients(self, sex: str) -> dict:
        """Get appropriate coefficients based on sex."""
        return _FINRISK_COEFFICIENTS_MEN_2017 if sex == "male" else _FINRISK_COEFFICIENTS_WOMEN_2017

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate 10-year CVD risk using FINRISK 2017.

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
        coeffs = self._get_coefficients(patient.sex)

        # Get BMI (use default if not provided)
        bmi = getattr(patient, 'bmi', 25.0)  # Default BMI 25

        # Convert cholesterol from mmol/L to mmol/L (FINRISK uses mmol/L)
        # No conversion needed as coefficients expect mmol/L

        # Calculate linear predictor
        linear_predictor = (
            coeffs["age"] * patient.age
            + coeffs["bmi"] * bmi
            + coeffs["smoking"] * float(patient.smoking)
            + coeffs["systolic_bp"] * patient.systolic_bp
            + coeffs["total_cholesterol"] * patient.total_cholesterol
            + coeffs["hdl_cholesterol"] * (patient.hdl_cholesterol if patient.hdl_cholesterol is not None else 1.5)
            + coeffs["diabetes"] * float(patient.diabetes if patient.diabetes is not None else False)
            + coeffs["family_history"] * float(getattr(patient, 'family_history', False))
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
            "bmi": bmi,
            "smoking": patient.smoking,
            "systolic_bp": patient.systolic_bp,
            "total_cholesterol": patient.total_cholesterol,
            "hdl_cholesterol": patient.hdl_cholesterol,
            "diabetes": patient.diabetes,
            "family_history": getattr(patient, 'family_history', False),
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
        """Categorize risk using FINRISK categories."""
        if risk_percentage < 7:
            return "low"
        elif risk_percentage < 15:
            return "moderate"
        elif risk_percentage < 25:
            return "high"
        else:
            return "very_high"

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized calculation for high-throughput analysis.
        """
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "smoking"]
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing required columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        # Fill missing optional columns with defaults
        defaults = {
            "bmi": 25.0,  # Default BMI
            "hdl_cholesterol": 1.5,  # Default HDL
            "diabetes": False,
            "family_history": False
        }

        for col, default in defaults.items():
            if col not in results.columns:
                results[col] = default

        # Vectorized calculations
        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any():
                continue

            coeffs = self._get_coefficients(sex)

            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"]
                + coeffs["bmi"] * results.loc[mask, "bmi"]
                + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float)
                + coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"]
                + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"]
                + coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"]
                + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float)
                + coeffs["family_history"] * results.loc[mask, "family_history"].astype(float)
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
            risk_scores < 7,
            (risk_scores >= 7) & (risk_scores < 15),
            (risk_scores >= 15) & (risk_scores < 25),
            risk_scores >= 25
        ]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")

        return results
