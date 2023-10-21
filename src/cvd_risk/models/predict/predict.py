"""
PREDICT CVD Risk Calculator.

PREDICT is New Zealand's web-based cardiovascular disease risk assessment tool.
It provides 5-year CVD risk prediction and has been extensively validated
in the New Zealand population.

Reference:
    Riddell T, Jackson R, Wells S, et al. Performance of Framingham cardiovascular
    risk equations in New Zealand: a calibration study in primary care. New Zealand
    Medical Journal. 2010;123(1323):48-58.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# PREDICT model coefficients (adapted for New Zealand population)
# Based on published equations
_PREDICT_COEFFICIENTS_MEN = {
    "age": 0.0635,
    "ln_systolic_bp": 0.0119,
    "smoking": 0.738,
    "diabetes": 0.649,
    "total_cholesterol": 0.0113,
    "hdl_cholesterol": -0.032,
    "ethnicity_maori": 0.317,
    "ethnicity_pacific": 0.417,
    "ethnicity_other": 0.099,
    "family_history": 0.269,
    "atrial_fibrillation": 0.531,
    "left_ventricular_hypertrophy": 0.434,
    "constant": -8.912
}

_PREDICT_COEFFICIENTS_WOMEN = {
    "age": 0.0622,
    "ln_systolic_bp": 0.0115,
    "smoking": 0.724,
    "diabetes": 0.582,
    "total_cholesterol": 0.0137,
    "hdl_cholesterol": -0.036,
    "ethnicity_maori": 0.312,
    "ethnicity_pacific": 0.398,
    "ethnicity_other": 0.087,
    "family_history": 0.245,
    "atrial_fibrillation": 0.487,
    "left_ventricular_hypertrophy": 0.412,
    "constant": -8.734
}


class PREDICT(RiskModel):
    """
    PREDICT CVD Risk Calculator for New Zealand.

    PREDICT is New Zealand's national cardiovascular risk assessment tool,
    providing 5-year CVD risk prediction with ethnicity-specific adjustments.

    Parameters
    ----------
    age : int
        Age in years (30-74 for optimal performance).
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
    ethnicity : Optional[Literal["european", "maori", "pacific", "other"]]
        Ethnic group (European = reference).
    family_history : Optional[bool]
        Family history of premature CVD.
    atrial_fibrillation : Optional[bool]
        Atrial fibrillation.
    left_ventricular_hypertrophy : Optional[bool]
        Left ventricular hypertrophy.

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 5-year CVD risk as percentage
        - risk_category: Risk classification
        - model_name: "PREDICT"
        - model_version: "2010"
    """

    model_name = "PREDICT"
    model_version = "2010"
    supported_regions = ["New Zealand"]  # New Zealand population

    def __init__(self) -> None:
        """Initialize PREDICT model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for PREDICT requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for PREDICT calculation.
        """
        super().validate_input(patient)

        if patient.age < 30 or patient.age > 74:
            logger.warning(
                f"Age {patient.age} outside optimal range [30, 74] years. "
                "Results may have reduced accuracy."
            )

        # PREDICT requires cholesterol measurements
        if patient.total_cholesterol is None:
            raise ValueError("PREDICT requires total cholesterol measurement")

        if patient.hdl_cholesterol is None:
            raise ValueError("PREDICT requires HDL cholesterol measurement")

        if patient.systolic_bp is None:
            raise ValueError("PREDICT requires systolic blood pressure measurement")

    def _get_ethnicity_coefficient(self, ethnicity: Optional[str]) -> float:
        """Get ethnicity coefficient for PREDICT model."""
        if ethnicity == "maori":
            return 0.317 if self._current_sex == "male" else 0.312
        elif ethnicity == "pacific":
            return 0.417 if self._current_sex == "male" else 0.398
        elif ethnicity == "other":
            return 0.099 if self._current_sex == "male" else 0.087
        else:  # european or None (reference)
            return 0.0

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate 5-year CVD risk using PREDICT.

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
        self._current_sex = patient.sex  # Store for ethnicity coefficient

        # Get appropriate coefficients
        coeffs = _PREDICT_COEFFICIENTS_MEN if patient.sex == "male" else _PREDICT_COEFFICIENTS_WOMEN

        # Get ethnicity coefficient
        ethnicity_coeff = self._get_ethnicity_coefficient(getattr(patient, 'ethnicity', None))

        # Calculate linear predictor
        linear_predictor = (
            coeffs["age"] * patient.age
            + coeffs["ln_systolic_bp"] * np.log(patient.systolic_bp)
            + coeffs["smoking"] * float(patient.smoking)
            + coeffs["diabetes"] * float(patient.diabetes if patient.diabetes is not None else False)
            + coeffs["total_cholesterol"] * patient.total_cholesterol
            + coeffs["hdl_cholesterol"] * patient.hdl_cholesterol
            + ethnicity_coeff
            + coeffs["family_history"] * float(getattr(patient, 'family_history', False))
            + coeffs["atrial_fibrillation"] * float(getattr(patient, 'atrial_fibrillation', False))
            + coeffs["left_ventricular_hypertrophy"] * float(getattr(patient, 'left_ventricular_hypertrophy', False))
            + coeffs["constant"]
        )

        # Convert to 5-year risk probability
        risk_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        risk_percentage = risk_probability * 100.0

        # Ensure risk is within reasonable bounds
        risk_percentage = np.clip(risk_percentage, 0.0, 25.0)  # 5-year risk, so lower maximum

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
            "ethnicity": getattr(patient, 'ethnicity', 'european'),
            "family_history": getattr(patient, 'family_history', False),
            "atrial_fibrillation": getattr(patient, 'atrial_fibrillation', False),
            "left_ventricular_hypertrophy": getattr(patient, 'left_ventricular_hypertrophy', False),
            "linear_predictor": linear_predictor,
            "note": "This is a 5-year risk prediction (not 10-year)"
        })

        return RiskResult(
            risk_score=float(risk_percentage),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata=metadata,
        )

    def _categorize_risk(self, risk_percentage: float) -> str:
        """Categorize risk using PREDICT categories (for 5-year risk)."""
        if risk_percentage < 2.5:
            return "low"
        elif risk_percentage < 5:
            return "low_moderate"
        elif risk_percentage < 10:
            return "moderate"
        elif risk_percentage < 15:
            return "high_moderate"
        else:
            return "high"

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
            "ethnicity": "european",
            "family_history": False,
            "atrial_fibrillation": False,
            "left_ventricular_hypertrophy": False
        }

        for col, default in defaults.items():
            if col not in results.columns:
                results[col] = default

        # Vectorized calculations
        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any():
                continue

            coeffs = _PREDICT_COEFFICIENTS_MEN if sex == 'male' else _PREDICT_COEFFICIENTS_WOMEN

            # Ethnicity coefficients
            ethnicity_coeffs = np.zeros(len(results))
            ethnicity_maori = (results.loc[mask, "ethnicity"] == "maori")
            ethnicity_pacific = (results.loc[mask, "ethnicity"] == "pacific")
            ethnicity_other = (results.loc[mask, "ethnicity"] == "other")

            if sex == 'male':
                ethnicity_coeffs[ethnicity_maori] = 0.317
                ethnicity_coeffs[ethnicity_pacific] = 0.417
                ethnicity_coeffs[ethnicity_other] = 0.099
            else:
                ethnicity_coeffs[ethnicity_maori] = 0.312
                ethnicity_coeffs[ethnicity_pacific] = 0.398
                ethnicity_coeffs[ethnicity_other] = 0.087

            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"]
                + coeffs["ln_systolic_bp"] * np.log(results.loc[mask, "systolic_bp"])
                + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float)
                + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float)
                + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"]
                + coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"]
                + ethnicity_coeffs
                + coeffs["family_history"] * results.loc[mask, "family_history"].astype(float)
                + coeffs["atrial_fibrillation"] * results.loc[mask, "atrial_fibrillation"].astype(float)
                + coeffs["left_ventricular_hypertrophy"] * results.loc[mask, "left_ventricular_hypertrophy"].astype(float)
                + coeffs["constant"]
            )

            # Convert to probability
            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        # Clip risks
        results["risk_score"] = np.clip(results["risk_score"], 0.0, 25.0)

        # Vectorized categorization
        risk_scores = results["risk_score"]
        conditions = [
            risk_scores < 2.5,
            (risk_scores >= 2.5) & (risk_scores < 5),
            (risk_scores >= 5) & (risk_scores < 10),
            (risk_scores >= 10) & (risk_scores < 15),
            risk_scores >= 15
        ]
        choices = ["low", "low_moderate", "moderate", "high_moderate", "high"]
        results["risk_category"] = np.select(conditions, choices, default="high")

        return results
