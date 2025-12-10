"""
PROCAM (Prospective Cardiovascular Münster) Risk Score.

The PROCAM study developed a risk score for coronary heart disease prediction
in the German population, incorporating lipid parameters and other risk factors.

Reference:
    Assmann G, Cullen P, Schulte H. Simple scoring scheme for calculating
    the risk of acute coronary events based on the 10-year follow-up of
    the prospective cardiovascular Münster (PROCAM) study. Circulation.
    2002;105(3):310-315.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# PROCAM coefficients for men and women
# From Assmann et al. 2002
_PROCAM_COEFFICIENTS_MEN = {
    "age": 0.0324,
    "smoking": 0.7126,
    "diabetes": 0.6990,
    "systolic_bp": 0.0112,
    "total_cholesterol": 0.0144,
    "hdl_cholesterol": -0.0138,
    "triglycerides": 0.0067,
    "ldl_cholesterol": 0.0085,
    "family_history": 0.4398,
    "constant": -9.0137
}

_PROCAM_COEFFICIENTS_WOMEN = {
    "age": 0.0386,
    "smoking": 0.8369,
    "diabetes": 0.7098,
    "systolic_bp": 0.0101,
    "total_cholesterol": 0.0132,
    "hdl_cholesterol": -0.0113,
    "triglycerides": 0.0073,
    "ldl_cholesterol": 0.0090,
    "family_history": 0.4352,
    "constant": -10.0115
}


class PROCAM(RiskModel):
    """
    PROCAM Risk Score for coronary heart disease prediction.

    The PROCAM model estimates 10-year risk of coronary heart disease
    using logistic regression coefficients from the German PROCAM study.

    Parameters
    ----------
    age : int
        Age in years (20-65 for optimal performance).
    sex : Literal["male", "female"]
        Biological sex.
    systolic_bp : float
        Systolic blood pressure in mmHg.
    total_cholesterol : float
        Total cholesterol in mmol/L.
    hdl_cholesterol : float
        HDL cholesterol in mmol/L.
    triglycerides : Optional[float]
        Triglycerides in mmol/L.
    ldl_cholesterol : Optional[float]
        LDL cholesterol in mmol/L (calculated if not provided).
    smoking : bool
        Current smoking status.
    diabetes : Optional[bool]
        Diabetes status.
    family_history : Optional[bool]
        Family history of premature coronary heart disease.

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year CHD risk as percentage
        - risk_category: Risk classification
        - model_name: "PROCAM"
        - model_version: Algorithm version identifier
    """

    model_name = "PROCAM"
    model_version = "2002"
    supported_regions = ["Europe", "Germany"]  # Primarily validated in German population

    def __init__(self) -> None:
        """Initialize PROCAM model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for PROCAM requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for PROCAM calculation.
        """
        super().validate_input(patient)

        if patient.age < 20 or patient.age > 75:
            logger.warning(
                f"Age {patient.age} outside optimal range [20, 75] years. "
                "Results may have reduced accuracy."
            )

        # PROCAM requires cholesterol measurements
        if patient.total_cholesterol is None:
            raise ValueError("PROCAM model requires total cholesterol measurement")

        if patient.hdl_cholesterol is None:
            raise ValueError("PROCAM model requires HDL cholesterol measurement")

        # Check if we have LDL or can calculate it
        if patient.ldl_cholesterol is None and patient.triglycerides is None:
            logger.warning(
                "LDL cholesterol not provided. Will estimate from total cholesterol, "
                "HDL, and triglycerides if available."
            )

    def _calculate_ldl(self, patient: PatientData) -> float:
        """Estimate LDL cholesterol using Friedewald equation if not provided."""
        if patient.ldl_cholesterol is not None:
            return patient.ldl_cholesterol

        # Friedewald equation: LDL = TC - HDL - (TG/5) when TG < 4.5 mmol/L
        if patient.triglycerides is not None and patient.triglycerides < 4.5:
            return patient.total_cholesterol - patient.hdl_cholesterol - (patient.triglycerides / 5.0)

        # If triglycerides >= 4.5 or not available, use simpler estimate
        return patient.total_cholesterol - patient.hdl_cholesterol - 0.8

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate 10-year CHD risk using PROCAM score.

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
        coeffs = _PROCAM_COEFFICIENTS_MEN if patient.sex == "male" else _PROCAM_COEFFICIENTS_WOMEN

        # Calculate LDL if needed
        ldl_chol = self._calculate_ldl(patient)

        # Calculate linear predictor
        linear_predictor = (
            coeffs["age"] * patient.age
            + coeffs["smoking"] * float(patient.smoking)
            + coeffs["diabetes"] * float(patient.diabetes if patient.diabetes is not None else False)
            + coeffs["systolic_bp"] * patient.systolic_bp
            + coeffs["total_cholesterol"] * patient.total_cholesterol
            + coeffs["hdl_cholesterol"] * patient.hdl_cholesterol
            + coeffs["triglycerides"] * (patient.triglycerides if patient.triglycerides is not None else 1.5)
            + coeffs["ldl_cholesterol"] * ldl_chol
            + coeffs["family_history"] * float(patient.family_history if patient.family_history is not None else False)
            + coeffs["constant"]
        )

        # Convert to probability using logistic function
        risk_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        risk_percentage = risk_probability * 100.0

        # Ensure risk is within reasonable bounds
        risk_percentage = np.clip(risk_percentage, 0.0, 50.0)  # PROCAM typically gives lower risks

        # Categorize risk
        risk_category = self._categorize_risk(risk_percentage)

        metadata = self._get_metadata()
        metadata.update({
            "age": patient.age,
            "sex": patient.sex,
            "smoking": patient.smoking,
            "diabetes": patient.diabetes,
            "systolic_bp": patient.systolic_bp,
            "total_cholesterol": patient.total_cholesterol,
            "hdl_cholesterol": patient.hdl_cholesterol,
            "triglycerides": patient.triglycerides,
            "ldl_cholesterol": ldl_chol,
            "family_history": patient.family_history,
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
        """Categorize risk using PROCAM categories."""
        if risk_percentage < 5:
            return "low"
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
            "family_history": False,
            "triglycerides": 1.5,
            "ldl_cholesterol": None
        }

        for col, default in defaults.items():
            if col not in results.columns:
                results[col] = default

        # Handle LDL calculation
        def calculate_ldl_row(row):
            if pd.notna(row.get('ldl_cholesterol')):
                return row['ldl_cholesterol']
            if pd.notna(row.get('triglycerides')) and row['triglycerides'] < 4.5:
                return row['total_cholesterol'] - row['hdl_cholesterol'] - (row['triglycerides'] / 5.0)
            return row['total_cholesterol'] - row['hdl_cholesterol'] - 0.8

        results['calculated_ldl'] = results.apply(calculate_ldl_row, axis=1)

        # Vectorized calculations
        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any():
                continue

            coeffs = _PROCAM_COEFFICIENTS_MEN if sex == 'male' else _PROCAM_COEFFICIENTS_WOMEN

            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"]
                + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float)
                + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float)
                + coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"]
                + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"]
                + coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"]
                + coeffs["triglycerides"] * results.loc[mask, "triglycerides"]
                + coeffs["ldl_cholesterol"] * results.loc[mask, "calculated_ldl"]
                + coeffs["family_history"] * results.loc[mask, "family_history"].astype(float)
                + coeffs["constant"]
            )

            # Convert to probability
            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        # Clip risks
        results["risk_score"] = np.clip(results["risk_score"], 0.0, 50.0)

        # Vectorized categorization
        risk_scores = results["risk_score"]
        conditions = [
            risk_scores < 5,
            (risk_scores >= 5) & (risk_scores < 10),
            (risk_scores >= 10) & (risk_scores < 20),
            risk_scores >= 20
        ]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="unknown")

        return results
