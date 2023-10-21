"""
Reynolds Risk Score for Cardiovascular Disease.

The Reynolds Risk Score enhances traditional risk prediction by incorporating
high-sensitivity C-reactive protein (hsCRP) and family history, providing
more personalized cardiovascular risk assessment.

Reference:
    Ridker PM, Buring JE, Rifai N, Cook NR. Development and validation of
    improved algorithms for the assessment of global cardiovascular risk
    in women: the Reynolds Risk Score. JAMA. 2007;297(6):611-619.

    Ridker PM, Paynter NP, Rifai N, Gaziano JM, Cook NR. C-reactive protein
    and parental history improve global cardiovascular risk prediction: the
    Reynolds Risk Score for men. Circulation. 2008;118(22):2243-2251.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# Reynolds Risk Score coefficients
# From Ridker et al. (2007) for women and (2008) for men
_REYNOLDS_COEFFICIENTS_WOMEN = {
    "age": 0.0799,
    "systolic_bp": 3.137,
    "smoking": 0.180,
    "total_cholesterol": 1.382,
    "hdl_cholesterol": -1.288,
    "hs_crp": 0.102,
    "family_history": 0.159,
    "constant": -12.823
}

_REYNOLDS_COEFFICIENTS_MEN = {
    "age": 0.3742,
    "systolic_bp": 2.469,
    "smoking": 0.463,
    "total_cholesterol": 0.100,
    "hdl_cholesterol": -0.659,
    "hs_crp": 0.078,
    "family_history": 0.263,
    "constant": -22.460
}


class Reynolds(RiskModel):
    """
    Reynolds Risk Score for cardiovascular disease prediction.

    The Reynolds Risk Score enhances traditional risk factors with
    high-sensitivity C-reactive protein (hsCRP) and family history
    for more accurate risk stratification.

    Parameters
    ----------
    age : int
        Age in years (45-80 for optimal performance).
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
    hs_crp : Optional[float]
        High-sensitivity C-reactive protein in mg/L.
    family_history : Optional[bool]
        Family history of premature coronary heart disease.

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year CVD risk as percentage
        - risk_category: Risk classification
        - model_name: "Reynolds"
        - model_version: "2007/2008"
    """

    model_name = "Reynolds"
    model_version = "2007/2008"
    supported_regions = ["US"]  # Developed and validated in US populations

    def __init__(self) -> None:
        """Initialize Reynolds Risk Score model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for Reynolds Risk Score requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for Reynolds calculation.
        """
        super().validate_input(patient)

        if patient.age < 45 or patient.age > 80:
            logger.warning(
                f"Age {patient.age} outside optimal range [45, 80] years. "
                "Results may have reduced accuracy."
            )

        # Reynolds requires cholesterol measurements
        if patient.total_cholesterol is None:
            raise ValueError("Reynolds Risk Score requires total cholesterol measurement")

        if patient.hdl_cholesterol is None:
            raise ValueError("Reynolds Risk Score requires HDL cholesterol measurement")

        # hsCRP is a key component but may be optional
        if getattr(patient, 'hs_crp', None) is None:
            logger.warning(
                "hsCRP not provided. The Reynolds Risk Score is designed to include "
                "high-sensitivity C-reactive protein for optimal performance."
            )

    def _get_coefficients(self, sex: str) -> dict:
        """Get appropriate coefficients based on sex."""
        return _REYNOLDS_COEFFICIENTS_MEN if sex == "male" else _REYNOLDS_COEFFICIENTS_WOMEN

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate 10-year CVD risk using Reynolds Risk Score.

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

        # Convert cholesterol from mmol/L to mg/dL (as expected by the equations)
        total_chol_mgdl = patient.total_cholesterol * 38.67
        hdl_mgdl = patient.hdl_cholesterol * 38.67

        # Get hsCRP value (use default if not provided)
        hs_crp = getattr(patient, 'hs_crp', 1.0)  # Default 1.0 mg/L

        # Get family history (default to False if not provided)
        family_history = getattr(patient, 'family_history', False)

        # Calculate linear predictor
        linear_predictor = (
            coeffs["age"] * np.log(patient.age)
            + coeffs["systolic_bp"] * np.log(patient.systolic_bp)
            + coeffs["smoking"] * float(patient.smoking)
            + coeffs["total_cholesterol"] * np.log(total_chol_mgdl)
            + coeffs["hdl_cholesterol"] * np.log(hdl_mgdl)
            + coeffs["hs_crp"] * np.log(hs_crp)
            + coeffs["family_history"] * float(family_history)
            + coeffs["constant"]
        )

        # Convert to risk probability using logistic function
        risk_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        risk_percentage = risk_probability * 100.0

        # Ensure risk is within reasonable bounds
        risk_percentage = np.clip(risk_percentage, 0.0, 30.0)

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
            "hs_crp": hs_crp,
            "family_history": family_history,
            "total_chol_mgdl": total_chol_mgdl,
            "hdl_mgdl": hdl_mgdl,
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
        """Categorize risk using Reynolds categories."""
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
            "hs_crp": 1.0,  # Default hsCRP 1.0 mg/L
            "family_history": False
        }

        for col, default in defaults.items():
            if col not in results.columns:
                results[col] = default

        # Convert cholesterol from mmol/L to mg/dL
        total_chol_mgdl = results["total_cholesterol"] * 38.67
        hdl_mgdl = results["hdl_cholesterol"] * 38.67

        # Vectorized calculations
        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any():
                continue

            coeffs = self._get_coefficients(sex)

            linear_predictor = (
                coeffs["age"] * np.log(results.loc[mask, "age"])
                + coeffs["systolic_bp"] * np.log(results.loc[mask, "systolic_bp"])
                + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float)
                + coeffs["total_cholesterol"] * np.log(total_chol_mgdl[mask])
                + coeffs["hdl_cholesterol"] * np.log(hdl_mgdl[mask])
                + coeffs["hs_crp"] * np.log(results.loc[mask, "hs_crp"])
                + coeffs["family_history"] * results.loc[mask, "family_history"].astype(float)
                + coeffs["constant"]
            )

            # Convert to probability
            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        # Clip risks
        results["risk_score"] = np.clip(results["risk_score"], 0.0, 30.0)

        # Vectorized categorization
        risk_scores = results["risk_score"]
        conditions = [
            risk_scores < 5,
            (risk_scores >= 5) & (risk_scores < 10),
            (risk_scores >= 10) & (risk_scores < 20),
            risk_scores >= 20
        ]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")

        return results
