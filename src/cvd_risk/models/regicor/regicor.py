"""
REGICOR (Registre Gironí del Cor) Risk Score.

REGICOR is a cardiovascular risk prediction model adapted from the Framingham
Heart Study for the Catalan population in Spain. It estimates 10-year risk
of coronary heart disease.

Reference:
    Marrugat J, D'Agostino R, Sullivan L, et al. An adaptation of the
    Framingham coronary heart disease risk function to European Mediterranean
    areas. Journal of Epidemiology & Community Health. 2003;57(8):634-638.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# REGICOR coefficients (adapted from Framingham for Catalan population)
# From Marrugat et al. 2003
_REGICOR_COEFFICIENTS_MEN = {
    "age": 0.04826,
    "ln_systolic_bp": 1.0566,
    "ln_total_cholesterol": 0.2145,
    "ln_hdl_cholesterol": -0.1569,
    "smoking": 0.4527,
    "diabetes": 0.4369,
    "constant": -22.259
}

_REGICOR_COEFFICIENTS_WOMEN = {
    "age": 0.33766,
    "ln_systolic_bp": 1.3954,
    "ln_total_cholesterol": 0.4361,
    "ln_hdl_cholesterol": -0.2107,
    "smoking": 0.5291,
    "diabetes": 0.4998,
    "constant": -26.193
}


class REGICOR(RiskModel):
    """
    REGICOR Risk Score for cardiovascular disease prediction.

    REGICOR is the official cardiovascular risk assessment tool in Catalonia,
    Spain, adapted from Framingham equations for the Mediterranean population.

    Parameters
    ----------
    age : int
        Age in years (35-74 for optimal performance).
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

    Returns
    -------
    RiskResult
        Contains:
        - risk_score: 10-year CVD risk as percentage
        - risk_category: Risk classification
        - model_name: "REGICOR"
        - model_version: "2003"
    """

    model_name = "REGICOR"
    model_version = "2003"
    supported_regions = ["Spain", "Catalonia", "Europe"]  # Catalan population

    def __init__(self) -> None:
        """Initialize REGICOR model."""
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        """
        Validate patient input for REGICOR requirements.

        Parameters
        ----------
        patient : PatientData
            Patient data to validate.

        Raises
        ------
        ValueError
            If patient data is invalid for REGICOR calculation.
        """
        super().validate_input(patient)

        if patient.age < 35 or patient.age > 74:
            logger.warning(
                f"Age {patient.age} outside optimal range [35, 74] years. "
                "Results may have reduced accuracy."
            )

        # REGICOR requires cholesterol measurements
        if patient.total_cholesterol is None:
            raise ValueError("REGICOR requires total cholesterol measurement")

        if patient.hdl_cholesterol is None:
            raise ValueError("REGICOR requires HDL cholesterol measurement")

        if patient.systolic_bp is None:
            raise ValueError("REGICOR requires systolic blood pressure measurement")

    def calculate(self, patient: PatientData) -> RiskResult:
        """
        Calculate 10-year CVD risk using REGICOR score.

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
        coeffs = _REGICOR_COEFFICIENTS_MEN if patient.sex == "male" else _REGICOR_COEFFICIENTS_WOMEN

        # Calculate linear predictor
        linear_predictor = (
            coeffs["age"] * patient.age
            + coeffs["ln_systolic_bp"] * np.log(patient.systolic_bp)
            + coeffs["ln_total_cholesterol"] * np.log(patient.total_cholesterol)
            + coeffs["ln_hdl_cholesterol"] * np.log(patient.hdl_cholesterol)
            + coeffs["smoking"] * float(patient.smoking)
            + coeffs["diabetes"] * float(patient.diabetes if patient.diabetes is not None else False)
            + coeffs["constant"]
        )

        # Convert to risk probability using logistic function
        risk_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        risk_percentage = risk_probability * 100.0

        # Ensure risk is within reasonable bounds
        risk_percentage = np.clip(risk_percentage, 0.0, 50.0)

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
        """Categorize risk using REGICOR categories."""
        if risk_percentage < 5:
            return "low"
        elif risk_percentage < 10:
            return "moderate"
        elif risk_percentage < 15:
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
            "diabetes": False
        }

        for col, default in defaults.items():
            if col not in results.columns:
                results[col] = default

        # Vectorized calculations
        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any():
                continue

            coeffs = _REGICOR_COEFFICIENTS_MEN if sex == 'male' else _REGICOR_COEFFICIENTS_WOMEN

            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"]
                + coeffs["ln_systolic_bp"] * np.log(results.loc[mask, "systolic_bp"])
                + coeffs["ln_total_cholesterol"] * np.log(results.loc[mask, "total_cholesterol"])
                + coeffs["ln_hdl_cholesterol"] * np.log(results.loc[mask, "hdl_cholesterol"])
                + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float)
                + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float)
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
            (risk_scores >= 10) & (risk_scores < 15),
            risk_scores >= 15
        ]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")

        return results
