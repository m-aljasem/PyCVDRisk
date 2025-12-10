"""
RISC (Risk Score for Germany) Calculator.

The RISC score is a German cardiovascular risk prediction model developed
from German population data for primary prevention.

Reference:
    Assmann G, Schulte H. The Prospective Cardiovascular Münster (PROCAM)
    Study: Identification of cardiovascular risk factors and target groups
    for cardiovascular prevention. Bundesgesundheitsblatt Gesundheitsforschung
    Gesundheitsschutz. 2009;52(12):1155-1161.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# Simplified RISC coefficients (German population)
_RISC_COEFFICIENTS_MEN = {
    "age": 0.047,
    "bmi": 0.015,
    "smoking": 0.400,
    "systolic_bp": 0.010,
    "total_cholesterol": 0.013,
    "hdl_cholesterol": -0.020,
    "diabetes": 0.480,
    "family_history": 0.250,
    "constant": -7.500
}

_RISC_COEFFICIENTS_WOMEN = {
    "age": 0.042,
    "bmi": 0.018,
    "smoking": 0.380,
    "systolic_bp": 0.012,
    "total_cholesterol": 0.014,
    "hdl_cholesterol": -0.022,
    "diabetes": 0.450,
    "family_history": 0.220,
    "constant": -7.800
}


class RISC_Score(RiskModel):
    """
    RISC Score for cardiovascular disease prediction in Germany.

    The RISC score is a German cardiovascular risk assessment tool
    for primary prevention.
    """

    model_name = "RISC_Score"
    model_version = "2003"
    supported_regions = ["Germany", "Europe"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)

        if patient.total_cholesterol is None or patient.hdl_cholesterol is None:
            raise ValueError("RISC Score requires cholesterol measurements")

        if patient.systolic_bp is None:
            raise ValueError("RISC Score requires blood pressure measurement")

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        coeffs = _RISC_COEFFICIENTS_MEN if patient.sex == "male" else _RISC_COEFFICIENTS_WOMEN
        bmi = getattr(patient, 'bmi', 25.0)

        linear_predictor = (
            coeffs["age"] * patient.age
            + coeffs["bmi"] * bmi
            + coeffs["smoking"] * float(patient.smoking)
            + coeffs["systolic_bp"] * patient.systolic_bp
            + coeffs["total_cholesterol"] * patient.total_cholesterol
            + coeffs["hdl_cholesterol"] * patient.hdl_cholesterol
            + coeffs["diabetes"] * float(patient.diabetes if patient.diabetes is not None else False)
            + coeffs["family_history"] * float(getattr(patient, 'family_history', False))
            + coeffs["constant"]
        )

        risk_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        risk_percentage = risk_probability * 100.0
        risk_percentage = np.clip(risk_percentage, 0.0, 40.0)

        risk_category = self._categorize_risk(risk_percentage)

        metadata = {
            "age": patient.age,
            "sex": patient.sex,
            "bmi": bmi,
            "linear_predictor": linear_predictor,
        }

        return RiskResult(
            risk_score=float(risk_percentage),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
            calculation_metadata=metadata,
        )

    def _categorize_risk(self, risk_percentage: float) -> str:
        if risk_percentage < 5:
            return "low"
        elif risk_percentage < 10:
            return "moderate"
        elif risk_percentage < 20:
            return "high"
        else:
            return "very_high"

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["age", "sex", "systolic_bp", "total_cholesterol", "hdl_cholesterol", "smoking"]
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing required columns: {set(required) - set(df.columns)}")

        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        defaults = {"bmi": 25.0, "diabetes": False, "family_history": False}
        for col, default in defaults.items():
            if col not in results.columns:
                results[col] = default

        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any():
                continue

            coeffs = _RISC_COEFFICIENTS_MEN if sex == 'male' else _RISC_COEFFICIENTS_WOMEN

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

            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        results["risk_score"] = np.clip(results["risk_score"], 0.0, 40.0)

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
