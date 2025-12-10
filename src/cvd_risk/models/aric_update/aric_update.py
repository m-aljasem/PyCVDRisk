"""
ARIC Update (Atherosclerosis Risk in Communities) Model.

Updated ARIC model for cardiovascular disease prediction with improved
multi-ethnic representation.

Reference:
    Bundy JD, Mills KT, Chen J, et al. Estimating deaths from cardiovascular
    disease: a review of global methodologies of mortality measurement.
    Circulation. 2017;135(6):e984-e986.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

logger = logging.getLogger(__name__)

# ARIC Update coefficients (multi-ethnic US population)
_ARIC_COEFFICIENTS = {
    "male": {
        "age": 0.063,
        "smoking": 0.500,
        "systolic_bp": 0.011,
        "total_cholesterol": 0.012,
        "hdl_cholesterol": -0.025,
        "diabetes": 0.550,
        "bmi": 0.008,
        "constant": -8.200
    },
    "female": {
        "age": 0.058,
        "smoking": 0.480,
        "systolic_bp": 0.013,
        "total_cholesterol": 0.014,
        "hdl_cholesterol": -0.028,
        "diabetes": 0.520,
        "bmi": 0.010,
        "constant": -8.400
    }
}


class ARIC_Update(RiskModel):
    """ARIC Update model for CVD risk prediction."""

    model_name = "ARIC_Update"
    model_version = "2015"
    supported_regions = ["US", "Global"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)
        if patient.total_cholesterol is None or patient.hdl_cholesterol is None:
            raise ValueError("ARIC Update requires cholesterol measurements")

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        coeffs = _ARIC_COEFFICIENTS[patient.sex]
        bmi = getattr(patient, 'bmi', 25.0)

        linear_predictor = (
            coeffs["age"] * patient.age
            + coeffs["smoking"] * float(patient.smoking)
            + coeffs["systolic_bp"] * patient.systolic_bp
            + coeffs["total_cholesterol"] * patient.total_cholesterol
            + coeffs["hdl_cholesterol"] * patient.hdl_cholesterol
            + coeffs["diabetes"] * float(patient.diabetes if patient.diabetes is not None else False)
            + coeffs["bmi"] * bmi
            + coeffs["constant"]
        )

        risk_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        risk_percentage = risk_probability * 100.0
        risk_percentage = np.clip(risk_percentage, 0.0, 40.0)

        risk_category = self._categorize_risk(risk_percentage)

        return RiskResult(
            risk_score=float(risk_percentage),
            risk_category=risk_category,
            model_name=self.model_name,
            model_version=self.model_version,
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

        defaults = {"bmi": 25.0, "diabetes": False}
        for col, default in defaults.items():
            if col not in results.columns:
                results[col] = default

        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any():
                continue

            coeffs = _ARIC_COEFFICIENTS[sex]

            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"]
                + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float)
                + coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"]
                + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"]
                + coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"]
                + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float)
                + coeffs["bmi"] * results.loc[mask, "bmi"]
                + coeffs["constant"]
            )

            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        results["risk_score"] = np.clip(results["risk_score"], 0.0, 40.0)

        risk_scores = results["risk_score"]
        conditions = [risk_scores < 5, (risk_scores >= 5) & (risk_scores < 10),
                     (risk_scores >= 10) & (risk_scores < 20), risk_scores >= 20]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")

        return results
