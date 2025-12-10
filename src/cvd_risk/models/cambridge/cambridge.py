"""
Cambridge CVD Risk Score.

The Cambridge Risk Score is a UK-based cardiovascular risk assessment tool
that emphasizes family history and is designed for primary care use.

Reference:
    Wilson S, Chambers D, Cavanagh J, et al. An evaluation of the
    performance of the NHS Health Check programme. British Journal
    of General Practice. 2014;64(621):e231-e236.
"""

import numpy as np
import pandas as pd
from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

# Cambridge Risk Score coefficients
_CAMBRIDGE_COEFFICIENTS_MEN = {
    "age": 0.046,
    "smoking": 0.380,
    "systolic_bp": 0.012,
    "total_cholesterol": 0.009,
    "hdl_cholesterol": -0.021,
    "diabetes": 0.450,
    "bmi": 0.007,
    "family_history": 0.350,
    "townsend_deprivation": 0.120,
    "constant": -7.800
}

_CAMBRIDGE_COEFFICIENTS_WOMEN = {
    "age": 0.043,
    "smoking": 0.360,
    "systolic_bp": 0.014,
    "total_cholesterol": 0.011,
    "hdl_cholesterol": -0.023,
    "diabetes": 0.430,
    "bmi": 0.009,
    "family_history": 0.330,
    "townsend_deprivation": 0.110,
    "constant": -8.000
}


class Cambridge(RiskModel):
    model_name = "Cambridge"
    model_version = "2004"
    supported_regions = ["UK", "Europe"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)
        if patient.total_cholesterol is None or patient.hdl_cholesterol is None:
            raise ValueError("Cambridge Risk Score requires cholesterol measurements")

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        coeffs = _CAMBRIDGE_COEFFICIENTS_MEN if patient.sex == "male" else _CAMBRIDGE_COEFFICIENTS_WOMEN
        bmi = getattr(patient, 'bmi', 26.0)  # UK average BMI
        townsend_deprivation = getattr(patient, 'townsend_deprivation', 0)  # Socioeconomic deprivation score

        linear_predictor = (
            coeffs["age"] * patient.age + coeffs["smoking"] * float(patient.smoking) +
            coeffs["systolic_bp"] * patient.systolic_bp + coeffs["total_cholesterol"] * patient.total_cholesterol +
            coeffs["hdl_cholesterol"] * patient.hdl_cholesterol +
            coeffs["diabetes"] * float(patient.diabetes or False) + coeffs["bmi"] * bmi +
            coeffs["family_history"] * float(getattr(patient, 'family_history', False)) +
            coeffs["townsend_deprivation"] * townsend_deprivation + coeffs["constant"]
        )

        risk_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        risk_percentage = np.clip(risk_probability * 100.0, 0.0, 40.0)

        risk_category = "low" if risk_percentage < 5 else "moderate" if risk_percentage < 10 else "high" if risk_percentage < 20 else "very_high"

        return RiskResult(risk_score=float(risk_percentage), risk_category=risk_category,
                         model_name=self.model_name, model_version=self.model_version)

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        if "bmi" not in results.columns: results["bmi"] = 26.0
        if "diabetes" not in results.columns: results["diabetes"] = False
        if "family_history" not in results.columns: results["family_history"] = False
        if "townsend_deprivation" not in results.columns: results["townsend_deprivation"] = 0

        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any(): continue

            coeffs = _CAMBRIDGE_COEFFICIENTS_MEN if sex == 'male' else _CAMBRIDGE_COEFFICIENTS_WOMEN
            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"] + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float) +
                coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"] + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"] +
                coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"] + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float) +
                coeffs["bmi"] * results.loc[mask, "bmi"] + coeffs["family_history"] * results.loc[mask, "family_history"].astype(float) +
                coeffs["townsend_deprivation"] * results.loc[mask, "townsend_deprivation"] + coeffs["constant"]
            )

            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        results["risk_score"] = np.clip(results["risk_score"], 0.0, 40.0)
        conditions = [results["risk_score"] < 5, (results["risk_score"] >= 5) & (results["risk_score"] < 10),
                     (results["risk_score"] >= 10) & (results["risk_score"] < 20), results["risk_score"] >= 20]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")
        return results
