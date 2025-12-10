"""
Dundee CVD Risk Calculator.

The Dundee Risk Calculator is designed for use in Scotland, combining elements
from the Framingham Risk Score and ASSIGN score adapted for the Scottish population.

Reference:
    Newby DE, Wright RA, Labinjoh C, Ludlam CA, Fox KA, MacLean Forbes A,
    Cobbe SM. Cardiovascular risk estimation in Scotland and England /
    Wales. Scott Med J. 2007;52(1):21-27.
"""

import numpy as np
import pandas as pd
from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

# Dundee Risk Calculator coefficients (adapted for Scotland)
_DUNDEE_COEFFICIENTS_MEN = {
    "age": 0.048,
    "smoking": 0.420,
    "systolic_bp": 0.012,
    "total_cholesterol": 0.010,
    "hdl_cholesterol": -0.022,
    "diabetes": 0.480,
    "family_history": 0.250,
    "deprivation_score": 0.180,
    "constant": -7.900
}

_DUNDEE_COEFFICIENTS_WOMEN = {
    "age": 0.045,
    "smoking": 0.400,
    "systolic_bp": 0.014,
    "total_cholesterol": 0.012,
    "hdl_cholesterol": -0.024,
    "diabetes": 0.460,
    "family_history": 0.230,
    "deprivation_score": 0.170,
    "constant": -8.100
}


class Dundee(RiskModel):
    model_name = "Dundee"
    model_version = "2007"
    supported_regions = ["Scotland", "UK"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)
        if patient.total_cholesterol is None or patient.hdl_cholesterol is None:
            raise ValueError("Dundee calculator requires cholesterol measurements")

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        coeffs = _DUNDEE_COEFFICIENTS_MEN if patient.sex == "male" else _DUNDEE_COEFFICIENTS_WOMEN
        deprivation_score = getattr(patient, 'deprivation_score', 0)  # Scottish Index of Multiple Deprivation

        linear_predictor = (
            coeffs["age"] * patient.age + coeffs["smoking"] * float(patient.smoking) +
            coeffs["systolic_bp"] * patient.systolic_bp + coeffs["total_cholesterol"] * patient.total_cholesterol +
            coeffs["hdl_cholesterol"] * patient.hdl_cholesterol +
            coeffs["diabetes"] * float(patient.diabetes or False) +
            coeffs["family_history"] * float(getattr(patient, 'family_history', False)) +
            coeffs["deprivation_score"] * deprivation_score + coeffs["constant"]
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

        if "diabetes" not in results.columns: results["diabetes"] = False
        if "family_history" not in results.columns: results["family_history"] = False
        if "deprivation_score" not in results.columns: results["deprivation_score"] = 0

        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any(): continue

            coeffs = _DUNDEE_COEFFICIENTS_MEN if sex == 'male' else _DUNDEE_COEFFICIENTS_WOMEN
            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"] + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float) +
                coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"] + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"] +
                coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"] + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float) +
                coeffs["family_history"] * results.loc[mask, "family_history"].astype(float) + coeffs["deprivation_score"] * results.loc[mask, "deprivation_score"] + coeffs["constant"]
            )

            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        results["risk_score"] = np.clip(results["risk_score"], 0.0, 40.0)
        conditions = [results["risk_score"] < 5, (results["risk_score"] >= 5) & (results["risk_score"] < 10),
                     (results["risk_score"] >= 10) & (results["risk_score"] < 20), results["risk_score"] >= 20]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")
        return results
