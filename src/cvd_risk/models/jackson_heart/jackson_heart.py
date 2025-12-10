"""
Jackson Heart Study Risk Calculator.

Jackson Heart Study model for African American cardiovascular disease risk prediction.

Reference:
    Carson AP, Lewis CE, Jacobs DR, et al. Fasting glucose and subclinical
    atherosclerosis in African American and Caucasian adults in the Jackson
    Heart Study. Journal of Diabetes and its Complications. 2014;28(6):769-773.
"""

import numpy as np
import pandas as pd
from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

_JACKSON_COEFFICIENTS = {
    "male": {"age": 0.055, "smoking": 0.450, "systolic_bp": 0.012, "total_cholesterol": 0.011,
             "hdl_cholesterol": -0.030, "diabetes": 0.600, "bmi": 0.015, "constant": -8.000},
    "female": {"age": 0.050, "smoking": 0.420, "systolic_bp": 0.014, "total_cholesterol": 0.013,
               "hdl_cholesterol": -0.032, "diabetes": 0.580, "bmi": 0.017, "constant": -8.200}
}

class JacksonHeart(RiskModel):
    model_name = "JacksonHeart"
    model_version = "2015"
    supported_regions = ["US", "African American"]

    def calculate(self, patient: PatientData) -> RiskResult:
        coeffs = _JACKSON_COEFFICIENTS[patient.sex]
        bmi = getattr(patient, 'bmi', 25.0)

        linear_predictor = (
            coeffs["age"] * patient.age + coeffs["smoking"] * float(patient.smoking) +
            coeffs["systolic_bp"] * patient.systolic_bp + coeffs["total_cholesterol"] * patient.total_cholesterol +
            coeffs["hdl_cholesterol"] * patient.hdl_cholesterol +
            coeffs["diabetes"] * float(patient.diabetes or False) + coeffs["bmi"] * bmi + coeffs["constant"]
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

        if "bmi" not in results.columns: results["bmi"] = 25.0
        if "diabetes" not in results.columns: results["diabetes"] = False

        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any(): continue

            coeffs = _JACKSON_COEFFICIENTS[sex]
            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"] + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float) +
                coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"] + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"] +
                coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"] + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float) +
                coeffs["bmi"] * results.loc[mask, "bmi"] + coeffs["constant"]
            )

            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        results["risk_score"] = np.clip(results["risk_score"], 0.0, 40.0)
        conditions = [results["risk_score"] < 5, (results["risk_score"] >= 5) & (results["risk_score"] < 10),
                     (results["risk_score"] >= 10) & (results["risk_score"] < 20), results["risk_score"] >= 20]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")
        return results
