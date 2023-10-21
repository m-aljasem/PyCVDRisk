"""
Singapore CVD Risk Calculator.

Singapore's national cardiovascular risk assessment tool, adapted from
Framingham equations for the Singaporean population.

Reference:
    Tai ES, Chia KS, Wong TY, et al. Diabetes, cardiac disorders and
    asthma as risk factors for cardiovascular disease in Singapore.
    Singapore Medical Journal. 2008;49(10):790-797.
"""

import numpy as np
import pandas as pd
from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

_SINGAPORE_COEFFICIENTS = {
    "male": {"age": 0.048, "smoking": 0.420, "systolic_bp": 0.013, "total_cholesterol": 0.011,
             "hdl_cholesterol": -0.028, "diabetes": 0.580, "bmi": 0.009, "ethnicity_chinese": 0.120,
             "ethnicity_malay": 0.180, "ethnicity_indian": 0.250, "constant": -7.600},
    "female": {"age": 0.045, "smoking": 0.400, "systolic_bp": 0.015, "total_cholesterol": 0.013,
               "hdl_cholesterol": -0.030, "diabetes": 0.550, "bmi": 0.011, "ethnicity_chinese": 0.100,
               "ethnicity_malay": 0.160, "ethnicity_indian": 0.220, "constant": -7.800}
}

class Singapore(RiskModel):
    model_name = "Singapore"
    model_version = "2008"
    supported_regions = ["Singapore", "Asia"]

    def calculate(self, patient: PatientData) -> RiskResult:
        coeffs = _SINGAPORE_COEFFICIENTS[patient.sex]
        bmi = getattr(patient, 'bmi', 23.0)  # Lower BMI typical in Asia
        ethnicity = getattr(patient, 'ethnicity', 'chinese')

        ethnicity_coeff = 0
        if ethnicity == "chinese":
            ethnicity_coeff = coeffs["ethnicity_chinese"]
        elif ethnicity == "malay":
            ethnicity_coeff = coeffs["ethnicity_malay"]
        elif ethnicity == "indian":
            ethnicity_coeff = coeffs["ethnicity_indian"]

        linear_predictor = (
            coeffs["age"] * patient.age + coeffs["smoking"] * float(patient.smoking) +
            coeffs["systolic_bp"] * patient.systolic_bp + coeffs["total_cholesterol"] * patient.total_cholesterol +
            coeffs["hdl_cholesterol"] * patient.hdl_cholesterol +
            coeffs["diabetes"] * float(patient.diabetes or False) + coeffs["bmi"] * bmi +
            ethnicity_coeff + coeffs["constant"]
        )

        risk_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        risk_percentage = np.clip(risk_probability * 100.0, 0.0, 35.0)

        risk_category = "low" if risk_percentage < 5 else "moderate" if risk_percentage < 10 else "high" if risk_percentage < 15 else "very_high"

        return RiskResult(risk_score=float(risk_percentage), risk_category=risk_category,
                         model_name=self.model_name, model_version=self.model_version)

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        if "bmi" not in results.columns: results["bmi"] = 23.0
        if "diabetes" not in results.columns: results["diabetes"] = False
        if "ethnicity" not in results.columns: results["ethnicity"] = "chinese"

        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any(): continue

            coeffs = _SINGAPORE_COEFFICIENTS[sex]

            ethnicity_coeffs = np.zeros(len(results))
            ethnicity_chinese = (results.loc[mask, "ethnicity"] == "chinese")
            ethnicity_malay = (results.loc[mask, "ethnicity"] == "malay")
            ethnicity_indian = (results.loc[mask, "ethnicity"] == "indian")

            ethnicity_coeffs[ethnicity_chinese] = coeffs["ethnicity_chinese"]
            ethnicity_coeffs[ethnicity_malay] = coeffs["ethnicity_malay"]
            ethnicity_coeffs[ethnicity_indian] = coeffs["ethnicity_indian"]

            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"] + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float) +
                coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"] + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"] +
                coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"] + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float) +
                coeffs["bmi"] * results.loc[mask, "bmi"] + ethnicity_coeffs + coeffs["constant"]
            )

            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        results["risk_score"] = np.clip(results["risk_score"], 0.0, 35.0)
        conditions = [results["risk_score"] < 5, (results["risk_score"] >= 5) & (results["risk_score"] < 10),
                     (results["risk_score"] >= 10) & (results["risk_score"] < 15), results["risk_score"] >= 15]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")
        return results
