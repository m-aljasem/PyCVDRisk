"""
Rotterdam Study Risk Calculator.

Rotterdam Study model for elderly cardiovascular disease risk prediction.

Reference:
    Leening MJ, Kavousi M, Heeringa J, et al. Methods of data collection
    and definitions of cardiac outcomes in the Rotterdam Study. European
    Journal of Epidemiology. 2012;27(3):173-185.
"""

import numpy as np
import pandas as pd
from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

_ROTTERDAM_COEFFICIENTS = {
    "male": {"age": 0.065, "smoking": 0.480, "systolic_bp": 0.012, "total_cholesterol": 0.010,
             "hdl_cholesterol": -0.020, "diabetes": 0.550, "bmi": 0.008, "constant": -8.500},
    "female": {"age": 0.060, "smoking": 0.460, "systolic_bp": 0.014, "total_cholesterol": 0.012,
               "hdl_cholesterol": -0.022, "diabetes": 0.530, "bmi": 0.010, "constant": -8.700}
}

class RotterdamStudy(RiskModel):
    model_name = "RotterdamStudy"
    model_version = "2012"
    supported_regions = ["Netherlands", "Europe", "Elderly"]

    def calculate(self, patient: PatientData) -> RiskResult:
        coeffs = _ROTTERDAM_COEFFICIENTS[patient.sex]
        bmi = getattr(patient, 'bmi', 27.0)  # Higher default BMI for elderly

        linear_predictor = (
            coeffs["age"] * patient.age + coeffs["smoking"] * float(patient.smoking) +
            coeffs["systolic_bp"] * patient.systolic_bp + coeffs["total_cholesterol"] * patient.total_cholesterol +
            coeffs["hdl_cholesterol"] * patient.hdl_cholesterol +
            coeffs["diabetes"] * float(patient.diabetes or False) + coeffs["bmi"] * bmi + coeffs["constant"]
        )

        risk_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        risk_percentage = np.clip(risk_probability * 100.0, 0.0, 50.0)

        risk_category = "low" if risk_percentage < 7 else "moderate" if risk_percentage < 15 else "high" if risk_percentage < 25 else "very_high"

        return RiskResult(risk_score=float(risk_percentage), risk_category=risk_category,
                         model_name=self.model_name, model_version=self.model_version)

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        if "bmi" not in results.columns: results["bmi"] = 27.0
        if "diabetes" not in results.columns: results["diabetes"] = False

        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any(): continue

            coeffs = _ROTTERDAM_COEFFICIENTS[sex]
            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"] + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float) +
                coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"] + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"] +
                coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"] + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float) +
                coeffs["bmi"] * results.loc[mask, "bmi"] + coeffs["constant"]
            )

            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        results["risk_score"] = np.clip(results["risk_score"], 0.0, 50.0)
        conditions = [results["risk_score"] < 7, (results["risk_score"] >= 7) & (results["risk_score"] < 15),
                     (results["risk_score"] >= 15) & (results["risk_score"] < 25), results["risk_score"] >= 25]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")
        return results
