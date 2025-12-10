"""
CARDIA (Coronary Artery Risk Development in Young Adults) Model.

CARDIA model for young adult cardiovascular disease risk prediction.

Reference:
    Rana JS, Tabada GH, Solomon MD, et al. Accuracy of the Atherosclerotic
    Cardiovascular Risk Equation in a Large Contemporary, Multiethnic
    Population. Journal of the American College of Cardiology. 2016;67(18):2118-2130.
"""

import numpy as np
import pandas as pd
from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

_CARDIA_COEFFICIENTS = {
    "male": {"age": 0.045, "smoking": 0.400, "systolic_bp": 0.010, "total_cholesterol": 0.012,
             "hdl_cholesterol": -0.025, "diabetes": 0.500, "bmi": 0.012, "constant": -7.500},
    "female": {"age": 0.040, "smoking": 0.380, "systolic_bp": 0.012, "total_cholesterol": 0.014,
               "hdl_cholesterol": -0.028, "diabetes": 0.480, "bmi": 0.014, "constant": -7.700}
}

class CARDIA(RiskModel):
    model_name = "CARDIA"
    model_version = "2015"
    supported_regions = ["US", "Young Adults"]

    def calculate(self, patient: PatientData) -> RiskResult:
        coeffs = _CARDIA_COEFFICIENTS[patient.sex]
        bmi = getattr(patient, 'bmi', 22.0)  # Lower default BMI for young adults

        linear_predictor = (
            coeffs["age"] * patient.age + coeffs["smoking"] * float(patient.smoking) +
            coeffs["systolic_bp"] * patient.systolic_bp + coeffs["total_cholesterol"] * patient.total_cholesterol +
            coeffs["hdl_cholesterol"] * patient.hdl_cholesterol +
            coeffs["diabetes"] * float(patient.diabetes or False) + coeffs["bmi"] * bmi + coeffs["constant"]
        )

        risk_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        risk_percentage = np.clip(risk_probability * 100.0, 0.0, 30.0)  # Lower risk for young adults

        risk_category = "low" if risk_percentage < 3 else "moderate" if risk_percentage < 7 else "high" if risk_percentage < 15 else "very_high"

        return RiskResult(risk_score=float(risk_percentage), risk_category=risk_category,
                         model_name=self.model_name, model_version=self.model_version)

    def calculate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = df.copy()
        results["risk_score"] = np.nan
        results["risk_category"] = ""
        results["model_name"] = self.model_name

        if "bmi" not in results.columns: results["bmi"] = 22.0
        if "diabetes" not in results.columns: results["diabetes"] = False

        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any(): continue

            coeffs = _CARDIA_COEFFICIENTS[sex]
            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"] + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float) +
                coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"] + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"] +
                coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"] + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float) +
                coeffs["bmi"] * results.loc[mask, "bmi"] + coeffs["constant"]
            )

            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        results["risk_score"] = np.clip(results["risk_score"], 0.0, 30.0)
        conditions = [results["risk_score"] < 3, (results["risk_score"] >= 3) & (results["risk_score"] < 7),
                     (results["risk_score"] >= 7) & (results["risk_score"] < 15), results["risk_score"] >= 15]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")
        return results
