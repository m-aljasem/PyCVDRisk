"""
Heinz Nixdorf Recall Study Risk Calculator.

The Heinz Nixdorf Recall study is a large prospective German cohort study
that developed comprehensive cardiovascular risk prediction models.

Reference:
    Erbel R, Lehmann N, Moebus S, et al. Subclinical coronary atherosclerosis
    and risk for myocardial infarction in a prospective cohort study. European
    Heart Journal. 2011;32(22):2784-2792.
"""

import numpy as np
import pandas as pd
from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

_HEINZ_NIXDORF_COEFFICIENTS = {
    "male": {"age": 0.059, "smoking": 0.480, "systolic_bp": 0.012, "total_cholesterol": 0.010,
             "hdl_cholesterol": -0.025, "diabetes": 0.520, "bmi": 0.008, "education": -0.150, "constant": -7.800},
    "female": {"age": 0.055, "smoking": 0.450, "systolic_bp": 0.014, "total_cholesterol": 0.012,
               "hdl_cholesterol": -0.028, "diabetes": 0.500, "bmi": 0.010, "education": -0.140, "constant": -7.900}
}

class HeinzNixdorf(RiskModel):
    model_name = "HeinzNixdorf"
    model_version = "2010"
    supported_regions = ["Germany"]

    def calculate(self, patient: PatientData) -> RiskResult:
        coeffs = _HEINZ_NIXDORF_COEFFICIENTS[patient.sex]
        bmi = getattr(patient, 'bmi', 25.0)
        education_high = getattr(patient, 'education_high', False)

        linear_predictor = (
            coeffs["age"] * patient.age + coeffs["smoking"] * float(patient.smoking) +
            coeffs["systolic_bp"] * patient.systolic_bp + coeffs["total_cholesterol"] * patient.total_cholesterol +
            coeffs["hdl_cholesterol"] * patient.hdl_cholesterol +
            coeffs["diabetes"] * float(patient.diabetes or False) + coeffs["bmi"] * bmi +
            coeffs["education"] * float(education_high) + coeffs["constant"]
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
        if "education_high" not in results.columns: results["education_high"] = False

        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any(): continue

            coeffs = _HEINZ_NIXDORF_COEFFICIENTS[sex]
            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"] + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float) +
                coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"] + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"] +
                coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"] + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float) +
                coeffs["bmi"] * results.loc[mask, "bmi"] + coeffs["education"] * results.loc[mask, "education_high"].astype(float) + coeffs["constant"]
            )

            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        results["risk_score"] = np.clip(results["risk_score"], 0.0, 40.0)
        conditions = [results["risk_score"] < 5, (results["risk_score"] >= 5) & (results["risk_score"] < 10),
                     (results["risk_score"] >= 10) & (results["risk_score"] < 20), results["risk_score"] >= 20]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")
        return results
