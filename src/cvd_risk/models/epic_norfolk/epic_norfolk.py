"""
EPIC-Norfolk Risk Calculator.

EPIC-Norfolk is a prospective population study in Norfolk, UK, that developed
comprehensive cardiovascular risk prediction models.

Reference:
    Hippisley-Cox J, Coupland C, Robson J, et al. Predicting risk of type 2
    diabetes in England and Wales: prospective derivation and validation of
    QDScore. BMJ. 2009;338:b880.
"""

import numpy as np
import pandas as pd
from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

_EPIC_NORFOLK_COEFFICIENTS = {
    "male": {"age": 0.052, "smoking": 0.450, "systolic_bp": 0.011, "total_cholesterol": 0.009,
             "hdl_cholesterol": -0.022, "diabetes": 0.480, "bmi": 0.008, "alcohol": -0.005, "physical_activity": -0.150, "constant": -7.200},
    "female": {"age": 0.048, "smoking": 0.420, "systolic_bp": 0.013, "total_cholesterol": 0.011,
               "hdl_cholesterol": -0.025, "diabetes": 0.460, "bmi": 0.010, "alcohol": -0.004, "physical_activity": -0.140, "constant": -7.400}
}

class EPIC_Norfolk(RiskModel):
    model_name = "EPIC_Norfolk"
    model_version = "2007"
    supported_regions = ["UK", "Europe"]

    def calculate(self, patient: PatientData) -> RiskResult:
        coeffs = _EPIC_NORFOLK_COEFFICIENTS[patient.sex]
        bmi = getattr(patient, 'bmi', 25.0)
        alcohol_units = getattr(patient, 'alcohol_units', 10)  # Default moderate drinking
        physical_activity_high = getattr(patient, 'physical_activity_high', False)

        linear_predictor = (
            coeffs["age"] * patient.age + coeffs["smoking"] * float(patient.smoking) +
            coeffs["systolic_bp"] * patient.systolic_bp + coeffs["total_cholesterol"] * patient.total_cholesterol +
            coeffs["hdl_cholesterol"] * patient.hdl_cholesterol +
            coeffs["diabetes"] * float(patient.diabetes or False) + coeffs["bmi"] * bmi +
            coeffs["alcohol"] * alcohol_units + coeffs["physical_activity"] * float(physical_activity_high) + coeffs["constant"]
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
        if "alcohol_units" not in results.columns: results["alcohol_units"] = 10
        if "physical_activity_high" not in results.columns: results["physical_activity_high"] = False

        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any(): continue

            coeffs = _EPIC_NORFOLK_COEFFICIENTS[sex]
            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"] + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float) +
                coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"] + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"] +
                coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"] + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float) +
                coeffs["bmi"] * results.loc[mask, "bmi"] + coeffs["alcohol"] * results.loc[mask, "alcohol_units"] +
                coeffs["physical_activity"] * results.loc[mask, "physical_activity_high"].astype(float) + coeffs["constant"]
            )

            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        results["risk_score"] = np.clip(results["risk_score"], 0.0, 40.0)
        conditions = [results["risk_score"] < 5, (results["risk_score"] >= 5) & (results["risk_score"] < 10),
                     (results["risk_score"] >= 10) & (results["risk_score"] < 20), results["risk_score"] >= 20]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")
        return results
