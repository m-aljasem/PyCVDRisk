"""
Malaysian CVD Risk Calculator.

Malaysia has developed comprehensive cardiovascular risk assessment tools
adapted for its multi-ethnic population (Malay, Chinese, Indian).

Reference:
    Selvarajah S, Haniff J, Kaur G, et al. Validation of the Framingham
    general cardiovascular risk score in a multiethnic Asian population:
    the Malaysian cohort study. BMC Cardiovascular Disorders. 2017;17(1):1-9.
"""

import numpy as np
import pandas as pd
from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

# Malaysian CVD coefficients (adapted for multi-ethnic population)
_MALAYSIAN_COEFFICIENTS_MEN = {
    "age": 0.049,
    "smoking": 0.410,
    "systolic_bp": 0.013,
    "total_cholesterol": 0.011,
    "hdl_cholesterol": -0.025,
    "diabetes": 0.490,
    "bmi": 0.008,
    "ethnicity_chinese": -0.120,
    "ethnicity_indian": 0.180,
    "family_history": 0.260,
    "constant": -8.000
}

_MALAYSIAN_COEFFICIENTS_WOMEN = {
    "age": 0.046,
    "smoking": 0.390,
    "systolic_bp": 0.015,
    "total_cholesterol": 0.013,
    "hdl_cholesterol": -0.027,
    "diabetes": 0.470,
    "bmi": 0.010,
    "ethnicity_chinese": -0.110,
    "ethnicity_indian": 0.170,
    "family_history": 0.240,
    "constant": -8.200
}


class Malaysian_CVD(RiskModel):
    model_name = "Malaysian_CVD"
    model_version = "2017"
    supported_regions = ["Malaysia", "Asia"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)
        if patient.total_cholesterol is None or patient.hdl_cholesterol is None:
            raise ValueError("Malaysian CVD requires cholesterol measurements")

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        coeffs = _MALAYSIAN_COEFFICIENTS_MEN if patient.sex == "male" else _MALAYSIAN_COEFFICIENTS_WOMEN
        bmi = getattr(patient, 'bmi', 24.0)  # Asian average BMI
        ethnicity = getattr(patient, 'ethnicity', 'malay')

        # Ethnicity coefficients (Malay = reference)
        ethnicity_coeff = 0
        if ethnicity == "chinese":
            ethnicity_coeff = coeffs["ethnicity_chinese"]
        elif ethnicity == "indian":
            ethnicity_coeff = coeffs["ethnicity_indian"]

        linear_predictor = (
            coeffs["age"] * patient.age + coeffs["smoking"] * float(patient.smoking) +
            coeffs["systolic_bp"] * patient.systolic_bp + coeffs["total_cholesterol"] * patient.total_cholesterol +
            coeffs["hdl_cholesterol"] * patient.hdl_cholesterol +
            coeffs["diabetes"] * float(patient.diabetes or False) + coeffs["bmi"] * bmi +
            ethnicity_coeff + coeffs["family_history"] * float(getattr(patient, 'family_history', False)) + coeffs["constant"]
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

        if "bmi" not in results.columns: results["bmi"] = 24.0
        if "diabetes" not in results.columns: results["diabetes"] = False
        if "ethnicity" not in results.columns: results["ethnicity"] = "malay"
        if "family_history" not in results.columns: results["family_history"] = False

        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any(): continue

            coeffs = _MALAYSIAN_COEFFICIENTS_MEN if sex == 'male' else _MALAYSIAN_COEFFICIENTS_WOMEN

            # Ethnicity coefficients
            ethnicity_coeffs = np.zeros(len(results))
            ethnicity_chinese = (results.loc[mask, "ethnicity"] == "chinese")
            ethnicity_indian = (results.loc[mask, "ethnicity"] == "indian")

            ethnicity_coeffs[ethnicity_chinese] = coeffs["ethnicity_chinese"]
            ethnicity_coeffs[ethnicity_indian] = coeffs["ethnicity_indian"]

            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"] + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float) +
                coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"] + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"] +
                coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"] + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float) +
                coeffs["bmi"] * results.loc[mask, "bmi"] + ethnicity_coeffs + coeffs["family_history"] * results.loc[mask, "family_history"].astype(float) + coeffs["constant"]
            )

            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        results["risk_score"] = np.clip(results["risk_score"], 0.0, 35.0)
        conditions = [results["risk_score"] < 5, (results["risk_score"] >= 5) & (results["risk_score"] < 10),
                     (results["risk_score"] >= 10) & (results["risk_score"] < 15), results["risk_score"] >= 15]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")
        return results
