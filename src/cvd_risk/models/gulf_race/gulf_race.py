"""
Gulf RACE Study CVD Risk Calculator.

The Gulf RACE (Registry of Acute Coronary Events) study provides cardiovascular
risk assessment for Gulf Cooperation Council countries (Saudi Arabia, UAE,
Kuwait, Qatar, Bahrain, Oman).

Reference:
    Almahmeed W, Arnaout MS, Chettaoui R, et al. Coronary artery disease
    in the Gulf Cooperation Council countries: Prevalence and clinical
    features. Journal of the Saudi Heart Association. 2010;22(4):197-202.
"""

import numpy as np
import pandas as pd
from cvd_risk.core.base import RiskModel
from cvd_risk.core.validation import PatientData, RiskResult

# Gulf RACE coefficients (adapted for Gulf populations)
_GULF_RACE_COEFFICIENTS_MEN = {
    "age": 0.051,
    "smoking": 0.430,
    "systolic_bp": 0.014,
    "total_cholesterol": 0.012,
    "hdl_cholesterol": -0.026,
    "diabetes": 0.510,
    "bmi": 0.009,
    "family_history": 0.280,
    "country_saudi": 0.120,
    "country_uae": 0.080,
    "country_kuwait": 0.060,
    "country_qatar": 0.040,
    "country_bahrain": 0.020,
    "constant": -8.100
}

_GULF_RACE_COEFFICIENTS_WOMEN = {
    "age": 0.048,
    "smoking": 0.410,
    "systolic_bp": 0.016,
    "total_cholesterol": 0.014,
    "hdl_cholesterol": -0.028,
    "diabetes": 0.490,
    "bmi": 0.011,
    "family_history": 0.260,
    "country_saudi": 0.110,
    "country_uae": 0.075,
    "country_kuwait": 0.055,
    "country_qatar": 0.035,
    "country_bahrain": 0.015,
    "constant": -8.300
}


class Gulf_RACE(RiskModel):
    model_name = "Gulf_RACE"
    model_version = "2010"
    supported_regions = ["Gulf Countries", "Middle East"]

    def __init__(self) -> None:
        super().__init__()

    def validate_input(self, patient: PatientData) -> None:
        super().validate_input(patient)
        if patient.total_cholesterol is None or patient.hdl_cholesterol is None:
            raise ValueError("Gulf RACE requires cholesterol measurements")

    def calculate(self, patient: PatientData) -> RiskResult:
        self.validate_input(patient)

        coeffs = _GULF_RACE_COEFFICIENTS_MEN if patient.sex == "male" else _GULF_RACE_COEFFICIENTS_WOMEN
        bmi = getattr(patient, 'bmi', 28.0)  # Higher BMI in Gulf region
        country = getattr(patient, 'country', 'saudi')

        # Country coefficients (Saudi Arabia = reference)
        country_coeff = 0
        if hasattr(coeffs, f"country_{country.lower()}"):
            country_coeff = coeffs[f"country_{country.lower()}"]

        linear_predictor = (
            coeffs["age"] * patient.age + coeffs["smoking"] * float(patient.smoking) +
            coeffs["systolic_bp"] * patient.systolic_bp + coeffs["total_cholesterol"] * patient.total_cholesterol +
            coeffs["hdl_cholesterol"] * patient.hdl_cholesterol +
            coeffs["diabetes"] * float(patient.diabetes or False) + coeffs["bmi"] * bmi +
            coeffs["family_history"] * float(getattr(patient, 'family_history', False)) +
            country_coeff + coeffs["constant"]
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

        if "bmi" not in results.columns: results["bmi"] = 28.0
        if "diabetes" not in results.columns: results["diabetes"] = False
        if "family_history" not in results.columns: results["family_history"] = False
        if "country" not in results.columns: results["country"] = "saudi"

        for sex in ['male', 'female']:
            mask = results['sex'].str.lower() == sex
            if not mask.any(): continue

            coeffs = _GULF_RACE_COEFFICIENTS_MEN if sex == 'male' else _GULF_RACE_COEFFICIENTS_WOMEN

            # Country coefficients
            country_coeffs = np.zeros(len(results))
            for country in ['saudi', 'uae', 'kuwait', 'qatar', 'bahrain']:
                country_mask = (results.loc[mask, "country"].str.lower() == country)
                if f"country_{country}" in coeffs:
                    country_coeffs[country_mask] = coeffs[f"country_{country}"]

            linear_predictor = (
                coeffs["age"] * results.loc[mask, "age"] + coeffs["smoking"] * results.loc[mask, "smoking"].astype(float) +
                coeffs["systolic_bp"] * results.loc[mask, "systolic_bp"] + coeffs["total_cholesterol"] * results.loc[mask, "total_cholesterol"] +
                coeffs["hdl_cholesterol"] * results.loc[mask, "hdl_cholesterol"] + coeffs["diabetes"] * results.loc[mask, "diabetes"].astype(float) +
                coeffs["bmi"] * results.loc[mask, "bmi"] + coeffs["family_history"] * results.loc[mask, "family_history"].astype(float) +
                country_coeffs + coeffs["constant"]
            )

            risk_prob = 1.0 / (1.0 + np.exp(-linear_predictor))
            results.loc[mask, "risk_score"] = risk_prob * 100.0

        results["risk_score"] = np.clip(results["risk_score"], 0.0, 40.0)
        conditions = [results["risk_score"] < 5, (results["risk_score"] >= 5) & (results["risk_score"] < 10),
                     (results["risk_score"] >= 10) & (results["risk_score"] < 20), results["risk_score"] >= 20]
        choices = ["low", "moderate", "high", "very_high"]
        results["risk_category"] = np.select(conditions, choices, default="very_high")
        return results
